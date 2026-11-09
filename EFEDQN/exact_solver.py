"""Gurobi exact MILP solver for the time-windowed target-collection routing problem.

Model semantics mirror Env.py exactly:
- Walk starts at node 0 at time 0.0, must visit every target node and return to node 0.
- Traversing directed arc (i, j) costs map[i][slot][1] and takes map[i][slot][2] minutes;
  clock time is taken modulo 24 h.
- If arc (i, j) carries a restricted window [s, e] (closed interval, wrap-around when s > e),
  departing on that arc while the clock lies inside the window is a violation.
- A feasible rollout must finish within `max_steps` moves (evaluate.py protocol: 49).

The problem is written as a pure linear (mixed-integer) program and solved with
Gurobi. Time is modelled in integer minutes. Windows are quantised to minute
precision by the instance generator, so the conversion below is exact for the
benchmark pools.

Decision variables (all integer; every constraint is linear):
- x[k, a]  binary: arc a is traversed as the k-th move; the arc set includes a
           zero-cost / zero-duration "wait at depot" self-loop (0 -> 0);
- c[k]     integer clock in [0, 1439] minutes at the moment step k begins;
- d[k]     integer number of midnights crossed by the k-th move, giving the
           exact minute-modulo recurrence
               c[k+1] = c[k] + dur(x[k, .]) - 1440 * d[k]
           (d[k] is uniquely determined because 0 <= c <= 1439, so unlike the
           classic big-M time propagation no relaxation slack is introduced
           here).

Linear constraints:
- exactly one arc per step, flow conservation between consecutive steps (this
  keeps the walk connected and anchored at the depot), arrival at the depot at
  the horizon end;
- every target is entered by at least one move;
- restricted windows: x[k, a] = 1 => c[k] outside the closed interval [s, e].
  Encoding uses big-M constants of 1440 (the full clock range). Wrapping
  windows (s > e) leave the single conjunction e+1 <= c[k] <= s-1; ordinary
  windows add one disjunction binary per arc/step.

The walk is padded after completion with the wait arc so that a fixed horizon
of `max_steps` steps can be used. Since every real arc has strictly positive cost,
any optimal model solution never moves after completion, hence the model optimum
equals the environment optimum; a linear symmetry-breaking constraint forces the
waits to form a trailing block once the walk is back at the depot (a mid-walk
wait can always be removed without changing cost or clock time). Every solution
is additionally replayed in Sample_Env as a guard.
"""

import time
from collections import deque, defaultdict

import gurobipy as gp
from gurobipy import GRB

from Env import Sample_Env, is_edge_active

MINUTES_PER_DAY = 1440

_ENV = None


def _gurobi_env():
    """Shared silent Gurobi environment (created lazily, reused across solves)."""
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(params={"OutputFlag": 0})
    return _ENV


def _bfs_dist(n, arcs, source, forward=True):
    """Hop distance from the depot (forward) or to the depot (backward); None = unreachable."""
    adj = [[] for _ in range(n)]
    for (i, j, _, _) in arcs:
        (adj[i] if forward else adj[j]).append(j if forward else i)
    dist = [None] * n
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def env_to_data(env):
    """Extract (arcs, windows, targets) from a Sample_Env.

    arcs: list of (i, j, length, duration_min), one per valid map slot.
    windows: dict (i, j) -> (start_min, end_min) restricted window (closed interval).
    """
    arcs = []
    seen = set()
    for i in range(env.space_size):
        for k in range(env.action_size):
            tgt, length, dur = env.map[i][k]
            if tgt == -1:
                continue
            if (i, tgt) in seen:
                raise ValueError(f"parallel arc {i}->{tgt}: solver assumes unique arcs")
            seen.add((i, tgt))
            arcs.append((i, tgt, int(length), int(dur)))

    windows = {}
    for (u, v, s, e) in env.edge_limits:
        s_min = int(round(float(s) * 60))
        e_min = int(round(float(e) * 60))
        key = (u, v)
        if key in windows and windows[key] != (s_min, e_min):
            raise ValueError(f"multiple windows on arc {u}->{v}: not supported")
        windows[key] = (s_min, e_min)

    targets = sorted(set(env.goal_states1))
    return arcs, windows, targets


def _forbid_window(model, arc_var, clock, s, e):
    """arc_var = 1 => clock outside the closed window [s, e] (wrap-around when s > e).

    Pure linear big-M encoding with M = 1440, the whole clock domain:
    - s <= e: forbidden iff s <= clock <= e, i.e. clock <= s-1 OR clock >= e+1;
      one extra binary selects the side of the disjunction.
    - s >  e: forbidden iff clock >= s or clock <= e, i.e. the single
      conjunction e+1 <= clock <= s-1; no extra binary is needed.
    """
    m = MINUTES_PER_DAY
    if s <= e:
        side = model.addVar(vtype=GRB.BINARY)          # 1: leave before the window
        model.addConstr(clock <= (s - 1) + m * (1 - side) + m * (1 - arc_var))
        model.addConstr(clock >= (e + 1) - m * side - m * (1 - arc_var))
    else:
        model.addConstr(clock <= (s - 1) + m * (1 - arc_var))
        model.addConstr(clock >= (e + 1) - m * (1 - arc_var))


def build_model(env, max_steps=49):
    """Build the pure-linear Gurobi MILP for one Sample_Env.

    Returns (model, x, allowed, all_arcs): x maps (step, arc index) to a binary
    variable, allowed[k] lists the arc indices usable at step k after
    reachability pruning, all_arcs is the arc list plus the depot wait loop.
    """
    arcs, windows, targets = env_to_data(env)
    n = env.space_size
    K = max_steps

    # wait arc at the depot: (0 -> 0), zero cost / zero duration, no window
    all_arcs = arcs + [(0, 0, 0, 0)]
    wait_idx = len(all_arcs) - 1

    model = gp.Model("exact_milp", env=_gurobi_env())
    model.Params.OutputFlag = 0

    # reachability pruning: arc (i, j) can be the k-th move only if node i is
    # at least dist0[i] hops from the depot and the depot is at most K-1-k hops
    # away from j (valid inequalities: no genuine walk is excluded)
    dist0 = _bfs_dist(n, arcs, 0, forward=True)
    distb = _bfs_dist(n, arcs, 0, forward=False)
    allowed = [[wait_idx] for _ in range(K)]
    for a, (i, j, _, _) in enumerate(arcs):
        if dist0[i] is None or distb[j] is None:
            continue
        for k in range(dist0[i], K - distb[j]):
            allowed[k].append(a)

    # variables
    x = model.addVars([(k, a) for k in range(K) for a in allowed[k]],
                      vtype=GRB.BINARY)
    max_dur = max((d for (_, _, _, d) in arcs), default=0)
    day_hi = (MINUTES_PER_DAY - 1 + max_dur) // MINUTES_PER_DAY
    clock = model.addVars(range(K + 1), vtype=GRB.INTEGER,
                          lb=0, ub=MINUTES_PER_DAY - 1)
    days = model.addVars(range(K), vtype=GRB.INTEGER, lb=0, ub=day_hi)

    # per-step incidence lists over the pruned arc set
    out_map = [defaultdict(list) for _ in range(K)]
    in_map = [defaultdict(list) for _ in range(K)]
    for k in range(K):
        for a in allowed[k]:
            i, j, _, _ = all_arcs[a]
            out_map[k][i].append(a)
            in_map[k][j].append(a)

    # exactly one arc (real traversal or depot self-loop) per step
    for k in range(K):
        model.addConstr(gp.quicksum(x[k, a] for a in allowed[k]) == 1)

    # flow conservation between consecutive steps: the head of move k-1 is the
    # tail of move k, so the walk stays connected; step 0 only offers arcs
    # leaving the depot, which anchors the start at node 0
    for k in range(1, K):
        for u in range(n):
            model.addConstr(
                gp.quicksum(x[k, a] for a in out_map[k].get(u, ()))
                == gp.quicksum(x[k - 1, a] for a in in_map[k - 1].get(u, ())))

    # be back at the depot at the horizon end
    if K:
        model.addConstr(
            gp.quicksum(x[K - 1, a] for a in in_map[K - 1].get(0, ())) == 1)

    # every target is visited at least once (arrival by some move)
    for t in targets:
        model.addConstr(gp.quicksum(
            x[k, a] for k in range(K) for a in in_map[k].get(t, ())) >= 1)

    # zero initial clock and exact minute-modulo time propagation
    model.addConstr(clock[0] == 0)
    for k in range(K):
        model.addConstr(
            clock[k + 1] == clock[k]
            + gp.quicksum(all_arcs[a][3] * x[k, a] for a in allowed[k])
            - MINUTES_PER_DAY * days[k])

    # restricted windows on the departure clock of windowed arcs
    for k in range(K):
        for a in allowed[k]:
            w = windows.get((all_arcs[a][0], all_arcs[a][1]))
            if w is not None:
                _forbid_window(model, x[k, a], clock[k], w[0], w[1])

    # symmetry breaking: once the walk waits at the depot it only keeps waiting
    # (a mid-walk wait arc can always be removed without changing cost or time)
    for k in range(K - 1):
        model.addConstr(x[k, wait_idx] <= x[k + 1, wait_idx])

    model.setObjective(
        gp.quicksum(all_arcs[a][2] * x[k, a]
                     for k in range(K) for a in allowed[k]),
        GRB.MINIMIZE)

    return model, x, allowed, all_arcs


def extract_path(model, x, allowed, all_arcs, K):
    """Node sequence with wait arcs stripped (the only self-transition is the wait arc,
    so dropping consecutive duplicates removes exactly the wait steps)."""
    nodes = [0]
    for k in range(K):
        for a in allowed[k]:
            if x[k, a].X > 0.5:
                nodes.append(all_arcs[a][1])
                break
    path = [nodes[0]]
    for v in nodes[1:]:
        if v != path[-1]:
            path.append(v)
    return path


def replay_path(env, path, max_steps=49):
    """Replay a node sequence in Sample_Env; returns None if the walk is invalid."""
    if not path or path[0] != 0 or path[-1] != 0:
        return None
    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)

    cost = 0
    violations = [0, 0]
    steps = 0
    for t in range(len(path) - 1):
        i, j = path[t], path[t + 1]
        act = next((k for k in range(env.action_size) if env.map[i][k][0] == j), None)
        if act is None:
            return None
        if is_edge_active(env.edge_limits, env.agent_time, i, j):
            violations[1] += 1
        cost += env.map[i][act][1]
        env.agent_pos = j
        env.agent_time = (env.agent_time + env.map[i][act][2] / 60.0) % 24.0
        if j in env.goal_states:
            env.goal_states.remove(j)
        steps += 1

    done = (len(env.goal_states) == 0 and env.agent_pos == env.start_pos)
    feasible = done and violations == [0, 0] and steps <= max_steps
    return {"cost": cost, "violations": violations, "steps": steps,
            "done": done, "feasible": feasible}


def solve_env(env, time_limit=60.0, workers=8, max_steps=49, seed=0):
    """Solve one Sample_Env to optimality (or certified bound on timeout).

    Gurobi statuses are mapped to the historical protocol of this module:
    OPTIMAL / INFEASIBLE keep their meaning, a timeout (or any other stop) with
    an incumbent becomes FEASIBLE, without one UNKNOWN.

    Returns dict with: status, cost, bound, wall_time, path, replay info.
    """
    t0 = time.perf_counter()
    model, x, allowed, all_arcs = build_model(env, max_steps=max_steps)

    model.Params.OutputFlag = 0
    model.Params.Threads = int(workers)
    model.Params.TimeLimit = float(time_limit)
    model.Params.Seed = int(seed) % (2 ** 31)
    model.Params.MIPGap = 0.0          # costs are integral: prove true optimality
    model.optimize()
    wall = time.perf_counter() - t0

    if model.Status == GRB.OPTIMAL:
        status_name = "OPTIMAL"
    elif model.Status == GRB.INFEASIBLE:
        status_name = "INFEASIBLE"
    elif model.SolCount > 0:
        status_name = "FEASIBLE"
    else:
        status_name = "UNKNOWN"

    result = {
        "status": status_name,
        "cost": None,
        "bound": None,
        "gap_to_bound": None,
        "wall_time": wall,
        "path": None,
        "replay_feasible": None,
        "replay_cost": None,
        "replay_match": None,
    }

    if status_name in ("OPTIMAL", "FEASIBLE"):
        result["cost"] = model.ObjVal
        bound = model.ObjBound
        if bound > -1e30:              # finite dual bound (else: none established)
            result["bound"] = bound
        if result["bound"] is not None and result["bound"] > 0:
            result["gap_to_bound"] = 100.0 * (result["cost"] - result["bound"]) / result["bound"]
        path = extract_path(model, x, allowed, all_arcs, max_steps)
        result["path"] = path
        rep = replay_path(env, path, max_steps=max_steps)
        if rep is None:
            result["replay_feasible"] = False
            result["replay_match"] = False
        else:
            result["replay_feasible"] = rep["feasible"]
            result["replay_cost"] = rep["cost"]
            result["replay_match"] = (rep["feasible"]
                                      and abs(rep["cost"] - result["cost"]) < 1e-6)
    elif status_name == "INFEASIBLE":
        result["bound"] = None
    elif status_name == "UNKNOWN":
        pass

    return result


def certify_env(env, time_limit=60.0, workers=8, max_steps=49):
    """True iff a feasible walk within max_steps is found (OPTIMAL or FEASIBLE)."""
    res = solve_env(env, time_limit=time_limit, workers=workers, max_steps=max_steps)
    return res["status"] in ("OPTIMAL", "FEASIBLE"), res


if __name__ == "__main__":
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    env = Sample_Env()
    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)
    try:
        res = solve_env(env, time_limit=60.0)
    except gp.GurobiError as exc:
        raise SystemExit(
            f"Gurobi error: {exc}\n"
            "(the bundled restricted-use license caps models at 2000 variables "
            "and 2000 constraints; this 26-node demo instance exceeds that cap)")
    print({k: v for k, v in res.items() if k != "path"})
    print("path:", res["path"])
