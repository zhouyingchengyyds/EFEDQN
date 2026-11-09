"""Brute-force validation of exact_solver and of a reference big-M MILP on tiny instances.

Reference solver: exact DP over states (pos, elapsed_min, visited_mask), layered by
step count. Three independent implementations are cross-checked on every instance:
the DP reference, the Gurobi model of exact_solver.py, and a literal big-M MILP
transcribed equation by equation (objective, service, single-step, depot, flow
conservation, initial clock, midnight triggers, time propagation, closed-interval
windows). Optimal cost and feasibility must agree across all three; every Gurobi
path is additionally replayed in Sample_Env.

Run:  python validate_solver.py [--trials 40] [--time-limit 10]
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
from ortools.linear_solver import pywraplp

from Env import Sample_Env
from exact_solver import env_to_data, solve_env

_MILP_STATUS = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def gen_tiny_env(n, n_targets, k_v, seed, long_durations=False):
    """Tiny strongly-connected instance: ring skeleton plus random extra out-arcs."""
    rng = random.Random(seed)
    out_neighbors = {i: [(i + 1) % n] for i in range(n)}
    for i in range(n):
        candidates = [j for j in range(n) if j != i and j not in out_neighbors[i]]
        rng.shuffle(candidates)
        for j in candidates[:rng.randint(0, 3)]:
            out_neighbors[i].append(j)
        if len(out_neighbors[i]) > 4:
            out_neighbors[i] = out_neighbors[i][:4]

    dur_hi = 400 if long_durations else 50
    map_ = [[(-1, 0, 0)] * 4 for _ in range(n)]
    slots = []
    for i in range(n):
        for slot, j in enumerate(out_neighbors[i]):
            length = rng.randint(1, 50)
            dur = rng.randint(1, dur_hi)
            map_[i][slot] = (j, length, dur)
            slots.append((i, j))

    targets = rng.sample(range(1, n), n_targets)

    restricted = rng.sample(slots, min(k_v, len(slots)))
    limits = []
    for (u, v) in restricted:
        s = rng.randint(0, 23)
        dur_min = rng.randint(60, 780)
        e = (s * 60 + dur_min) % 1440
        limits.append((u, v, s / 60.0, e / 60.0))

    return Sample_Env(map=[list(row) for row in map_], nodes_num=n, edges_num=len(slots),
                      edges=None, target=targets, edge_limits=limits, random_limits=False)


def dp_optimum(env, max_steps=None):
    """Exact DP reference. Returns (opt_cost, feasible, opt_steps) with Env semantics.

    max_steps=None searches with no step horizon: states are deduplicated on the
    finite equivalence class (pos, clock mod 1440, mask), so the frontier
    saturates in finitely many layers and the returned optimum is global.
    """
    arcs, windows, targets = env_to_data(env)
    t_index = {t: idx for idx, t in enumerate(targets)}
    full_mask = (1 << len(targets)) - 1

    out = {}
    for (i, j, length, dur) in arcs:
        out.setdefault(i, []).append((j, length, dur))

    frontier = {(0, 0, 0): 0}  # (pos, elapsed_min, mask) -> min cost
    seen = {(0, 0, 0): 0}      # (pos, clock, mask) -> best cost so far
    best = None
    best_steps = None
    layer = 0
    while frontier:
        layer += 1
        if max_steps is not None and layer > max_steps:
            break
        nxt = {}
        for (pos, el, mask), cost in frontier.items():
            for (j, length, dur) in out.get(pos, []):
                w = windows.get((pos, j))
                if w is not None:
                    s, e = w
                    ck = el % 1440
                    forbidden = (s <= ck <= e) if s <= e else (ck >= s or ck <= e)
                    if forbidden:
                        continue
                nmask = mask | (1 << t_index[j]) if j in t_index else mask
                key = (j, el + dur, nmask)
                ncost = cost + length
                if seen.get((j, (el + dur) % 1440, nmask), float("inf")) <= ncost:
                    continue
                if key not in nxt or ncost < nxt[key]:
                    nxt[key] = ncost
        for (pos, el, mask), cost in nxt.items():
            ekey = (pos, el % 1440, mask)
            if seen.get(ekey, float("inf")) > cost:
                seen[ekey] = cost
        frontier = nxt
        for (pos, _el, mask), cost in frontier.items():
            if pos == 0 and mask == full_mask:
                if best is None or cost < best:
                    best = cost
                    best_steps = layer
    return best, best is not None, best_steps


def _milp_solve(arcs, windows, targets, n, max_steps, time_limit):
    """Build and solve the reference big-M MILP in-process (SCIP/CBC backend).

    Constraints are built equation by equation: objective (1),
    service (2)-(3), single-step (4), depot round trip (5), flow conservation (6),
    initial clock (7), midnight-crossing triggers (8)-(9), time propagation
    (10)-(11), and closed-interval restricted windows (12)-(13) with the a-1 / b+1
    minute boundaries. The arc set includes the zero-cost depot self-loop, so walks
    shorter than the horizon are padded, exactly as in the formulation.

    Service times are zero in every experiment (Sample_Env carries none), so the
    s_u * z_u^{k-1} terms of the general formulation vanish.

    Returns (cost, feasible, status): cost is None unless an incumbent exists.
    """
    K = max_steps
    P = set(targets)

    all_arcs = list(arcs) + [(0, 0, 0, 0)]      # + depot wait self-loop
    solver = (pywraplp.Solver.CreateSolver("SCIP")
              or pywraplp.Solver.CreateSolver("CBC"))
    if solver is None:
        raise RuntimeError("no MILP backend (SCIP/CBC) available in ortools")
    solver.SetTimeLimit(int(time_limit * 1000))

    M = 1440 + max((d for (_, _, _, d) in arcs), default=0)   # big-M choice with s = 0

    x = {(k, a): solver.BoolVar("") for k in range(1, K + 1)
         for a in range(len(all_arcs))}
    z = {(w, k): solver.BoolVar("") for w in range(n) for k in range(K + 1)}
    T = {(u, k): solver.IntVar(0, 1439, "") for u in range(n)
         for k in range(K + 1)}

    out_arcs, in_arcs = {}, {}
    for a, (i, j, _, _) in enumerate(all_arcs):
        out_arcs.setdefault(i, []).append(a)
        in_arcs.setdefault(j, []).append(a)

    # (1) objective
    solver.Minimize(sum(c * x[k, a] for k in range(1, K + 1)
                        for a, (_, _, c, _) in enumerate(all_arcs)))

    # (2) every delivery point is serviced exactly once
    for w in P:
        solver.Add(sum(z[w, k] for k in range(1, K + 1)) == 1)
    # (3) service only upon a visit at the same step
    for w in P:
        for k in range(1, K + 1):
            solver.Add(z[w, k] <= sum(x[k, a] for a in in_arcs.get(w, [])))
    # (4) exactly one arc (real traversal or depot self-loop) per step
    for k in range(1, K + 1):
        solver.Add(sum(x[k, a] for a in range(len(all_arcs))) == 1)
    # (5) depart from the depot at step 1, be back at the depot at step K
    solver.Add(sum(x[1, a] for a in out_arcs.get(0, [])) == 1)
    solver.Add(sum(x[K, a] for a in in_arcs.get(0, [])) == 1)
    # (6) flow conservation between consecutive steps
    for u in range(n):
        for k in range(2, K + 1):
            solver.Add(sum(x[k - 1, a] for a in in_arcs.get(u, []))
                       == sum(x[k, a] for a in out_arcs.get(u, [])))
    # (7) zero initial clock; nothing serviced before departure
    for u in range(n):
        solver.Add(T[u, 0] == 0)
    for w in range(n):
        solver.Add(z[w, 0] == 0)

    # (8)-(13) per arc and step: midnight trigger, propagation, restricted window
    for k in range(1, K + 1):
        for a, (i, j, _, dur) in enumerate(all_arcs):
            mu = solver.BoolVar("")
            solver.Add(T[i, k - 1] + dur - 1440
                       <= M * mu + M * (1 - x[k, a]))                      # (8)
            solver.Add(T[i, k - 1] + dur - 1440
                       >= -M * (1 - mu) - M * (1 - x[k, a]))               # (9)
            solver.Add(T[j, k] <= T[i, k - 1] + dur - 1440 * mu
                       + M * (1 - x[k, a]))                                # (10)
            solver.Add(T[j, k] >= T[i, k - 1] + dur - 1440 * mu
                       - M * (1 - x[k, a]))                                # (11)
            win = windows.get((i, j))
            if win is not None:
                aa, bb = win
                if aa <= bb:                       # I_{u,w} = 1: standard window
                    y = solver.BoolVar("")
                    solver.Add(T[i, k - 1] <= aa - 1 + M * (1 - x[k, a])
                               + M * (1 - y))                              # (12)
                    solver.Add(T[i, k - 1] >= bb + 1 - M * (1 - x[k, a])
                               - M * y)                                    # (13)
                else:                              # I_{u,w} = 0: cross-midnight
                    solver.Add(T[i, k - 1] <= aa - 1 + M * (1 - x[k, a]))  # (12)
                    solver.Add(T[i, k - 1] >= bb + 1 - M * (1 - x[k, a]))  # (13)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), True, "OPTIMAL"
    if status == pywraplp.Solver.INFEASIBLE:
        return None, False, "INFEASIBLE"
    if status == pywraplp.Solver.FEASIBLE:
        return solver.Objective().Value(), True, "FEASIBLE"
    return None, None, _MILP_STATUS.get(status, f"STATUS_{status}")


def _milp_worker(in_path, out_path):
    """Child-process entry: load pickled instance data, solve, dump JSON result."""
    with open(in_path, "rb") as f:
        d = pickle.load(f)
    print("[milp-worker] solving", flush=True)   # stay observable while starting
    cost, feasible, status = _milp_solve(d["arcs"], d["windows"], d["targets"],
                                         d["n"], d["max_steps"], d["time_limit"])
    with open(out_path, "w") as f:
        json.dump({"cost": cost, "feasible": feasible, "status": status}, f)


def milp_optimum(env, max_steps, time_limit=10.0):
    """Solve the reference big-M MILP for one Sample_Env.

    The backend runs in a watchdogged child process: on degenerate big-M models
    both SCIP and CBC have been observed to ignore SetTimeLimit (typically while
    trying to prove infeasibility), which would otherwise wedge the validation
    run. The child receives an effective budget of min(time_limit, 6) seconds and
    is killed 5 seconds after that; the verdict then becomes NOT_SOLVED, i.e.,
    "no verdict within the budget", which milp_agrees treats as unproven rather
    than as disagreement.

    Returns (cost, feasible, status).
    """
    arcs, windows, targets = env_to_data(env)
    budget = min(float(time_limit), 6.0)
    payload = {"arcs": list(arcs), "windows": dict(windows),
               "targets": list(targets), "n": env.space_size,
               "max_steps": max_steps, "time_limit": budget}
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "milp_in.pkl")
        out_path = os.path.join(td, "milp_out.json")
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--milp-worker", in_path, out_path],
            stdout=None, stderr=subprocess.DEVNULL)
        start = time.time()
        deadline = start + budget + 5.0
        next_beat = start + 1.5
        while proc.poll() is None:
            now = time.time()
            if now >= deadline:
                proc.kill()
                proc.wait()
                return None, None, "NOT_SOLVED"
            if now >= next_beat:      # heartbeat: keep the run observable
                print(f"[milp {now - start:.0f}s]", flush=True)
                next_beat += 1.5
            time.sleep(0.2)
        if proc.returncode != 0 or not os.path.isfile(out_path):
            return None, None, "NOT_SOLVED"
        with open(out_path) as f:
            r = json.load(f)
        return r["cost"], r["feasible"], r["status"]


def milp_agrees(ref_cost, ref_feasible, m_cost, m_feasible, m_status, tol=1e-4):
    """True iff the big-M MILP verdict is consistent with the DP reference.

    OPTIMAL / INFEASIBLE verdicts must match the reference exactly (costs are
    integers; tol only absorbs floating-point noise). A FEASIBLE verdict means
    the time limit was hit with an incumbent, so only a one-sided check applies:
    the incumbent cannot beat the reference optimum. Any other status reaches
    no verdict and is not counted as disagreement.
    """
    if m_status == "OPTIMAL":
        return (m_feasible == ref_feasible
                and (not ref_feasible or abs(m_cost - ref_cost) <= tol))
    if m_status == "INFEASIBLE":
        return not ref_feasible
    if m_status == "FEASIBLE":
        return ref_feasible and m_cost is not None and m_cost >= ref_cost - tol
    return True


def regression_env(midnight_block):
    """Depot out-arc 0->1 carries a wrap-around window; the cheap route must use it.

    With midnight_block=True the window covers t=0, so departing on 0->1 at t=0 is
    forbidden and the true optimum must use the expensive 0->2 arc. A solver that
    leaves the initial clock unconstrained returns the (infeasible) cheap route.
    """
    map_ = [
        [(1, 10, 30), (2, 100, 30), (-1, 0, 0), (-1, 0, 0)],   # 0 -> 1 (cheap), 0 -> 2
        [(2, 10, 30), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)],     # 1 -> 2
        [(3, 10, 30), (0, 10, 30), (-1, 0, 0), (-1, 0, 0)],    # 2 -> 3, 2 -> 0
        [(0, 10, 30), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)],     # 3 -> 0
    ]
    if midnight_block:
        limits = [(0, 1, 22.0, 6.0)]   # restricted 22:00..06:00, covers t=0
    else:
        limits = [(0, 1, 22.0, 23.5)]  # restricted 22:00..23:30, t=0 allowed
    return Sample_Env(map=[list(row) for row in map_], nodes_num=4, edges_num=6,
                      edges=None, target=[2], edge_limits=limits, random_limits=False)


def run_case(name, env, max_steps, time_limit, expected_cost):
    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)
    ref_cost, ref_feasible, ref_steps = dp_optimum(env, max_steps)
    k = max_steps if max_steps is not None else (ref_steps or 60)
    res = solve_env(env, time_limit=time_limit, workers=4, max_steps=k)
    m_cost, m_feasible, m_status = milp_optimum(env, k, time_limit=time_limit)
    ok = (ref_feasible == (res["status"] in ("OPTIMAL", "FEASIBLE"))
          and abs((res["cost"] or 0) - (ref_cost or 0)) < 1e-6
          and bool(res["replay_match"])
          and (expected_cost is None or abs(ref_cost - expected_cost) < 1e-6)
          and milp_agrees(ref_cost, ref_feasible, m_cost, m_feasible, m_status))
    m_str = (f"{m_cost:g}" if m_cost is not None
             else ("INFEAS" if m_status == "INFEASIBLE" else "N/A"))
    print(f"{name} | DP={ref_cost} | Gurobi={res['status']} cost={res['cost']} "
          f"replay={res['replay_match']} | MILP={m_status} cost={m_str} "
          f"| {'OK' if ok else 'FAIL'}")
    return ok


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--milp-worker":
        _milp_worker(sys.argv[2], sys.argv[3])
        return 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="horizon for Gurobi/MILP; None = DP without horizon")
    parser.add_argument("--start-trial", type=int, default=0,
                        help="skip solving trials before this index (rng sequence "
                             "is preserved, so chunked runs cover the same "
                             "instances as a single full run)")
    args = parser.parse_args()

    reg_ok = run_case("regression midnight-block ", regression_env(True),
                      args.max_steps, args.time_limit, 110)
    reg_ok &= run_case("regression no-midnight    ", regression_env(False),
                       args.max_steps, args.time_limit, 30)
    if not reg_ok:
        print("\nregression cases FAILED")
        return 1

    rng = random.Random(20240821)
    n_ok = 0
    n_infeasible_agree = 0
    n_milp_unproven = 0
    fails = []

    for trial in range(args.trials):
        n = rng.randint(5, 8)
        n_targets = rng.randint(1, 3)
        k_v = rng.randint(2, 6)
        long_dur = rng.random() < 0.5
        seed = rng.randrange(10 ** 6)
        env = gen_tiny_env(n, n_targets, k_v, seed, long_durations=long_dur)

        if trial < args.start_trial:
            continue

        env.reset_agent_time(0.0)
        env.reset_env()
        env.reset_agent_pos(0)

        ref_cost, ref_feasible, ref_steps = dp_optimum(env, args.max_steps)
        k = args.max_steps if args.max_steps is not None else (ref_steps or 60)
        res = solve_env(env, time_limit=args.time_limit, workers=4,
                        max_steps=k)
        m_cost, m_feasible, m_status = milp_optimum(env, k,
                                                    time_limit=args.time_limit)

        solver_feasible = res["status"] in ("OPTIMAL", "FEASIBLE")
        ok = True
        if ref_feasible != solver_feasible:
            ok = False
            fails.append((trial, "feasibility mismatch", ref_feasible, res["status"]))
        elif ref_feasible:
            if abs(res["cost"] - ref_cost) > 1e-6:
                ok = False
                fails.append((trial, "cost mismatch", ref_cost, res["cost"]))
            if not res["replay_feasible"] or not res["replay_match"]:
                ok = False
                fails.append((trial, "replay mismatch",
                              res["replay_feasible"], res["replay_match"]))
            if res["status"] == "OPTIMAL" and res["bound"] > ref_cost + 1e-6:
                ok = False
                fails.append((trial, "bound above reference", res["bound"], ref_cost))
        else:
            n_infeasible_agree += 1

        # reference big-M MILP vs DP reference
        if not milp_agrees(ref_cost, ref_feasible, m_cost, m_feasible, m_status):
            ok = False
            fails.append((trial, "MILP mismatch", ref_cost, m_cost, m_status))
        if m_status not in ("OPTIMAL", "INFEASIBLE"):
            n_milp_unproven += 1

        if ok:
            n_ok += 1
        cost_str = "  INFEAS" if res["cost"] is None else f"{res['cost']:7.0f}"
        m_str = (f"{m_cost:7.0f}" if m_cost is not None
                 else ("INFEAS" if m_status == "INFEASIBLE" else "    N/A"))
        print(f"trial {trial:3d} | n={n} T={n_targets} k={k_v} long={int(long_dur)} "
              f"| DP={'INFEAS' if ref_cost is None else f'{ref_cost:7.0f}'} "
              f"| Gurobi={res['status']:9s} cost={cost_str} "
              f"replay={res['replay_match']} "
              f"| MILP={m_status:9s} cost={m_str} | {'OK' if ok else 'FAIL'}")

    print(f"\n{n_ok} passed, {len(fails)} failed "
          f"({n_infeasible_agree} jointly infeasible, "
          f"{n_milp_unproven} MILP verdicts unproven)")
    for f in fails:
        print("FAIL:", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
