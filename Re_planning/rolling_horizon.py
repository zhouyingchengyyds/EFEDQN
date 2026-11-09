# -*- coding: utf-8 -*-
"""rolling_horizon.py — time-dependent rolling-horizon heuristic solver.

Design:

1. Time-dependent shortest path (TDSP): waiting is not allowed and every arc has
   positive duration, so the departure time at the source fixes the arrival
   time; different paths still trade off (arrival time, accumulated cost) in
   the Pareto sense, since a later departure may be cheaper and an earlier
   arrival leaves more room for the following windows. Labels are maintained by
   label-correcting with a priority queue ordered by arrival time, one
   non-dominated label set per node; relaxing an outgoing arc checks the closed
   (possibly midnight-crossing) window at that departure time. Each node keeps
   at most label_cap labels (default 20) and drops the latest arrivals when the
   cap is exceeded.
2. Greedy target order (nearest-cheapest): each step runs a single-source TDSP
   from the current position and time, picks the reachable remaining target
   with the lowest cost (earliest arrival on ties) and backtracks its optimal
   label into an action sequence; after all targets are visited the vehicle
   returns to the depot the same way.
3. Optional 2-opt (first improvement, limited iterations): reverses segments of
   the target order and replays the TDSP for each candidate order, because a
   time-dependent network cannot cache the segments. Disabled by default;
   suitable for instances with few targets.
4. Online replanning: solve(env, init=position/time/remaining targets) re-solves
   from an intermediate state, which provides the rolling-horizon behaviour
   used as a step-level replanning reference.

Complexity: one single-source TDSP is O(label_cap * |A| * log); a greedy order
needs T TDSP runs.
"""

import heapq
import itertools

_EPS = 1e-9


def _insert_label(labs, nl, cap):
    """Label management for a no-wait time-varying network (safe dominance plus
    a practical cap).

    - Strict dominance: keep the cheaper label only for the same arrival time
      (within tolerance). Pareto dominance across arrival times (t1 <= t2 and
      c1 <= c2) is unsafe without waiting, because "earlier and cheaper" would
      prune the only feasible path that arrives later to avoid a closed window.
    - Truncation: keep the cheapest cap labels per node by (cost, t). Positive
      arc costs make detour labels monotonically more expensive, so truncation
      bounds the number of detours and guarantees termination.
    """
    for l in labs:
        if abs(l.t - nl.t) <= _EPS and l.cost <= nl.cost + _EPS:
            return False
    keep = [l for l in labs if abs(l.t - nl.t) > _EPS or nl.cost < l.cost - _EPS]
    keep.append(nl)
    if len(keep) > cap:
        keep.sort(key=lambda l: (l.cost, l.t))
        if keep[-1] is nl:
            return False  # nl is the most expensive label and the cap is exceeded -> reject
        keep = keep[:cap]  # drop the most expensive old labels
    labs[:] = keep
    return True


class _Label:
    __slots__ = ("node", "t", "cost", "parent", "action", "hop")

    def __init__(self, node, t, cost, parent=None, action=None, hop=0):
        self.node = node
        self.t = t          # arrival time (hours, mod 24)
        self.cost = cost    # accumulated cost
        self.parent = parent
        self.action = action  # action (slot) used to enter this node from parent.node
        self.hop = hop      # number of steps taken (bounded by max_steps)


class RollingHorizonSolver:
    def __init__(self, max_steps=200, two_opt=False, label_cap=20,
                 two_opt_swaps=10):
        self.max_steps = max_steps
        self.two_opt = two_opt
        self.label_cap = label_cap
        self.two_opt_swaps = two_opt_swaps

    # Time-dependent shortest path (single source, Pareto label sets for all nodes)
    def tdsp(self, env, src, t0, max_hops=None):
        """Single-source time-dependent shortest path from (src, t0); labels carry a hop limit.

        Returns {node: [label, ...]} with the retained labels per node (only the
        cheapest label is kept for a given arrival time).
        """
        labels = {v: [] for v in range(env.space_size)}
        src_lab = _Label(src, t0, 0.0)
        labels[src].append(src_lab)
        counter = itertools.count()
        pq = [(t0, 0.0, next(counter), src_lab)]
        while pq:
            t, c, _, lab = heapq.heappop(pq)
            # Stale label (still referenced as a parent after truncation); filter by membership
            if lab not in labels.get(lab.node, []):
                continue
            if max_hops is not None and lab.hop >= max_hops:
                continue
            u = lab.node
            for k in range(env.action_size):
                slot = env.map[u][k]
                v = slot[0]
                if v == -1:
                    continue
                if _is_active(env.edge_limits, t, u, v):
                    continue  # departure forbidden at this time (closed window)
                nt = (t + slot[2] / 60.0) % 24.0
                nc = c + slot[1]
                nl = _Label(v, nt, nc, parent=lab, action=k, hop=lab.hop + 1)
                if not _insert_label(labels[v], nl, self.label_cap):
                    continue
                heapq.heappush(pq, (nt, nc, next(counter), nl))
        return labels

    # Point to point: cheapest Pareto label (earliest arrival on ties)
    @staticmethod
    def best_label(labels, dst):
        labs = labels.get(dst) or []
        if not labs:
            return None
        return min(labs, key=lambda l: (l.cost, l.t))

    def _backtrack(self, label):
        """Label -> action sequence."""
        actions = []
        cur = label
        while cur.parent is not None:
            actions.append(cur.action)
            cur = cur.parent
        actions.reverse()
        return actions

    # For a given target order: replay the TDSP segment by segment; returns (actions, total cost, feasible)
    def _play_order(self, env, pos, t, order, depot, steps_budget):
        actions = []
        cost = 0.0
        for g in list(order) + [depot]:
            labs = self.tdsp(env, pos, t, max_hops=steps_budget - len(actions))
            lab = self.best_label(labs, g)
            if lab is None:
                return None, float("inf"), False
            seg = self._backtrack(lab)
            if len(actions) + len(seg) > steps_budget:
                return None, float("inf"), False
            actions.extend(seg)
            cost += lab.cost
            pos, t = g, lab.t
        return actions, cost, True

    # Main entry point
    def solve(self, env, init=None, start_time=None, max_steps=None):
        """Return an action sequence (scored by exp_common.replay_actions).

        init: dict(pos, time, goals, start_pos) -- partial initial state for online replanning.
        """
        ms = max_steps or self.max_steps
        if init is not None:
            pos = init["pos"]
            t = init["time"]
            goals = list(init["goals"]) if init.get("goals") is not None \
                else list(env.goal_states1)
            depot = init.get("start_pos", 0)
        else:
            pos = 0
            t = start_time if start_time is not None else 0.0
            goals = list(env.goal_states1)
            depot = 0

        # ---- 1. Greedy target order (nearest-cheapest) ----
        order = []
        actions = []
        cost = 0.0
        cur_pos, cur_t = pos, t
        remaining = list(goals)
        ok = True
        while remaining:
            labs = self.tdsp(env, cur_pos, cur_t)
            best_g, best_lab = None, None
            for g in remaining:
                lab = self.best_label(labs, g)
                if lab is None:
                    continue
                if best_lab is None or (lab.cost, lab.t) < (best_lab.cost, best_lab.t):
                    best_g, best_lab = g, lab
            if best_g is None:
                ok = False  # a remaining target is unreachable from here -> return the partial plan
                break
            seg = self._backtrack(best_lab)
            if len(actions) + len(seg) > ms:
                ok = False
                break
            actions.extend(seg)
            cost += best_lab.cost
            order.append(best_g)
            remaining.remove(best_g)
            cur_pos, cur_t = best_g, best_lab.t

        if ok:
            # Back to the depot
            labs = self.tdsp(env, cur_pos, cur_t)
            lab = self.best_label(labs, depot)
            if lab is not None and len(actions) + len(self._backtrack(lab)) <= ms:
                actions.extend(self._backtrack(lab))
                cost += lab.cost
            else:
                ok = False

        # ---- 2. Optional 2-opt (first improvement, limited iterations) ----
        if ok and self.two_opt and 2 <= len(order) and len(order) <= 16:
            actions, cost = self._two_opt(env, pos, t, order, depot, ms,
                                          actions, cost)

        return actions

    def _two_opt(self, env, pos, t, order, depot, ms, base_actions, base_cost):
        best_actions, best_cost = base_actions, base_cost
        best_order = list(order)
        improved = 0
        n = len(best_order)
        for _ in range(self.two_opt_swaps):
            found = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best_order[:i] + best_order[i:j + 1][::-1] \
                        + best_order[j + 1:]
                    if cand == best_order:
                        continue
                    acts, cost, ok = self._play_order(env, pos, t, cand,
                                                      depot, ms)
                    if ok and cost < best_cost - _EPS:
                        best_order, best_actions, best_cost = cand, acts, cost
                        improved += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return best_actions, best_cost


def _is_active(edge_limits, t, u, v):
    """Closed-window check with the same semantics as Env.is_edge_active; local copy to avoid the dependency."""
    for (a, b, lo, hi) in edge_limits:
        if a != u or b != v:
            continue
        if lo < hi:
            if lo <= t <= hi:
                return True
        else:  # crosses midnight
            if t >= lo or t <= hi:
                return True
    return False


if __name__ == "__main__":
    # Self-check: 6-node network with a lightweight mock env (no torch dependency, any environment works)
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", "EFEDQN"))

    class _MockEnv:
        """Only the interface required by the solver: map / edge_limits / space_size / action_size."""

        def __init__(self, map_, limits):
            self.map = [[tuple(s) for s in row] for row in map_]
            self.edge_limits = limits
            self.space_size = len(map_)
            self.action_size = 4
            self.goal_states1 = [3, 5]

    random_map = [
        [[1, 10, 10], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]],
        [[0, 10, 10], [2, 5, 50], [-1, 0, 0], [-1, 0, 0]],
        [[1, 5, 50], [3, 5, 5], [-1, 0, 0], [4, 8, 8]],
        [[2, 5, 5], [4, 3, 3], [-1, 0, 0], [-1, 0, 0]],
        [[3, 3, 3], [5, 4, 4], [-1, 0, 0], [-1, 0, 0]],
        [[4, 4, 4], [0, 20, 20], [-1, 0, 0], [-1, 0, 0]],
    ]
    # Window: 1->2 is forbidden on [0, 0.5] h. Departing at t = 0 cannot take the
    # direct road; waiting is not allowed, so the solver must detour via 1->0->...
    # or accept a later-arrival label.
    env = _MockEnv(random_map, [(1, 2, 0.0, 0.5)])

    solver = RollingHorizonSolver(max_steps=49)
    labels = solver.tdsp(env, 0, 0.0)
    print("node2 labels (t, cost):",
          sorted((round(l.t, 3), l.cost) for l in labels[2]))
    acts = solver.solve(env)
    print("greedy actions:", acts)

    # Manual replay check with the same semantics as exp_common.replay_actions
    pos, t, cost = 0, 0.0, 0.0
    visited = set()
    ok = True
    for a in acts:
        v, c, dur = env.map[pos][a]
        if v == -1 or _is_active(env.edge_limits, t, pos, v):
            ok = False
            break
        cost += c
        t = (t + dur / 60.0) % 24.0
        pos = v
        visited.add(v)
    done = ok and pos == 0 and visited >= {3, 5}
    print(f"replay: feasible={done} cost={cost} pos={pos} visited={sorted(visited)}")
    assert done, "自检失败: 贪心序未产生可行回路"
    print("self-check OK")
