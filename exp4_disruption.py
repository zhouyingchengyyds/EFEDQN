# -*- coding: utf-8 -*-
"""exp4_disruption.py — online disruption and replanning experiment.

After the vehicle has started executing its route, unforeseen disruptions are
injected and three replanning methods are compared:
  EFEDQN(re-encoding) — the full model continues from the current state and
                        re-encodes at every step
  EFEDQN(frozen)      — the frozen-encoder version (model_seed1_no_dy.pth)
                        continues from the current state
  RollingHorizon      — time-dependent shortest-path rolling-horizon solver
                        that re-solves the remaining problem as a whole from the
                        current position, time and remaining targets (classic
                        re-optimization baseline)

Protocol:
- 100 nodes / 64 windows / 32 targets (--num-instances, default 100,
  adjustable);
- first generate a planned route with standard greedy inference, then inject a
  disruption at 25%/50%/75% of the planned steps:
    closure — close one unvisited edge of the remaining route (assigned the
              full [0,24] window)
    window  — attach a new random window (1-13 hours) to a free edge of the
              remaining route
- continue from (current position, current time, remaining targets, remaining
  step budget);
- metrics: post-disruption feasibility rate, incremental cost (post cost minus
  planned cost, feasible cases only), replanning latency (wall-clock time of a
  single forward pass for EFEDQN/Frozen; full re-solve time for RH), failure
  mode counts (step cap exceeded / dead end).

Usage:
  python exp4_disruption.py --num-instances 100
  python exp4_disruption.py --num-instances 10 --points 0.5   # small trial run
"""

import argparse
import copy
import os
import random
import time

import numpy as np

import exp_common as ec


def plan_prefix_state(env, actions, upto, max_steps):
    """Replay the first upto steps of the plan;
    returns (pos, time, remaining_goals, cost_so_far)."""
    env.reset_env()
    env.reset_agent_pos(0)
    env.reset_agent_time(0.0)
    cost = 0.0
    for i, a in enumerate(actions[:upto]):
        if i >= max_steps:
            break
        cur = env.agent_pos
        nxt = env.map[cur][a][0]
        if nxt == -1:
            cost += 1000.0
        else:
            cost += env.map[cur][a][1]
        env.one_step(a, Q=False)
    return env.agent_pos, env.agent_time % 24.0, list(env.goal_states), cost


def planned_total_cost(env, actions, max_steps):
    m = ec.replay_actions(env, actions, max_steps=max_steps)
    return m["cost"], m["feasible"]


def remaining_route_edges(env, actions, from_step):
    """Sequence of (u, v) edges of the planned route after from_step
    (deduplicated, order preserved)."""
    env.reset_env()
    env.reset_agent_pos(0)
    env.reset_agent_time(0.0)
    edges = []
    for i, a in enumerate(actions):
        u = env.agent_pos
        v = env.map[u][a][0]
        if i >= from_step and v != -1:
            if (u, v) not in edges:
                edges.append((u, v))
        env.one_step(a, Q=False)
    return edges


def inject_disruption(env, actions, from_step, kind, rng):
    """Inject a disruption into edge_limits of the env copy;
    returns (description, success)."""
    edges = remaining_route_edges(env, actions, from_step)
    if not edges:
        return None, False
    if kind == "closure":
        u, v = rng.choice(edges)
        env.edge_limits.append((u, v, 0.0, 24.0))  # closed for the whole day
        return {"kind": "closure", "edge": (u, v)}, True
    elif kind == "window":
        # Prefer edges without a window; if every edge has one, replace the
        # window of an existing edge instead
        with_win = {(u, v) for (u, v, _, _) in env.edge_limits}
        free = [e for e in edges if e not in with_win]
        if free:
            u, v = rng.choice(free)
            s = rng.uniform(0.0, 24.0)
            dur = rng.uniform(1.0, 13.0)
            env.edge_limits.append((u, v, s, (s + dur) % 24.0))
            return {"kind": "window", "edge": (u, v)}, True
        u, v = rng.choice(edges)
        env.edge_limits = [(a, b, lo, hi) for (a, b, lo, hi) in env.edge_limits
                           if (a, b) != (u, v)]
        s = rng.uniform(0.0, 24.0)
        dur = rng.uniform(1.0, 13.0)
        env.edge_limits.append((u, v, s, (s + dur) % 24.0))
        return {"kind": "window(re)", "edge": (u, v)}, True
    raise ValueError(kind)


def single_forward_latency_ms(solver, env, init, device_unused=None):
    """Wall-clock time of one forward pass (re-encode + decode), i.e. the
    replanning latency of the EFEDQN family."""
    env_c = copy.deepcopy(env)
    env_c.reset_env()
    env_c.agent_pos = init["pos"]
    env_c.start_pos = init.get("start_pos", 0)
    env_c.agent_time = init["time"]
    if init.get("goals") is not None:
        env_c.goal_states = list(init["goals"])
    t0 = time.perf_counter()
    state, nsize = env_c.get_state()
    # The continuation point is the (re-)encoding moment, so the elapsed time
    # is 0; the frozen version receives it through needs_time
    qs = ec._qnet_forward(solver.qnet, state, nsize, solver.device, elapsed=0.0)
    _ = int(np.argmax(qs.detach().cpu().numpy()))
    return (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="表10 在线扰动与重规划(C8)")
    parser.add_argument('--num-instances', type=int, default=100)
    parser.add_argument('--points', type=str, default="0.25,0.5,0.75",
                        help='扰动注入点(占计划步数比例)')
    parser.add_argument('--kinds', type=str, default="closure,window")
    parser.add_argument('--n', type=int, default=100, help='实例节点数')
    parser.add_argument('--k-v', type=int, default=64, help='限制窗数')
    parser.add_argument('--targets', type=int, default=32, help='目标数')
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp4_disruption'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    points = [float(p) for p in args.points.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    efq = ec.make_efedqn_solvers(device=device)
    full = efq["EFEDQN"]
    frozen = efq["EFEDQN-Frozen"]
    if frozen is None:
        print("[警告] 缺少 test/Freeze_code/model_seed1_no_dy.pth (C8 对照方法), "
              "本次 EFEDQN(frozen) 行留空, 仅运行 EFEDQN(re-encoding) 与 "
              "RollingHorizon; 权重补齐后重跑 exp4 即可补全该列")
    rh = ec.RollingHorizonAdapter(max_steps=args.max_steps, two_opt=False)

    inst_dir = os.path.join(args.output_dir, "instances")
    os.makedirs(inst_dir, exist_ok=True)

    rows = []
    for i in range(args.num_instances):
        seed = ec.derive_seed("exp4", i)
        inst = ec.gen_random_instance(args.n, args.k_v, args.targets, seed)
        inst["id"] = f"disrupt_i{i:03d}"
        ec.save_instances([inst], inst_dir)
        env_plan = ec.env_from_inst(inst)

        # (1) Plan with standard greedy
        plan = full.solve(env_plan, seed=ec.derive_seed(seed, "plan"),
                          max_steps=args.max_steps)
        plan_cost, plan_ok = plan["cost"], plan["feasible"]
        actions = plan.get("actions", [])
        if not plan_ok or not actions:
            print(f"[{inst['id']}] 计划不可行, 跳过扰动")
            continue
        # Note: rollout calls env.shuffle_space(), which reorders the action
        # slots of env_plan in place, and the planned action sequence is
        # recorded against that reordered slot layout. Every replay of the
        # planned actions (prefix state / remaining route edges / disruption
        # injection) must therefore run on a deep copy of env_plan rather than
        # on a fresh env_from_inst, which would restore the original layout and
        # make the actions point at the wrong edges.
        for p in points:
            step_idx = max(1, int(len(actions) * p))
            for kind in kinds:
                rng = random.Random(ec.derive_seed(seed, p, kind))
                env_d = copy.deepcopy(env_plan)  # same slot layout as the plan
                desc, ok = inject_disruption(env_d, actions, step_idx, kind, rng)
                if not ok:
                    continue
                pos, t, goals, pre_cost = plan_prefix_state(
                    copy.deepcopy(env_d), actions, step_idx, args.max_steps)
                init = {"pos": pos, "time": t, "goals": goals, "start_pos": 0}
                budget = args.max_steps - step_idx

                base = {
                    "condition": f"p={p:g},{desc['kind']}",
                    "nodes": args.n, "tw": args.k_v, "targets": args.targets,
                    "instance_id": inst["id"],
                    "disrupt_point": p, "disrupt_kind": desc["kind"],
                    "disrupt_edge": str(desc["edge"]),
                    "step_idx": step_idx, "plan_cost": plan_cost,
                    "pre_cost": pre_cost,
                }

                # (2a) EFEDQN with re-encoding
                r = full.solve(copy.deepcopy(env_d),
                               seed=ec.derive_seed(seed, "reenc", p, kind),
                               max_steps=budget, init=init)
                lat = single_forward_latency_ms(full, env_d, init)
                post_cost = pre_cost + r["cost"]
                rows.append({**base, "protocol": "EFEDQN(re-encoding)",
                             "feasible": int(r["feasible"]),
                             "completed": int(r["completed"]),
                             "post_cost": post_cost,
                             "incr_cost": (post_cost - plan_cost)
                             if r["feasible"] else float("nan"),
                             "latency_ms": lat, "time_ms": r["time_ms"],
                             "n_targets": len(goals)})

                # (2b) frozen encoding (left empty when the weights are missing)
                if frozen is not None:
                    r = frozen.solve(copy.deepcopy(env_d),
                                    seed=ec.derive_seed(seed, "frozen", p, kind),
                                    max_steps=budget, init=init)
                    lat = single_forward_latency_ms(frozen, env_d, init)
                    post_cost = pre_cost + r["cost"]
                    rows.append({**base, "protocol": "EFEDQN(frozen)",
                                 "feasible": int(r["feasible"]),
                                 "completed": int(r["completed"]),
                                 "post_cost": post_cost,
                                 "incr_cost": (post_cost - plan_cost)
                                 if r["feasible"] else float("nan"),
                                 "latency_ms": lat, "time_ms": r["time_ms"],
                                 "n_targets": len(goals)})

                # (2c) rolling-horizon solver
                r = rh.solve(copy.deepcopy(env_d),
                             seed=ec.derive_seed(seed, "rh", p, kind),
                             max_steps=budget, init=init)
                post_cost = pre_cost + r["cost"]
                rows.append({**base, "protocol": "RollingHorizon",
                             "feasible": int(r["feasible"]),
                             "completed": int(r["completed"]),
                             "post_cost": post_cost,
                             "incr_cost": (post_cost - plan_cost)
                             if r["feasible"] else float("nan"),
                             "latency_ms": r["time_ms"], "time_ms": r["time_ms"],
                             "n_targets": len(goals)})

                tail = {r2["protocol"]: r2 for r2 in rows[-3:]}
                renc = tail.get("EFEDQN(re-encoding)", {})
                froz = tail.get("EFEDQN(frozen)", {})
                rhc = tail.get("RollingHorizon", {})
                print(f"[{inst['id']}] {base['condition']} "
                      f"reenc={renc.get('feasible', '-')}/"
                      f"{renc.get('latency_ms', 0.0):.1f}ms "
                      f"frozen={froz.get('feasible', '-')}/"
                      f"{froz.get('latency_ms', 0.0):.1f}ms "
                      f"rh={rhc.get('feasible', '-')}/"
                      f"{rhc.get('latency_ms', 0.0):.1f}ms")

    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), rows)

    # Summary by (disruption point + kind, method)
    summary = []
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["protocol"]), []).append(r)
    for (cond, proto), rs in sorted(groups.items()):
        feas = [r for r in rs if r["feasible"]]
        incrs = [r["incr_cost"] for r in feas
                 if r["incr_cost"] == r["incr_cost"]]
        summary.append({
            "condition": cond, "protocol": proto, "n": len(rs),
            "feasible_rate": len(feas) / len(rs),
            "mean_incr_cost": np.mean(incrs) if incrs else float("nan"),
            "mean_latency_ms": np.mean([r["latency_ms"] for r in rs]),
            "median_latency_ms": float(np.median(
                [r["latency_ms"] for r in rs])),
        })
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"), summary)
    print("\n=== 汇总（可行率 / 增量成本 / 重规划延迟） ===")
    for s in summary:
        print(f"{s['condition']:>18} {s['protocol']:>20} | "
              f"feas={s['feasible_rate']:.3f} | "
              f"incr={s['mean_incr_cost']:.2f} | "
              f"latency={s['median_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
