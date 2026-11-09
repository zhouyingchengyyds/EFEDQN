"""Instance-pool evaluation: load the fixed instances written by instances.py and pair
them with the solver results to compute the gap.

Usage:
  Plain run (all defaults: evaluate the benchmark pool):
      python evaluate_pool.py
  Self-check (replay the solver path and verify the instance/solution pairing):
      python evaluate_pool.py --model none
  Model evaluation (custom directory and number of reruns):
      python evaluate_pool.py --instances-dir benchmark_k100 --aug-k 10
"""

import argparse
import csv
import glob
import json
import os
import random
import time

import numpy as np

from instances import env_from_dict
from exact_solver import replay_path

ROLLOUT_SAFETY_CAP = 1000


def rollout(env, qnet, device, use_mask=True):
    """One greedy rollout; returns (total_reward, cost, invalid_actions, window_violations, completed, steps)

    use_mask=True  : apply the action mask (no action that violates a time window or has no road),
                     consistent with training;
    use_mask=False : plain no-mask protocol, pure argmax over the Q values.
    """
    import torch

    total_reward = 0.0
    cost = 0.0
    v = [0, 0]
    steps = 0
    done = False
    while not done:
        state, nodes_size = env.get_state()
        t_state = [s.to(device) for s in state]
        t_nodes_size = torch.tensor([nodes_size], dtype=torch.int32).to(device)
        with torch.no_grad():
            qs, _ = qnet(t_state[0], t_state[1], t_state[2], t_state[3], t_nodes_size)
        qs = qs.view(-1)
        if use_mask:
            # Action mask: exclude actions that violate an edge time window or have no road
            action_mask = env.get_action_mask()
            if any(action_mask):
                mask_t = torch.tensor(action_mask, dtype=torch.bool, device=qs.device)
                qs = qs.masked_fill(~mask_t, float('-inf'))
        action = torch.argmax(qs).item()
        if env.map[env.agent_pos][action][0] == -1:
            cost += 1000
        else:
            cost += env.map[env.agent_pos][action][1]
        _, reward, done, violation = env.one_step(action, Q=False)
        total_reward += reward
        v[0] += violation[0]
        v[1] += violation[1]
        steps += 1
        if steps >= ROLLOUT_SAFETY_CAP:
            break
    completed = done
    return total_reward, cost, v[0], v[1], completed, steps


def evaluate_env(env, qnet, device, aug_k, use_mask=True):
    """Two protocols: EFEDQN (one greedy rollout) and EFEDQN-E (best of aug_k rollouts).
    use_mask is forwarded to rollout: True = masked evaluation (default), False = no mask.
    Returns a dict of metrics per protocol."""
    results = {}
    n_targets = len(env.goal_states1)

    t0 = time.perf_counter()
    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)
    env.shuffle_space()
    r, c, i0, i1, comp, s = rollout(env, qnet, device, use_mask=use_mask)
    t_ms = (time.perf_counter() - t0) * 1000
    results["EFEDQN"] = dict(reward=r, cost=c, invalid=i0, window_viol=i1,
                             completed=comp, steps=s, time_ms=t_ms, n_targets=n_targets)

    t0 = time.perf_counter()
    best = None
    for _ in range(aug_k):
        env.reset_agent_time(0.0)
        env.reset_env()
        env.reset_agent_pos(0)
        env.shuffle_space()
        r, c, i0, i1, comp, s = rollout(env, qnet, device, use_mask=use_mask)
        cand = dict(reward=r, cost=c, invalid=i0, window_viol=i1,
                    completed=comp, steps=s, n_targets=n_targets)
        if best is None or r > best["reward"]:
            best = cand
    t_ms = (time.perf_counter() - t0) * 1000
    best["time_ms"] = t_ms
    results["EFEDQN-E"] = best
    return results


def summarize(per_instance_rows):
    """Aggregate by (condition, protocol): pass rate / fitness / conditional cost / time"""
    summary = []
    groups = {}
    for row in per_instance_rows:
        groups.setdefault((row["condition"], row["protocol"]), []).append(row)

    for (condition, protocol), rows in groups.items():
        n = len(rows)
        costs_feasible = [r["cost"] for r in rows if r["feasible"]]
        z_list = [0 if r["feasible"] else 1 for r in rows]
        fitness_list = [6 * r["n_targets"] / (r["cost"] + 1e4 * z)
                        for r, z in zip(rows, z_list)]
        gaps = [r["gap_pct"] for r in rows
                if r.get("gap_pct") is not None and not np.isnan(r["gap_pct"])]
        summary.append({
            "condition": condition,
            "protocol": protocol,
            "n_instances": n,
            "pass_rate": (z_list.count(0) / n) if n else 0.0,
            "mean_cost_all": sum(r["cost"] for r in rows) / n,
            "mean_cost_feasible": (sum(costs_feasible) / len(costs_feasible)) if costs_feasible else float("nan"),
            "median_cost_feasible": (sorted(costs_feasible)[len(costs_feasible) // 2]) if costs_feasible else float("nan"),
            "n_feasible": len(costs_feasible),
            "mean_gap_pct": (sum(gaps) / len(gaps)) if gaps else float("nan"),
            "n_gap": len(gaps),
            "mean_fitness": sum(fitness_list) / n,
            "mean_time_ms": sum(r["time_ms"] for r in rows) / n,
            "median_time_ms": sorted(r["time_ms"] for r in rows)[n // 2],
        })
    return summary


def load_solver_results(instances_dir):
    path = os.path.join(instances_dir, "solver_results.csv")
    by_id = {}
    if not os.path.isfile(path):
        # instances.py writes the CSV only at the end; fall back to the per-instance
        # JSONs (self-check mode) or proceed without gap reference (model mode)
        for jp in sorted(glob.glob(os.path.join(instances_dir, "n*", "n*.json"))):
            with open(jp, encoding="utf-8") as f:
                inst = json.load(f)
            s = inst.get("solver") or {}
            if not s:
                continue
            by_id[inst["id"]] = {
                "status": s["status"],
                "opt_cost": float(s["cost"]),
                "bound": float(s["bound"]) if s.get("bound") else None,
                "wall_time_s": float(s["wall_time_s"]),
            }
        return by_id
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_id[row["instance_id"]] = {
                "status": row["status"],
                "opt_cost": float(row["opt_cost"]),
                "bound": float(row["bound"]) if row["bound"] else None,
                "wall_time_s": float(row["wall_time_s"]),
            }
    return by_id


def load_model(path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch has no weights_only argument
        return torch.load(path, map_location=device)


def run_pool_mode(args):
    """Fixed-pool mode: load the instances from instances_dir and compute the gap against the solver results."""
    files = sorted(glob.glob(os.path.join(args.instances_dir, "n*", "n*.json")))
    if not files:
        raise SystemExit(f"no instances found under {args.instances_dir}")
    solver_by_id = load_solver_results(args.instances_dir)

    qnet, device = None, None
    if args.model != "none":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qnet = load_model(args.model, device)
        qnet.eval()

    per_instance_rows = []
    for idx, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            inst = json.load(f)
        env = env_from_dict(inst)
        inst_id = inst["id"]
        solver = solver_by_id.get(inst_id)

        # per-instance deterministic seed: shuffle_space() permutes action slots,
        # so without seeding the rollout is not reproducible across reruns
        random.seed(args.seed * 1000003 + idx)
        np.random.seed((args.seed * 1000003 + idx) % (2 ** 31))

        if qnet is None:
            # Self-check mode: replay the solver path to verify the instance/solution pairing
            rep = replay_path(env, inst["solver"]["path"],
                              max_steps=inst.get("max_steps", 49))
            ok = (rep is not None and rep["feasible"]
                  and abs(rep["cost"] - inst["solver"]["cost"]) < 1e-6)
            print(f"[check] {inst_id}: replay_ok={ok} "
                  f"status={inst['solver']['status']} cost={inst['solver']['cost']}")
            continue

        res = evaluate_env(env, qnet, device, args.aug_k, use_mask=not args.no_mask)
        for protocol, m in res.items():
            feasible = 1 if (m["completed"] and m["invalid"] == 0
                             and m["window_viol"] == 0) else 0
            gap_pct = float("nan")
            gap_kind = ""
            if feasible and solver is not None:
                if solver["status"] == "OPTIMAL":
                    gap_pct = 100.0 * (m["cost"] - solver["opt_cost"]) / solver["opt_cost"]
                    gap_kind = "opt"
                elif solver["bound"]:
                    gap_pct = 100.0 * (m["cost"] - solver["bound"]) / solver["bound"]
                    gap_kind = "bound"
            per_instance_rows.append({
                "condition": f"n{inst['nodes']}", "nodes": inst["nodes"],
                "tw": inst["num_windows"], "targets": inst["num_targets"],
                "instance_id": inst_id, "protocol": protocol,
                "cost": m["cost"], "reward": m["reward"],
                "invalid_actions": m["invalid"], "window_violations": m["window_viol"],
                "completed": int(m["completed"]), "feasible": feasible,
                "steps": m["steps"], "n_targets": m["n_targets"],
                "time_ms": m["time_ms"],
                "solver_status": solver["status"] if solver else "",
                "solver_opt_cost": solver["opt_cost"] if solver else "",
                "solver_bound": solver["bound"] if solver else "",
                "gap_pct": gap_pct, "gap_kind": gap_kind,
            })

    if qnet is None:
        print("model=none: only instance/solver consistency was checked")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "per_instance_pool.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_instance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_instance_rows)

    summary = summarize(per_instance_rows)
    summary_path = os.path.join(args.output_dir, "summary_pool.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n逐实例明细: {detail_path}")
    print(f"汇总结果:   {summary_path}")
    print("\n=== 汇总（pass rate / 条件成本 / gap / 平均时间） ===")
    for s in summary:
        print(f"{s['condition']:>6} {s['protocol']:>9} | pass={s['pass_rate']:.3f} "
              f"| cost(feas)={s['mean_cost_feasible']:.2f} "
              f"| gap={s['mean_gap_pct']:.2f}% (n={s['n_gap']}) "
              f"| time={s['mean_time_ms']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="实例池评估：固定实例 + 与求解器最优解配对计算 gap")
    parser.add_argument('--instances-dir', type=str, default='benchmark',
                        help='实例池目录，默认 benchmark')
    parser.add_argument('--model', type=str, default='model_seed1.pth',
                        help="模型权重路径；'none' 表示只做实例/求解器一致性自检")
    parser.add_argument('--seed', type=int, default=1, help='rollout 随机种子')
    parser.add_argument('--aug-k', type=int, default=10, help='EFEDQN-E 的重排次数')
    parser.add_argument('--no-mask', action='store_true', default=False,
                        help='关闭动作掩码，按原始无掩码协议 rollout（默认开启，与训练一致）')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    run_pool_mode(args)


if __name__ == "__main__":
    main()
