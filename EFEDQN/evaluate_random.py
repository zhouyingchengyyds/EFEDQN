"""Random-instance evaluation: instances are generated on the fly, each one uniquely
determined by its seed and therefore reproducible. No solver is involved, so no gap
is computed; the metrics are pass rate / fitness / cost / time.

Usage:
  python evaluate_random.py --model model_seed0.pth --num-instances 100
  python evaluate_random.py --model model_seed0.pth --no-mask   # plain no-mask protocol
"""

import argparse
import csv
import os
import random

import numpy as np
import torch

from Env import Sample_Env
from Gragh import generate_directed_graph, graph_to_dict
from evaluate_pool import evaluate_env, summarize, load_model

# Six conditions matching test_envs in train.py (five distinct groups after deduplication)
CONDITIONS = [
    # (name, nodes, number of restricted windows, number of targets, label)
    ("25n_16tw",   25,  16,  8, "Table 6: 25 nodes"),
    ("50n_32tw",   50,  32, 16, "Table 6: 50 nodes"),
    ("100n_64tw", 100,  64, 32, "Table 6: 100 nodes / Table 7: 64 tw"),
    ("100n_32tw", 100,  32, 32, "Table 7: 32 tw"),
    ("100n_128tw", 100, 128, 32, "Table 7: 128 tw"),
]


def make_env(n, k_v, target_num):
    G1 = generate_directed_graph(n)
    map_, node_size, edge_size = graph_to_dict(G1)
    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    return Sample_Env(map=map_, nodes_num=node_size, edges_num=edge_size,
                      edges=edge, target=target_nodes, times=times, k_num=k_v)


def main():
    parser = argparse.ArgumentParser(
        description="随机生成评估：种子逐实例可复现，无 gap")
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--seed', type=int, default=2024,
                        help='实例生成种子（独立于训练种子）')
    parser.add_argument('--num-instances', type=int, default=100, help='每条件实例数')
    parser.add_argument('--aug-k', type=int, default=10, help='EFEDQN-E 的重排次数')
    parser.add_argument('--no-mask', action='store_true', default=False,
                        help='关闭动作掩码，按原始无掩码协议 rollout（默认开启，与训练一致）')
    parser.add_argument('--output-dir', type=str, default='results_random')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnet = load_model(args.model, device)
    qnet.eval()

    use_mask = not args.no_mask
    print(f"action mask: {'on (masked protocol)' if use_mask else 'off (original no-mask protocol)'}")

    os.makedirs(args.output_dir, exist_ok=True)
    per_instance_rows = []

    for name, n, k_v, t_num, table in CONDITIONS:
        print(f"[{name}] ({table}) generating {args.num_instances} instances ...")
        for inst_id in range(args.num_instances):
            # The seed is unique per instance: rerunning or reordering leaves the result unchanged
            seed = args.seed * 1000003 + n * 1009 + inst_id
            random.seed(seed)
            np.random.seed(seed % (2 ** 31))

            env = make_env(n, k_v, t_num)
            res = evaluate_env(env, qnet, device, args.aug_k, use_mask=use_mask)
            for protocol, m in res.items():
                feasible = 1 if (m["completed"] and m["invalid"] == 0
                                 and m["window_viol"] == 0) else 0
                per_instance_rows.append({
                    "condition": name, "nodes": n, "tw": k_v, "targets": t_num,
                    "instance_id": inst_id, "seed": seed, "protocol": protocol,
                    "cost": m["cost"], "reward": m["reward"],
                    "invalid_actions": m["invalid"], "window_violations": m["window_viol"],
                    "completed": int(m["completed"]), "feasible": feasible,
                    "steps": m["steps"], "n_targets": m["n_targets"],
                    "time_ms": m["time_ms"],
                })
        done = sum(r["feasible"] for r in per_instance_rows
                   if r["condition"] == name and r["protocol"] == "EFEDQN")
        total = sum(1 for r in per_instance_rows
                    if r["condition"] == name and r["protocol"] == "EFEDQN")
        print(f"[{name}] EFEDQN pass: {done}/{total}")

    detail_path = os.path.join(args.output_dir, "per_instance.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_instance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_instance_rows)

    summary = summarize(per_instance_rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n逐实例明细: {detail_path}")
    print(f"汇总结果:   {summary_path}")
    print("\n=== 汇总（pass rate / fitness / 条件成本 / 平均时间） ===")
    for s in summary:
        print(f"{s['condition']:>10} {s['protocol']:>9} | pass={s['pass_rate']:.3f} "
              f"| fitness={s['mean_fitness']:.4f} "
              f"| cost(feas)={s['mean_cost_feasible']:.2f} "
              f"| time={s['mean_time_ms']:.1f}ms")


if __name__ == "__main__":
    main()
