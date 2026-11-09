# -*- coding: utf-8 -*-
"""exp1_main_comparison.py — main comparison over 5 conditions x 7 methods.

5 condition groups (25/50/100 nodes plus 32/64/128 window density at 100
nodes) x 7 methods:
  TD-ACO / GA / PSO / EFormer / EFormer-K10 / EFEDQN / EFEDQN-E

Common protocol:
- a fixed instance pool per condition (seeded and materialized to disk first,
  so all methods are evaluated on exactly the same instances);
- shared evaluation step cap --max-steps (default 200, adjustable);
- metrics: pass rate, fitness, condition cost (feasible routes only), mean
  time per instance;
- the experiment seed defaults to 1 and is derived per instance and per
  method, so runs are fully reproducible;
- EFEDQN and EFEDQN-E share the same weights (model_seed1.pth) and differ only
  in the inference protocol;
- EFormer uses checkpoint-best.pt; EFormer-K10 is best-of-10 over 10 samples,
  a search budget symmetric to EFEDQN-E.

Usage:
  python exp1_main_comparison.py                          # all methods, 100 instances/condition
  python exp1_main_comparison.py --num-instances 20       # small trial run
  python exp1_main_comparison.py --methods EFEDQN,EFormer # method subset
"""

import argparse
import os

import exp_common as ec


def build_methods(args, device):
    """Build the method table."""
    methods = {}
    if "TD-ACO" in args.methods:
        methods["TD-ACO"] = ec.HeuristicSolver("aco", restarts=1)
    if "GA" in args.methods:
        methods["GA"] = ec.HeuristicSolver("ga", restarts=1)
    if "PSO" in args.methods:
        methods["PSO"] = ec.HeuristicSolver("pso", restarts=1)
    if "EFormer" in args.methods:
        methods["EFormer"] = ec.EFormerSolver(
            os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=1,
            max_steps=args.max_steps)
    if "EFormer-K10" in args.methods:
        methods["EFormer-K10"] = ec.EFormerSolver(
            os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=10,
            name="EFormer-K10", max_steps=args.max_steps)
    if "EFEDQN" in args.methods:
        s = ec.make_efedqn_solvers(device=device)
        methods["EFEDQN"] = s["EFEDQN"]
        methods["EFEDQN-E"] = s["EFEDQN-E"]
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="表6/7 主对比: 5 条件 × 7 方法, 统一 max-steps")
    parser.add_argument('--num-instances', type=int, default=100,
                        help='每条件实例数(默认 100)')
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS,
                        help='统一评估步数上限(默认 200)')
    parser.add_argument('--methods', type=str,
                        default="TD-ACO,GA,PSO,EFormer,EFormer-K10,EFEDQN",
                        help='逗号分隔: TD-ACO,GA,PSO,EFormer,EFormer-K10,EFEDQN'
                             '(EFEDQN 自动包含 EFEDQN-E)')
    parser.add_argument('--conditions', type=str, default="all",
                        help='逗号分隔条件名或 all(5 组)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp1_main'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    methods = build_methods(args, device)
    if not methods:
        raise SystemExit("未选择任何方法")
    print(f"方法: {list(methods)} | 设备: {device} | max_steps: {args.max_steps}")

    conds = ec.CONDITIONS
    if args.conditions != "all":
        want = {c.strip() for c in args.conditions.split(",")}
        conds = [c for c in ec.CONDITIONS if c[0] in want]

    inst_dir = os.path.join(args.output_dir, "instances")
    os.makedirs(inst_dir, exist_ok=True)

    all_rows = []
    for (name, n, k_v, t_num, table) in conds:
        print(f"\n=== [{name}] ({table}) — 生成并固化 {args.num_instances} 实例 ===")
        instances = []
        for i in range(args.num_instances):
            seed = ec.derive_seed("exp1", name, i)
            inst = ec.gen_random_instance(n, k_v, t_num, seed)
            inst["id"] = f"{name}_i{i:03d}"
            instances.append(inst)
        ec.save_instances(instances, os.path.join(inst_dir, name))

        gap_refs = None  # random instances have no certified reference (see exp2)
        rows = ec.run_methods_on_instances(
            methods, instances, name, args.max_steps, gap_refs=gap_refs)
        all_rows.extend(rows)

        summary = ec.summarize(rows)
        ec.print_summary(summary)
        ec.write_csv(os.path.join(args.output_dir, "summary.csv"),
                     ec.summarize(all_rows))

    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), all_rows)
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"),
                 ec.summarize(all_rows))
    print(f"\n完成: {len(all_rows)} 行结果 -> {args.output_dir}")


if __name__ == "__main__":
    main()
