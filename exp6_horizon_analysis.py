# -*- coding: utf-8 -*-
"""exp6_horizon_analysis.py — sensitivity of scale extrapolation to the evaluation step cap.

Background: with a common K_eval=200, the EFEDQN pass rate falls to 3.3% at 300
nodes, yet its rollouts show no window violations, no illegal actions and hit
the 200-step cap in 100% of the cases — the shortest complete route for 300
nodes / 96 targets typically needs ~200+ steps on its own, so the cap becomes
binding at that scale. On the same materialized instance pools this script
re-evaluates only the neural methods with a raised cap (K_eval=400) to quantify
the relative contribution of the cap constraint versus policy quality
degradation:
  EFEDQN / EFEDQN-E / EFormer @ 200n_128tw and 300n_192tw, max_steps=400.

Output: results/exp6_horizon/per_instance.csv (+ summary).
Usage:  python exp6_horizon_analysis.py [--max-steps 400]
"""

import argparse
import os
import time

import exp_common as ec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--num-instances', type=int, default=30)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp6_horizon'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the materialized instance pool of exp6ext (seeded, the same pool);
    # under the parallel schedule the exp6ext job may not have finished writing
    # the pool yet, so poll for up to 2 hours
    methods = {}
    methods["EFormer"] = ec.EFormerSolver(
        os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=1,
        max_steps=args.max_steps)
    s = ec.make_efedqn_solvers(device=device)
    methods["EFEDQN"] = s["EFEDQN"]
    methods["EFEDQN-E"] = s["EFEDQN-E"]

    import glob, json
    conds = ["200n_128tw", "300n_192tw"]
    for _ in range(240):  # wait at most 2 hours (30s x 240)
        n_ready = sum(len(glob.glob(os.path.join(
            ec.RESULTS_DIR, "_jobs", "exp6ext_TD-ACO", "instances",
            cond, "*.json"))) for cond in conds)
        if n_ready >= 2 * args.num_instances:
            break
        print(f"[horizon] 等待 exp6ext 实例池就绪({n_ready}/"
              f"{2 * args.num_instances})...", flush=True)
        time.sleep(30)

    rows = []
    for cond in conds:
        insts = []
        for p in sorted(glob.glob(os.path.join(
                ec.RESULTS_DIR, "_jobs", "exp6ext_TD-ACO", "instances",
                cond, "*.json")))[:args.num_instances]:
            with open(p, encoding="utf-8") as f:
                insts.append(json.load(f))
        print(f"[horizon-{args.max_steps}] {cond}: {len(insts)} 实例", flush=True)
        if not insts:
            print(f"[horizon] {cond} 实例池仍为空, 跳过(请先跑 exp6ext)", flush=True)
            continue
        rows.extend(ec.run_methods_on_instances(
            methods, insts, cond, args.max_steps))

    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), rows)
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"),
                 ec.summarize(rows))
    print("\n=== 汇总(max_steps=%d) ===" % args.max_steps)
    ec.print_summary(ec.summarize(rows))


if __name__ == "__main__":
    main()
