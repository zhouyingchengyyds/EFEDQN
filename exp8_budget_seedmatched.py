# -*- coding: utf-8 -*-
"""exp8_budget_seedmatched.py — 50-node budget experiment with seeds matched bitwise to the K=1 rows.

Background: exp8_budget_sensitivity.py uses the method keys "TD-ACO"/"GA"/"PSO",
whereas the K=1 rows use keys such as "TD-ACO[K=1]". Because
exp_common.run_methods_on_instances derives seeds from the method key, the two
runs use the same certified pool but different random streams and cannot be
paired directly. This script fixes the method key to "<METHOD>[K=1]", so the
default level matches the K=1 rows bitwise; the `original` level only changes
the budget parameters, which makes the budget comparison strictly paired.

Usage (inside test/):
  python exp8_budget_seedmatched.py --kinds aco --budget default --limit 1
  python exp8_budget_seedmatched.py --kinds aco,ga,pso --budget original --workers 12
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import exp_common as ec
from exp8_budget_sensitivity import METHOD_NAME, ORIGINAL_BUDGET

CONDITION = "n50"
PARAM_KEY = {"aco": "aco_params", "ga": "ga_params", "pso": "pso_params"}


def solve_one(task):
    """Evaluate a single (instance, method, budget) task; a top-level function
    so the process pool can pickle it."""
    kind, budget, inst, ref = task
    overrides = dict(ORIGINAL_BUDGET[kind] if budget == "original" else {})
    solver = ec.HeuristicSolver(kind, **{PARAM_KEY[kind]: overrides})
    key = f"{METHOD_NAME[kind]}[K=1]"
    env = ec.env_from_inst(inst)
    seed = ec.derive_seed(CONDITION, inst["id"], key)
    m = solver.solve(env, seed=seed, max_steps=ec.DEFAULT_MAX_STEPS)
    row = ec.standard_row(CONDITION, inst, key, m)
    ec.attach_gap(row, ref)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", type=str, default="aco,ga,pso")
    parser.add_argument("--budget", choices=["default", "original"],
                        default="original")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    pool = [(i, r) for (i, r) in ec.load_benchmark_pool()
            if i["nodes"] == 50]
    pool.sort(key=lambda x: x[0]["id"])
    if args.limit:
        pool = pool[:args.limit]
    print(f"pool: {len(pool)} instances (n50) | budget={args.budget} "
          f"| workers={args.workers}")

    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        tasks = [(kind, args.budget, inst, ref) for (inst, ref) in pool]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(solve_one, tasks))
        out_dir = os.path.join("results", "exp8_budget_seedmatched",
                               f"{kind}_{args.budget}_e2_50n")
        ec.write_csv(os.path.join(out_dir, "per_instance.csv"), rows)
        summary = ec.summarize(rows)
        ec.write_csv(os.path.join(out_dir, "summary.csv"), summary)
        ec.print_summary(summary)


if __name__ == "__main__":
    main()
