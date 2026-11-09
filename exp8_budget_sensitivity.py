# -*- coding: utf-8 -*-
"""exp8_budget_sensitivity.py — budget sensitivity of the metaheuristic baselines.

This script reruns the three metaheuristics with alternate budget settings
(TD-ACO 100 ants x 1000 iterations or 50 x 100; GA/PSO 1000 or 100 generations)
on exactly the same instance pools, seed derivation and unified replay scoring
as the main comparison, changing only the budget parameters:

  pools:
    e1_<cond>  exp1 instance pool: generated with derive_seed("exp1", cond, i),
               solve seed = derive_seed(cond, id, method name), bitwise
               identical to the main comparison rows.
    e2_50n     exp2 certified 50-node subset: the Gurobi certified opt_cost is
               embedded in each instance, the condition name is n50 and the
               solve seeds match the K=1 rows bitwise, so the optimality gap can
               be reported.
  budget levels:
    default    aco: 50 ants x 100; ga/pso: 200 x 100 -- used for protocol
               validation only, the results must reproduce the corresponding
               rows.
    original   aco: 100 ants x 1000; ga/pso: 200 x 1000.

Usage (inside test/):
  python exp8_budget_sensitivity.py --kind aco --budget default --num 3 --pools e1_100n_64tw  # protocol check
  python exp8_budget_sensitivity.py --kind pso --budget original --pools e2_50n,e1_100n_64tw  # final run
  (--params "w=0.7" overrides any parameter, e.g. the PSO inertia weight)

Output: results/exp8_budget/{kind}_{budget}_{pool}/per_instance.csv + summary.csv
(the time column is the wall clock of the parallel job and is not used; only
pass/cost/gap are reported.)
"""

import argparse
import os

import exp_common as ec

# Budget parameters of the `original` level; only budget-related items differ
# from the default level.
ORIGINAL_BUDGET = {
    "aco": {"ant_num": 100, "max_iter": 1000},
    "ga": {"max_iter": 1000},
    "pso": {"max_iter": 1000},
}

METHOD_NAME = {"aco": "TD-ACO", "ga": "GA", "pso": "PSO"}


def parse_params(s):
    out = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        out[k.strip()] = float(v) if "." in v else int(v)
    return out


def build_e1_pool(cond_name, num):
    """exp1 protocol instance pool: bitwise identical to exp1_main_comparison.py."""
    cond = next(c for c in ec.CONDITIONS if c[0] == cond_name)
    name, n, k_v, t_num, _ = cond
    instances = []
    for i in range(num):
        seed = ec.derive_seed("exp1", name, i)
        inst = ec.gen_random_instance(n, k_v, t_num, seed)
        inst["id"] = f"{name}_i{i:03d}"
        instances.append(inst)
    return name, instances, None


def build_e2_pool(scale):
    """exp2 certified pool subset: instances and gap references are the same as
    in exp2_quality_fair.py."""
    pool = ec.load_benchmark_pool()
    pool = [(i, r) for (i, r) in pool if i["nodes"] == scale]
    if not pool:
        raise SystemExit(f"认证池为空: nodes={scale}")
    insts = [i for (i, _) in pool]
    refs = {i["id"]: r for (i, r) in pool}
    return f"n{scale}", insts, refs


def main():
    parser = argparse.ArgumentParser(description="元启发式预算敏感性实验")
    parser.add_argument("--kind", type=str, required=True,
                        choices=["aco", "ga", "pso"])
    parser.add_argument("--budget", type=str, default="original",
                        choices=["default", "original"])
    parser.add_argument("--pools", type=str, default="e2_50n,e1_100n_64tw",
                        help="逗号分隔: e1_<cond> 取 exp_common.CONDITIONS;"
                             " e2_<n> 取认证池该规模子集")
    parser.add_argument("--num", type=int, default=50,
                        help="e1 池每条件实例数(exp1 协议为 50)")
    parser.add_argument("--params", type=str, default="",
                        help="逗号分隔 k=v, 最终覆盖(如 w=0.7)")
    parser.add_argument("--max-steps", type=int,
                        default=ec.DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    # Parameter assembly: budget-level overrides first, command-line last
    param_key = {"aco": "aco_params", "ga": "ga_params",
                 "pso": "pso_params"}[args.kind]
    overrides = dict(ORIGINAL_BUDGET[args.kind]
                     if args.budget == "original" else {})
    overrides.update(parse_params(args.params))
    solver = ec.HeuristicSolver(args.kind, **{param_key: overrides})
    eff = {"aco": solver.aco_params, "ga": solver.ga_params,
           "pso": solver.pso_params}[args.kind]
    print(f"kind={args.kind} budget={args.budget} max_steps={args.max_steps}")
    print(f"effective params: {eff} (overrides={overrides})")

    mname = METHOD_NAME[args.kind]
    out_root = os.path.join("results", "exp8_budget")
    os.makedirs(out_root, exist_ok=True)

    for pool_tag in [p.strip() for p in args.pools.split(",") if p.strip()]:
        if pool_tag.startswith("e1_"):
            cond_name, instances, refs = build_e1_pool(pool_tag[3:], args.num)
        elif pool_tag.startswith("e2_"):
            cond_name, instances, refs = build_e2_pool(
                int(pool_tag[3:].rstrip("n")))
        else:
            raise SystemExit(f"未知池: {pool_tag}")

        print(f"\n=== [{pool_tag}] {cond_name}: {len(instances)} 实例 ===")
        rows = ec.run_methods_on_instances(
            {mname: solver}, instances, cond_name,
            args.max_steps, gap_refs=refs)

        out_dir = os.path.join(out_root,
                               f"{args.kind}_{args.budget}_{pool_tag}")
        os.makedirs(out_dir, exist_ok=True)
        ec.write_csv(os.path.join(out_dir, "per_instance.csv"), rows)
        ec.write_csv(os.path.join(out_dir, "summary.csv"), ec.summarize(rows))
        ec.print_summary(ec.summarize(rows))
        print(f"完成 {len(rows)} 行 -> {out_dir}")


if __name__ == "__main__":
    main()
