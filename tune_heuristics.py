# -*- coding: utf-8 -*-
"""tune_heuristics.py — quick parameter-tuning comparison for the three Heuristic algorithms (aco/ga/pso).

On a fixed instance pool (seed-derived, generated with the same protocol as
exp1), HeuristicSolver is run under different parameter configurations and
feasible/completed/viol/steps/cost/time are printed per instance, for comparing
pass rates before and after tuning.

Usage (inside test/):
  python tune_heuristics.py --kind ga                          # current default baseline
  python tune_heuristics.py --kind ga  --params "pop_size=200,tour_size=4"
  python tune_heuristics.py --kind aco --params "alpha=0.05"
  python tune_heuristics.py --kind pso  --cond 50n_32tw --num 5
  python tune_heuristics.py --kind ga  --diagnose              # also print the terminal node of the solution
"""

import argparse

import exp_common as ec


def parse_params(s):
    """'pop_size=200,alpha=0.05' -> dict; values containing '.' are floats,
    others ints."""
    out = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        out[k.strip()] = float(v) if "." in v else int(v)
    return out


def final_node(env, actions):
    """Replay the action sequence and return the terminal node (for diagnosis; a
    feasible solution must end at the depot, node 0)."""
    env.reset_env()
    env.reset_agent_pos(0)
    env.reset_agent_time(0.0)
    pos = 0
    for a in actions:
        if a is None:
            continue
        nxt = env.map[pos][a][0]
        if nxt != -1:
            pos = nxt
    return pos


def diagnose_final_node(kind, env, params, max_steps):
    """Run GA/PSO once directly and return the terminal node of the solution
    (same path as HeuristicSolver)."""
    if kind == "ga":
        from GA import GA
        ga = GA(env=env, start_pos=0, start_time=0.0, **params)
        sol = ga.optimize()
        return final_node(env, sol.actions), len(sol.actions)
    if kind == "pso":
        from PSO import PSO
        pso = PSO(env=env, start_pos=0, start_time=0.0, **params)
        sol = pso.optimize()
        return final_node(env, sol.actions), len(sol.actions)
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="Heuristic 调参对比工具")
    parser.add_argument("--kind", type=str, required=True,
                        choices=["aco", "ga", "pso"])
    parser.add_argument("--params", type=str, default="",
                        help="逗号分隔的 k=v 参数, 覆盖 exp_common 默认值")
    parser.add_argument("--cond", type=str, default="25n_16tw",
                        help="实验条件名(exp_common.CONDITIONS)")
    parser.add_argument("--num", type=int, default=10, help="实例数")
    parser.add_argument("--max-steps", type=int,
                        default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument("--diagnose", action="store_true",
                        help="ga/pso: 直接运行一次并打印解的终点节点")
    args = parser.parse_args()

    cond = next(c for c in ec.CONDITIONS if c[0] == args.cond)
    name, n, k_v, t_num, _ = cond
    overrides = parse_params(args.params)

    param_key = {"aco": "aco_params", "ga": "ga_params",
                 "pso": "pso_params"}[args.kind]
    solver = ec.HeuristicSolver(args.kind, **{param_key: overrides})
    default_params = {"aco": solver.aco_params, "ga": solver.ga_params,
                      "pso": solver.pso_params}[args.kind]
    print(f"kind={args.kind} cond={name} num={args.num} "
          f"max_steps={args.max_steps}")
    print(f"effective params: {default_params} "
          f"(overrides={overrides})")

    n_pass = 0
    costs, times = [], []
    for i in range(args.num):
        seed = ec.derive_seed(name, i, "tune")
        inst = ec.gen_random_instance(n, k_v, t_num, seed)
        env = ec.env_from_inst(inst)

        r = solver.solve(env, seed=ec.derive_seed(seed, args.kind),
                         max_steps=args.max_steps)
        if r["feasible"]:
            n_pass += 1
            costs.append(r["cost"])
        times.append(r["time_ms"] / 1000.0)

        tail = ""
        if args.diagnose and args.kind in ("ga", "pso"):
            fn, n_act = diagnose_final_node(args.kind, env, default_params,
                                            args.max_steps)
            tail = f" | final_node={fn} (depot=0) n_actions={n_act}"
        print(f"[{i:02d}] feasible={int(r['feasible'])} "
              f"completed={int(r['completed'])} "
              f"viol=({r['invalid']},{r['window_viol']}) "
              f"steps={r['steps']} cost={r['cost']:.0f} "
              f"t={r['time_ms']/1000.0:.2f}s{tail}")

    mean_cost = sum(costs) / len(costs) if costs else float("nan")
    print(f"\npass rate: {n_pass}/{args.num} = {n_pass/args.num:.2f} | "
          f"mean cost(feas): {mean_cost:.1f} | "
          f"mean time: {sum(times)/len(times):.2f}s")


if __name__ == "__main__":
    main()
