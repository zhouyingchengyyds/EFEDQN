# -*- coding: utf-8 -*-
"""exp2_quality_fair.py — solution quality on a certified pool under fair budgets.

Combines the solution-quality check with a budget-fair comparison:
- instances: benchmark certified pool (n10x50 / n20x50 / n30x50 / n50x50, all
  proven OPTIMAL by Gurobi; the pool is generated so that "the no-window
  optimum is infeasible with time windows", which also makes it a useful
  subset for temporal diagnosis);
- certified reference: opt_cost from benchmark/solver_results.csv (also
  embedded as the solver field of each instance JSON);
- methods x three budget levels:
    [K=1]    single run: TD-ACO/GA/PSO (1 restart), EFormer (greedy), EFEDQN,
             EFEDQN-noTW (static encoder without time-window features),
             RollingHorizon (time-dependent shortest-path heuristic)
    [K=10]   equal number of forward trajectories: TD-ACO/GA/PSO (10
             independent restarts), EFormer-K10 (10 samples), EFEDQN-E
             (best-of-10 over 10 permutations)
    [wall]   equal wall clock: each method restarts or rolls out repeatedly
             within the same per-instance budget --wall-clock (default 10s)
             and keeps the best result
- metrics: optimality gap% (vs the certified optimum), condition cost, pass
  rate, time per instance;
- shared evaluation step cap --max-steps (default 200, consistent with the
  other experiments; the step caps 50/50/60 of the certified solver are only
  solving parameters used when generating the references and do not constrain
  the evaluated methods).

Usage:
  python exp2_quality_fair.py                     # all
  python exp2_quality_fair.py --num-skip-heavy    # (example) use --methods to trim
  python exp2_quality_fair.py --wall-clock 10
"""

import argparse
import os
import time

import exp_common as ec


class BudgetWrap:
    """Equal wall-clock wrapper: repeatedly calls the wrapped solver within the
    time budget and keeps the best (feasible, -cost)."""

    def __init__(self, solver, time_budget_s, tag):
        self.solver = solver
        self.budget = time_budget_s
        self.tag = tag

    def solve(self, env, seed=1, max_steps=ec.DEFAULT_MAX_STEPS, init=None,
              start_time=None):
        t0 = time.perf_counter()
        best = None
        n = 0
        while n == 0 or (time.perf_counter() - t0) < self.budget:
            r = self.solver.solve(env, seed=ec.derive_seed(seed, "w", n),
                                  max_steps=max_steps, init=init,
                                  start_time=start_time)
            best = ec.better_result(best, r)
            n += 1
        best["time_ms"] = (time.perf_counter() - t0) * 1000.0
        best["n_restarts"] = n
        return best


def build_methods(args, device, only=None):
    M = {}
    aco = lambda **kw: ec.HeuristicSolver("aco", **kw)          # noqa: E731
    ga = lambda **kw: ec.HeuristicSolver("ga", **kw)            # noqa: E731
    pso = lambda **kw: ec.HeuristicSolver("pso", **kw)          # noqa: E731
    efr = lambda k, **kw: ec.EFormerSolver(                     # noqa: E731
        os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=k,
        max_steps=args.max_steps, **kw)
    efq = ec.make_efedqn_solvers(device=device)

    if "K=1" in args.budgets:
        M["TD-ACO[K=1]"] = aco(restarts=1)
        M["GA[K=1]"] = ga(restarts=1)
        M["PSO[K=1]"] = pso(restarts=1)
        M["EFormer[K=1]"] = efr(k=1)
        M["EFEDQN[K=1]"] = efq["EFEDQN"]
        M["EFEDQN-noTW[K=1]"] = efq["EFEDQN-noTW"]
        M["RollingHorizon[K=1]"] = ec.RollingHorizonAdapter(
            max_steps=args.max_steps, two_opt=True)
    if "K=10" in args.budgets:
        M["TD-ACO[K=10]"] = aco(restarts=10)
        M["GA[K=10]"] = ga(restarts=10)
        M["PSO[K=10]"] = pso(restarts=10)
        M["EFormer[K=10]"] = efr(k=10, name="EFormer[K=10]")
        M["EFEDQN-E[K=10]"] = efq["EFEDQN-E"]
        M["EFEDQN-noTW[K=10]"] = ec.EFEDQNSolver(
            os.path.join(ec.TEST_DIR, "No_timwin", "model_seed1_no_tw.pth"),
            device, k=10, no_tw=True, name="EFEDQN-noTW[K=10]")
    if "wall" in args.budgets:
        w = args.wall_clock
        M[f"TD-ACO[wall={w}s]"] = BudgetWrap(aco(restarts=1), w, "aco")
        M[f"GA[wall={w}s]"] = BudgetWrap(ga(restarts=1), w, "ga")
        M[f"PSO[wall={w}s]"] = BudgetWrap(pso(restarts=1), w, "pso")
        M[f"EFormer[wall={w}s]"] = BudgetWrap(efr(k=1), w, "eformer")
        M[f"EFEDQN[wall={w}s]"] = BudgetWrap(efq["EFEDQN"], w, "efedqn")
        M[f"EFEDQN-E[wall={w}s]"] = BudgetWrap(efq["EFEDQN-E"], w, "efedqn-e")
        M[f"EFEDQN-noTW[wall={w}s]"] = BudgetWrap(efq["EFEDQN-noTW"], w, "efedqn-notw")
    # --methods filter: keep only methods whose protocol name contains any of
    # the given substrings (used to splice reruns of selected protocols)
    if only:
        M = {k: v for k, v in M.items()
             if any(s in k for s in only)}
    return M


def gurobi_rows(pool):
    """Gurobi reference rows: the certified optimum and its runtime come from
    the solve record made when the pool was generated."""
    rows = []
    for inst, ref in pool:
        if ref is None:
            continue
        rows.append({
            "condition": f"n{inst['nodes']}", "nodes": inst["nodes"],
            "tw": inst["num_windows"], "targets": inst["num_targets"],
            "instance_id": inst["id"], "protocol": "Gurobi(exact)",
            "cost": ref["opt_cost"], "feasible": 1, "completed": 1,
            "steps": "", "n_targets": inst["num_targets"],
            "invalid": 0, "window_viol": 0,
            "time_ms": ref["wall_time_s"] * 1000.0,
            "gap_pct": 0.0, "gap_kind": "opt",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="表8(认证池解质量+公平预算合并表): C3+C4")
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS,
                        help='统一评估步数上限(默认 200)')
    parser.add_argument('--budgets', type=str, default="K=1,K=10,wall",
                        help='预算档: K=1,K=10,wall 的逗号子集')
    parser.add_argument('--wall-clock', type=float, default=10.0,
                        help='等时钟档的每实例时间预算(秒)')
    parser.add_argument('--scales', type=str, default="10,20,30,50",
                        help='认证池规模子集')
    parser.add_argument('--methods', type=str, default=None,
                        help='可选: 逗号分隔协议名子串, 只重跑匹配的方法'
                             '(用于重跑受影响协议后按行拼接, 不写整表)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp2_quality'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    args.budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Cap the main-process torch thread count: this prevents oversubscription
    # of the shared CPU by the multi-threaded main process (EFormer encoding)
    # and the 10 worker processes of EFEDQN-E
    torch.set_num_threads(max(2, (os.cpu_count() or 8) // 4))
    # Pre-warm the process pool for EFEDQN-E parallel rollouts (moves worker
    # weight loading out of the first-instance timing)
    if "EFEDQN-E" in (args.methods or "EFEDQN-E"):
        ec.prewarm_efedqn_pool(
            os.path.join(ec.EFEDQN_DIR, "model_seed1.pth"), None,
            device, k=10)
    if "K=10" in args.budgets and "noTW" in (args.methods or "noTW"):
        ec.prewarm_efedqn_pool(
            os.path.join(ec.TEST_DIR, "No_timwin", "model_seed1_no_tw.pth"),
            None, device, k=10)

    pool = ec.load_benchmark_pool()
    scales = {int(s) for s in args.scales.split(",")}
    pool = [(i, r) for (i, r) in pool if i["nodes"] in scales]
    if not pool:
        raise SystemExit(f"认证池为空: {ec.BENCHMARK_DIR}")
    print(f"认证池: {len(pool)} 实例 | 预算档: {args.budgets} | "
          f"max_steps: {args.max_steps} | 设备: {device}")

    # --methods filter: comma-separated protocol name substrings; only matching
    # methods are rerun and their rows are spliced into the existing output.
    # The string is split into a set before it is passed to build_methods.
    only = None
    if args.methods:
        only = {m.strip() for m in args.methods.split(",") if m.strip()}
    methods = build_methods(args, device, only=only)
    if not methods:
        raise SystemExit("--methods 过滤后没有任何方法可跑")
    print(f"方法: {list(methods)}")

    by_scale = {}
    for inst, ref in pool:
        by_scale.setdefault(inst["nodes"], []).append((inst, ref))

    all_rows = gurobi_rows(pool)
    for n in sorted(by_scale):
        sub = by_scale[n]
        insts = [i for (i, _) in sub]
        refs = {i["id"]: r for (i, r) in sub}
        print(f"\n=== n={n}: {len(insts)} 实例 ===")
        rows = ec.run_methods_on_instances(
            methods, insts, f"n{n}", args.max_steps, gap_refs=refs)
        all_rows.extend(rows)
        ec.print_summary(ec.summarize(rows))

    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), all_rows)
    summary = ec.summarize(all_rows)
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"), summary)

    # Tabular pivot (scale x method): gap / cost / pass / time
    pivot_path = os.path.join(args.output_dir, "table_quality_fair.txt")
    with open(pivot_path, "w", encoding="utf-8") as f:
        f.write(f"{'scale':>5} {'method':>22} {'gap%':>8} {'cost(feas)':>11} "
                f"{'pass':>6} {'time(s)':>8}\n")
        for s in summary:
            f.write(f"{s['condition']:>5} {s['protocol']:>22} "
                    f"{s['mean_gap_pct']:>8.2f} "
                    f"{s['mean_cost_feasible']:>11.2f} "
                    f"{s['pass_rate']:>6.3f} "
                    f"{s['mean_time_ms'] / 1000.0:>8.3f}\n")
    print(f"\n论文表格(透视): {pivot_path}")


if __name__ == "__main__":
    main()
