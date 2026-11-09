"""Generate the LaTeX table body for solution-quality validation from
benchmark/solver_results.csv (Gurobi exact rows) and optionally
results/per_instance_pool.csv + results/summary_pool.csv (EFEDQN / EFEDQN-E rows).

Baseline rows are emitted as placeholders.

Run:  python gen_table.py --benchmark benchmark [--results results]
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

SCALES = [10, 20, 30, 50]
BASELINES = ["TD-ACO", "GA", "PSO", "EFormer"]
METHODS = ["EFEDQN", "EFEDQN-E"]


def load_solver(benchmark_dir):
    rows = []
    path = os.path.join(benchmark_dir, "solver_results.csv")
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["nodes"] = int(row["nodes"])
                row["opt_cost"] = float(row["opt_cost"])
                row["bound"] = float(row["bound"]) if row["bound"] else None
                row["wall_time_s"] = float(row["wall_time_s"])
                row["gap_to_bound_pct"] = (float(row["gap_to_bound_pct"])
                                           if row["gap_to_bound_pct"] else None)
                rows.append(row)
        return rows

    # fallback: rebuild rows from the per-instance JSONs (e.g. after an interrupted
    # run of instances.py, which writes the CSV only at the very end)
    for jp in sorted(glob.glob(os.path.join(benchmark_dir, "n*", "n*.json"))):
        with open(jp, encoding="utf-8") as f:
            inst = json.load(f)
        s = inst["solver"]
        rows.append({
            "instance_id": inst["id"], "nodes": inst["nodes"],
            "tw": inst["num_windows"], "targets": inst["num_targets"],
            "status": s["status"], "opt_cost": float(s["cost"]),
            "bound": float(s["bound"]) if s.get("bound") else None,
            "gap_to_bound_pct": (float(s["gap_to_bound_pct"])
                                 if s.get("gap_to_bound_pct") is not None else None),
            "wall_time_s": float(s["wall_time_s"]),
            "replay_match": s.get("replay_match"),
        })
    return rows


def load_method_summary(results_dir):
    path = os.path.join(results_dir, "summary_pool.csv")
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["condition"], row["protocol"])] = row
    return out


def fmt(x, nd=2):
    if x is None or x == "":
        return r"\ldots"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return r"\ldots"
    if v != v:  # NaN
        return r"\ldots"
    return f"{v:.{nd}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="benchmark")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    solver_rows = load_solver(args.benchmark)
    method_summary = load_method_summary(args.results)

    by_scale = defaultdict(list)
    for r in solver_rows:
        by_scale[r["nodes"]].append(r)

    lines = []
    notes = []
    for si, n in enumerate(SCALES):
        rows = by_scale.get(n, [])
        if not rows:
            continue
        n_inst = len(rows)
        n_opt = sum(1 for r in rows if r["status"] == "OPTIMAL")
        mean_cost = sum(r["opt_cost"] for r in rows) / n_inst
        mean_time = sum(r["wall_time_s"] for r in rows) / n_inst
        gaps = [r["gap_to_bound_pct"] for r in rows if r["gap_to_bound_pct"] is not None]
        mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0

        tag = f"\\multirow{{7}}{{*}}{{{n} nodes}}"
        if si > 0:
            lines.append("\\midrule")
        gap_str = f"{mean_gap:.2f}" if n_opt == n_inst else f"{mean_gap:.2f}$^{{\\dagger}}$"
        if n_opt < n_inst:
            notes.append(f"{n} nodes: optimality proven on {n_opt}/{n_inst} instances; "
                         "gap measured against the certified best bound on the remaining ones.")
        lines.append(f"{tag} & Gurobi (exact) & {gap_str} & {mean_cost:.2f} & 1.000 & {mean_time:.1f} \\\\")
        for b in BASELINES:
            lines.append(f" & {b} & \\ldots & \\ldots & \\ldots & \\ldots \\\\")
        for m in METHODS:
            s = method_summary.get((f"n{n}", m))
            if s:
                gap = fmt(s["mean_gap_pct"])
                feas = fmt(float(s["pass_rate"]), 3)
                cost = fmt(s["mean_cost_feasible"])
                t = fmt(float(s["mean_time_ms"]) / 1000.0, 1)
                lines.append(f" & {m} & {gap} & {cost} & {feas} & {t} \\\\")
            else:
                lines.append(f" & {m} & \\ldots & \\ldots & \\ldots & \\ldots \\\\")

    print("\n".join(lines))
    if notes:
        print()
        print("% notes for the caption / footnote:")
        for nt in notes:
            print("%", nt)


if __name__ == "__main__":
    main()
