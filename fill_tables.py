# -*- coding: utf-8 -*-
"""fill_tables.py — generate LaTeX fragments for the result tables from merged experiment results.

Reads summary/per_instance CSVs under results/<experiment>/ and prints the LaTeX
rows of each table (same column layout as the tables) for direct inclusion in
the LaTeX document:
- main comparison:     exp1_main    (pass / fitness / cost(feas) / time;
                                     condition x method)
- quality and budgets: exp2_quality (gap / cost / pass / time;
                                     scale x method x budget level)
- temporal diagnosis:  exp3_diag    (pass / cost / gap; variants)
- component ablation:  exp5_ablation (per-variant mean±std; K rows)
- disruption:          exp4_disruption (feas / incr / latency;
                                     disruption point x method)
- scale extrapolation: exp6_general extrap
- distribution shift:  exp6_general distshift
- real network:        exp7_real
It also prints summary numbers.
"""

import csv
import os
import sys

import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(TEST_DIR, "results")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


CLEAN = os.path.join(RESULTS, "clean_timing")


def clean_time(csv_name, cond, method):
    """Average seconds for (cond, method) from the clean_rerun timing CSV."""
    rows = read_csv(os.path.join(CLEAN, csv_name))
    if not rows:
        return None
    ts = [float(r["time_ms"]) for r in rows
          if r["condition"] == cond and r["method"] == method]
    if not ts:
        return None
    return sum(ts) / len(ts) / 1000.0


def f(x, nd=1, scale=1.0):
    try:
        v = float(x) * scale
    except (TypeError, ValueError):
        return "--"
    if v != v:
        return "--"
    return f"{v:.{nd}f}"


def pct(x, nd=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v != v:
        return "--"
    return f"{100*v:.{nd}f}"


def get(rows, cond, proto, key, nd=1, scale=1.0):
    for r in rows:
        if r["condition"] == cond and r["protocol"] == proto:
            return f(r[key], nd, scale)
    return "--"


def _row(rows, cond, proto):
    for r in rows:
        if r["condition"] == cond and r["protocol"] == proto:
            return r
    return None


def table_exp1_simple():
    rows = read_csv(os.path.join(RESULTS, "exp1_main", "summary.csv"))
    if not rows:
        return
    methods = ["TD-ACO", "GA", "PSO", "EFormer", "EFormer-K10", "EFEDQN",
               "EFEDQN-E"]
    groups = [("TABLE 6 (scale)",
               [("25 nodes", "25n_16tw"), ("50 nodes", "50n_32tw"),
                ("100 nodes", "100n_64tw")]),
              ("TABLE 7 (tw)",
               [("32 Time window", "100n_32tw"),
                ("64 Time window", "100n_64tw"),
                ("128 Time window", "100n_128tw")])]
    for title, conds in groups:
        print(f"\n===== {title} =====")
        for label, cond in conds:
            print(f"% --- {label} ---")
            for m in methods:
                r = _row(rows, cond, m)
                if r is None:
                    print(f" & {m:<12} & -- & -- & -- & -- \\\\")
                    continue
                ct = clean_time("exp1_timing.csv", cond, m)
                tstr = f"{ct:.2f}" if ct is not None else "TBD"
                print(f" & {m:<12} & {pct(r['pass_rate'])} & "
                      f"{f(r['mean_fitness'], 4)} & "
                      f"{f(r['mean_cost_feasible'], 1)} & "
                      f"{tstr} \\\\")


def table_exp2():
    rows = read_csv(os.path.join(RESULTS, "exp2_quality", "summary.csv"))
    if not rows:
        return
    scales = ["n10", "n20", "n30", "n50"]
    k1 = [("TD-ACO", "TD-ACO[K=1]"), ("GA", "GA[K=1]"), ("PSO", "PSO[K=1]"),
          ("EFormer", "EFormer[K=1]"), ("EFEDQN", "EFEDQN[K=1]"),
          ("EFEDQN-noTW", "EFEDQN-noTW[K=1]"),
          ("RollingHorizon", "RollingHorizon[K=1]")]
    k10 = [("TD-ACO (10 restarts)", "TD-ACO[K=10]"),
           ("GA (10 restarts)", "GA[K=10]"), ("PSO (10 restarts)", "PSO[K=10]"),
           ("EFormer (10 samples)", "EFormer[K=10]"),
           ("EFEDQN-E (best of 10)", "EFEDQN-E[K=10]"),
           ("EFEDQN-noTW (best of 10)", "EFEDQN-noTW[K=10]")]
    wall = [("TD-ACO", "TD-ACO[wall=10.0s]"), ("GA", "GA[wall=10.0s]"),
            ("PSO", "PSO[wall=10.0s]"), ("EFormer", "EFormer[wall=10.0s]"),
            ("EFEDQN", "EFEDQN[wall=10.0s]"),
            ("EFEDQN-E", "EFEDQN-E[wall=10.0s]"),
            ("EFEDQN-noTW", "EFEDQN-noTW[wall=10.0s]")]

    # Gurobi reference rows (older data files label the protocol "CP-SAT(exact)",
    # accepted for compatibility)
    print("\n===== TABLE 8: Gurobi reference rows =====")
    for s in scales:
        r = (_row(rows, s, "Gurobi(exact)")
             or _row(rows, s, "CP-SAT(exact)"))
        if r:
            print(f"{s[1:]} nodes & Gurobi (exact) & 0.00 & "
                  f"{f(r['mean_cost_feasible'], 1)} & 1.000 & "
                  f"{f(r['mean_time_ms'], 1, 0.001)} \\\\")
    for title, mapping in [("--- K=1 regime ---", k1),
                           ("--- K=10 regime ---", k10),
                           ("--- wall regime ---", wall)]:
        print(f"\n===== TABLE 8: {title} =====")
        for s in scales:
            print(f"% --- {s} ---")
            for disp, proto in mapping:
                r = _row(rows, s, proto)
                if r is None:
                    print(f" & {disp:<24} & -- & -- & -- & -- \\\\")
                    continue
                print(f" & {disp:<24} & {f(r['mean_gap_pct'], 2)} & "
                      f"{f(r['mean_cost_feasible'], 1)} & "
                      f"{pct(r['pass_rate'])} & "
                      f"{f(r['mean_time_ms'], 1, 0.001)} \\\\")


def table_exp3():
    rows = read_csv(os.path.join(RESULTS, "exp3_diag", "summary.csv"))
    if not rows:
        return
    # Aggregate instances over all scales (re-aggregating per-instance rows is
    # more accurate)
    per = read_csv(os.path.join(RESULTS, "exp3_diag", "per_instance.csv"))
    if per:
        print("\n===== TABLE 9 (aggregated over full pool) =====")
        for var in ["Full(EFEDQN)", "Full(EFEDQN-E)", "(a)No-TW",
                    "(b)Frozen", "(c)Mask-only"]:
            sub = [r for r in per if r["protocol"] == var]
            if not sub:
                print(f"{var:<28} & -- & -- & -- \\\\")
                continue
            n = len(sub)
            pr = sum(int(r["feasible"]) for r in sub) / n
            costs = [float(r["cost"]) for r in sub if str(r["feasible"]) == "1"]
            gaps = [float(r["gap_pct"]) for r in sub
                    if str(r["feasible"]) == "1" and r["gap_pct"] not in ("", "nan")
                    and float(r["gap_pct"]) == float(r["gap_pct"])]
            print(f"{var:<28} & {pct(pr)} & "
                  f"{f(np.mean(costs), 1) if costs else '--'} & "
                  f"{f(np.mean(gaps), 1) if gaps else '--'} \\\\")
    # Per-scale pass rates (for diagnosis)
    print("\n--- per-scale pass rates ---")
    for var in ["Full(EFEDQN)", "Full(EFEDQN-E)", "(a)No-TW", "(b)Frozen",
                "(c)Mask-only"]:
        sub = [r for r in per if r["protocol"] == var]
        if not sub:
            continue
        by = {}
        for r in sub:
            by.setdefault(r["condition"], []).append(int(r["feasible"]))
        line = " ".join(f"{c}={np.mean(v):.2f}" for c, v in sorted(by.items()))
        print(f"{var:<16} {line}")


def table_exp5():
    rows = read_csv(os.path.join(RESULTS, "exp5_ablation", "summary.csv"))
    if not rows:
        return
    print("\n===== TABLE 10 (ablation) =====")
    for r in rows:
        pr = f"{pct(r['pass_rate_mean'])} ± {pct(r['pass_rate_std'])}"
        cf = f"{f(r['cost_feas_mean'], 1)} ± {f(r['cost_feas_std'], 1)}"
        print(f"{r['variant']:<28} & {pr} & {cf} \\\\")


def table_exp4():
    rows = read_csv(os.path.join(RESULTS, "exp4_disruption", "summary.csv"))
    if not rows:
        return
    print("\n===== TABLE 11 (disruption) =====")
    # Aggregate by disruption point (kinds merged)
    per = read_csv(os.path.join(RESULTS, "exp4_disruption", "per_instance.csv"))
    if per:
        by = {}
        for r in per:
            key = (r["disrupt_point"], r["protocol"])
            by.setdefault(key, []).append(r)
        for (p, proto), rs in sorted(by.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
            feas = [int(r["feasible"]) for r in rs]
            incrs = [float(r["incr_cost"]) for r in rs
                     if str(r["feasible"]) == "1" and r["incr_cost"] not in ("", "nan")
                     and float(r["incr_cost"]) == float(r["incr_cost"])]
            lats = [float(r["latency_ms"]) for r in rs]
            print(f"p={p:<5} {proto:<22} & feas={np.mean(feas):.3f} & "
                  f"incr={f(np.mean(incrs), 1) if incrs else '--'} & "
                  f"lat(med)={np.median(lats):.1f}ms & n={len(rs)}")
    print("\n--- by point x kind ---")
    for r in rows:
        print(f"{r['condition']:<20} {r['protocol']:<22} "
              f"feas={f(r['feasible_rate'], 3)} incr={f(r['mean_incr_cost'], 1)} "
              f"lat={f(r['mean_latency_ms'], 1)}/{f(r['median_latency_ms'], 1)}ms "
              f"n={r['n']}")


def table_exp6():
    ext = read_csv(os.path.join(RESULTS, "exp6_general", "extrap_summary.csv"))
    if ext:
        print("\n===== TABLE 12 (scale extrapolation) =====")
        for cond in ["150n_96tw", "200n_128tw", "300n_192tw"]:
            print(f"% --- {cond} ---")
            for m in ["TD-ACO", "GA", "PSO", "EFormer", "EFEDQN", "EFEDQN-E"]:
                r = _row(ext, cond, m)
                if r is None:
                    print(f" & {m:<10} & -- & -- & -- \\\\")
                    continue
                ct = clean_time("exp6ext_timing.csv", cond, m)
                tstr = f"{ct:.2f}" if ct is not None else "TBD"
                print(f" & {m:<10} & {pct(r['pass_rate'])} & "
                      f"{f(r['mean_cost_feasible'], 1)} & "
                      f"{tstr} \\\\")
    ds = read_csv(os.path.join(RESULTS, "exp6_general", "distshift_summary.csv"))
    if ds:
        print("\n===== TABLE 13 (distribution shift) =====")
        conds = ["no-shift(ref)", "target-ratio-0.1", "target-ratio-0.5",
                 "restr-dense-16", "restr-dense-192", "win-short", "win-long",
                 "degree-thin"]
        for cond in conds:
            parts = []
            for m in ["EFEDQN", "EFEDQN-E", "EFormer"]:
                r = _row(ds, cond, m)
                if r is None:
                    parts.extend(["--", "--"])
                else:
                    parts.append(pct(r["pass_rate"]))
                    parts.append(f(r["mean_cost_feasible"], 1))
            print(f"{cond:<18} & " + " & ".join(parts) + " \\\\")


def table_exp7():
    rows = read_csv(os.path.join(RESULTS, "exp7_real", "summary.csv"))
    if not rows:
        return
    print("\n===== TABLE 14 (real network, per departure + aggregated) =====")
    per = read_csv(os.path.join(RESULTS, "exp7_real", "per_instance.csv"))
    if per:
        by = {}
        for r in per:
            by.setdefault(r["protocol"], []).append(r)
        print("--- aggregated ---")
        for m in ["TD-ACO", "GA", "PSO", "EFormer", "EFormer-K10", "EFEDQN",
                  "EFEDQN-E"]:
            rs = by.get(m, [])
            if not rs:
                print(f"{m:<12} & -- & -- & --")
                continue
            pr = np.mean([int(r["feasible"]) for r in rs])
            costs = [float(r["cost"]) for r in rs if str(r["feasible"]) == "1"]
            times = [float(r["time_ms"]) for r in rs]
            print(f"{m:<12} & pass={pr:.3f} & "
                  f"cost={f(np.mean(costs), 2) if costs else '--'} & "
                  f"t={np.mean(times)/1000:.2f}s")
        print("--- per departure ---")
        byc = {}
        for r in per:
            byc.setdefault((r["condition"], r["protocol"]), []).append(r)
        for (cond, m), rs in sorted(byc.items()):
            pr = np.mean([int(r["feasible"]) for r in rs])
            costs = [float(r["cost"]) for r in rs if str(r["feasible"]) == "1"]
            print(f"{cond:<14} {m:<12} pass={pr:.2f} "
                  f"cost={f(np.mean(costs), 2) if costs else '--'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        table_exp1_simple()
    if which in ("all", "2"):
        table_exp2()
    if which in ("all", "3"):
        table_exp3()
    if which in ("all", "4"):
        table_exp4()
    if which in ("all", "5"):
        table_exp5()
    if which in ("all", "6"):
        table_exp6()
    if which in ("all", "7"):
        table_exp7()
