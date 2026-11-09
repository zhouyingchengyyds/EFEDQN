# -*- coding: utf-8 -*-
"""merge_results.py — merge the parallel job outputs of run_all_experiments.py.

Each parallel job writes to results/_jobs/<name>/; this script merges rows of the
same kind into the standard location results/<experiment>/ and recomputes the
summary (instance pools are seed-derived and identical across jobs, so the
merged result equals a single sequential run):
- exp1  -> results/exp1_main/       (per_instance.csv + summary.csv + instances/)
- exp2  -> results/exp2_quality/    (per_instance.csv + summary.csv + pivot table)
- exp5  -> results/exp5_ablation/   (per_instance.csv + per-variant mean±std summary)
- exp6  -> results/exp6_general/    (extrap_*.csv + distshift_*.csv)
exp3/exp4/exp7 are single jobs that write directly to the standard location and
need no merging.

Usage:  python merge_results.py
"""

import csv
import glob
import os
import shutil

import numpy as np

import exp_common as ec

RESULTS = ec.RESULTS_DIR
JOBS = os.path.join(RESULTS, "_jobs")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        # Convert numeric columns back to float (everything is read as str)
        for r in rows:
            out = dict(r)
            for k in ("cost", "time_ms", "gap_pct", "steps"):
                if k in out and out[k] not in ("", None):
                    try:
                        out[k] = float(out[k])
                    except ValueError:
                        pass
            for k in ("feasible", "completed", "nodes", "tw", "targets"):
                if k in out and out[k] not in ("", None):
                    try:
                        out[k] = int(out[k])
                    except ValueError:
                        pass
            w.writerow(out)
    print(f"写出: {path} ({len(rows)} 行)")


def to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def numify(rows):
    """Convert string fields read from CSV to numbers (required by summarize)."""
    for r in rows:
        for k in ("cost", "time_ms", "gap_pct", "n_targets", "steps",
                  "invalid", "window_viol"):
            if k in r:
                r[k] = to_float(r[k], float("nan"))
        for k in ("feasible", "completed", "nodes", "tw", "targets"):
            if k in r:
                r[k] = int(to_float(r[k], 0))
    return rows


def merge_exp1():
    rows = []
    for p in sorted(glob.glob(os.path.join(JOBS, "exp1_*", "per_instance.csv"))):
        rows.extend(read_csv(p))
    if not rows:
        print("exp1: 无输出")
        return
    # Restore numeric column types (summarize needs float)
    numify(rows)
    write_csv(os.path.join(RESULTS, "exp1_main", "per_instance.csv"), rows)
    write_csv(os.path.join(RESULTS, "exp1_main", "summary.csv"),
              ec.summarize(rows))
    # Instance-pool directories (the same condition directory is identical across
    # jobs, so any copy is fine)
    for jd in sorted(glob.glob(os.path.join(JOBS, "exp1_*"))):
        src_root = os.path.join(jd, "instances")
        if not os.path.isdir(src_root):
            continue
        dst_root = os.path.join(RESULTS, "exp1_main", "instances")
        for cond in os.listdir(src_root):
            dst = os.path.join(dst_root, cond)
            if not os.path.exists(dst):
                shutil.copytree(os.path.join(src_root, cond), dst)
    print("exp1 合并完成")


def merge_exp2():
    # Prefer the single-process, contention-free rerun of clean_rerun.py (clean
    # timing columns; results match the parallel run under the same seeds); fall
    # back to merging the parallel jobs when clean_rerun has not been run.
    clean = os.path.join(RESULTS, "clean_timing", "exp2_quality",
                         "per_instance.csv")
    if os.path.exists(clean):
        rows = read_csv(clean)
        src_note = "clean"
    else:
        rows = []
        for p in sorted(glob.glob(os.path.join(JOBS, "exp2_*",
                                               "per_instance.csv"))):
            rows.extend(read_csv(p))
        src_note = "parallel"
    if not rows:
        print("exp2: 无输出")
        return
    print(f"exp2 数据来源: {src_note}")
    numify(rows)
    write_csv(os.path.join(RESULTS, "exp2_quality", "per_instance.csv"), rows)
    summary = ec.summarize(rows)
    write_csv(os.path.join(RESULTS, "exp2_quality", "summary.csv"), summary)
    pivot_path = os.path.join(RESULTS, "exp2_quality", "table_quality_fair.txt")
    with open(pivot_path, "w", encoding="utf-8") as f:
        f.write(f"{'scale':>5} {'method':>22} {'gap%':>8} {'cost(feas)':>11} "
                f"{'pass':>6} {'time(s)':>8}\n")
        for s in summary:
            f.write(f"{s['condition']:>5} {s['protocol']:>22} "
                    f"{s['mean_gap_pct']:>8.2f} "
                    f"{s['mean_cost_feasible']:>11.2f} "
                    f"{s['pass_rate']:>6.3f} "
                    f"{s['mean_time_ms'] / 1000.0:>8.3f}\n")
    print(f"exp2 合并完成, 透视表: {pivot_path}")


def merge_exp5():
    rows = []
    for p in sorted(glob.glob(os.path.join(JOBS, "exp5_*", "per_instance.csv"))):
        rows.extend(read_csv(p))
    if not rows:
        print("exp5: 无输出")
        return
    numify(rows)
    write_csv(os.path.join(RESULTS, "exp5_ablation", "per_instance.csv"), rows)

    # Per-variant mean±std (same two-level summary as exp5_ablation.py)
    proto_summary = {s["protocol"]: s for s in ec.summarize(rows)}
    per_variant_seed = {}
    for r in rows:
        proto = r["protocol"]
        if "[seed" in proto:
            var, s = proto.rsplit("[seed", 1)
            per_variant_seed.setdefault(var, {})[int(s.rstrip("]"))] = proto
        else:
            per_variant_seed.setdefault(proto, {})["single"] = proto
    summary = []
    for var, seed_protos in per_variant_seed.items():
        stats = []
        for s, proto in seed_protos.items():
            ps = proto_summary[proto]
            stats.append({"seed": s, "pass_rate": ps["pass_rate"],
                          "mean_cost_feasible": ps["mean_cost_feasible"]})
        pr = [x["pass_rate"] for x in stats]
        cf = [x["mean_cost_feasible"] for x in stats
              if x["mean_cost_feasible"] == x["mean_cost_feasible"]]
        summary.append({
            "variant": var, "n_seeds": len(stats),
            "seeds": str([x["seed"] for x in stats]),
            "pass_rate_mean": float(np.mean(pr)),
            "pass_rate_std": float(np.std(pr)) if len(pr) > 1 else 0.0,
            "cost_feas_mean": float(np.mean(cf)) if cf else float("nan"),
            "cost_feas_std": float(np.std(cf)) if len(cf) > 1 else 0.0,
        })
    write_csv(os.path.join(RESULTS, "exp5_ablation", "summary.csv"), summary)
    for s in summary:
        print(f"{s['variant']:>22} ({s['n_seeds']} runs): "
              f"pass={s['pass_rate_mean']:.3f}±{s['pass_rate_std']:.3f} | "
              f"cost(feas)={s['cost_feas_mean']:.2f}±{s['cost_feas_std']:.2f}")
    print("exp5 合并完成")


def merge_exp6():
    # Block A: scale extrapolation
    rows = []
    for p in sorted(glob.glob(os.path.join(JOBS, "exp6ext_*",
                                           "extrap_per_instance.csv"))):
        rows.extend(read_csv(p))
    if rows:
        numify(rows)
        write_csv(os.path.join(RESULTS, "exp6_general",
                               "extrap_per_instance.csv"), rows)
        write_csv(os.path.join(RESULTS, "exp6_general", "extrap_summary.csv"),
                  ec.summarize(rows))
    # Block B: distribution shift
    rows = []
    for p in sorted(glob.glob(os.path.join(JOBS, "exp6ds_*",
                                           "distshift_per_instance.csv"))):
        rows.extend(read_csv(p))
    if rows:
        numify(rows)
        write_csv(os.path.join(RESULTS, "exp6_general",
                               "distshift_per_instance.csv"), rows)
        write_csv(os.path.join(RESULTS, "exp6_general",
                               "distshift_summary.csv"), ec.summarize(rows))
    print("exp6 合并完成")


if __name__ == "__main__":
    merge_exp1()
    merge_exp2()
    merge_exp5()
    merge_exp6()
    print("\n全部合并完成")
