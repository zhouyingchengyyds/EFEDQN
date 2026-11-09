# -*- coding: utf-8 -*-
"""splice_results.py — splice experiment results row by row.

Use case: when only the affected methods are rerun, the new per_instance.csv rows
replace the matching ones in the existing result directory by
(condition, instance_id, protocol), and summary.csv is recomputed -- rows of the
methods that were not rerun are kept unchanged.

Usage:
  # regular experiments (exp1/exp2/exp7, summary = ec.summarize):
  python splice_results.py --dest results/exp1_main --src <new run dir>/per_instance.csv

  # splice only the rows of the given protocols (default: all protocols in src):
  python splice_results.py --dest results/exp2_quality --src ... \
      --protocols "EFormer[K=10],EFEDQN-E"

  # exp5 (summary = per-variant aggregation summarize_variants):
  python splice_results.py --dest results/exp5_ablation --src ... --exp5

  # exp2 also regenerates the pivot table table_quality_fair.txt:
  python splice_results.py --dest results/exp2_quality --src ... --exp2
"""

import argparse
import csv
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import exp_common as ec  # noqa: E402


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(r):
    return (r.get("condition", ""), r.get("instance_id", ""), r["protocol"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dest", required=True,
                   help="目标结果目录(含 per_instance.csv / summary.csv)")
    p.add_argument("--src", required=True, help="新跑的 per_instance.csv")
    p.add_argument("--protocols", type=str, default=None,
                   help="逗号分隔: 只拼接这些 protocol 的行(默认 src 全部)")
    p.add_argument("--exp5", action="store_true",
                   help="summary 用 exp5 的变体级聚合(summarize_variants)")
    p.add_argument("--exp2", action="store_true",
                   help="附带重生成 exp2 的 table_quality_fair.txt 透视")
    args = p.parse_args()

    dest_per = os.path.join(args.dest, "per_instance.csv")
    old_rows = read_rows(dest_per)
    new_rows = read_rows(args.src)

    if args.protocols:
        keep = {s.strip() for s in args.protocols.split(",") if s.strip()}
        new_rows = [r for r in new_rows if r["protocol"] in keep]
    if not new_rows:
        raise SystemExit("src 中没有可拼接的行(protocols 过滤后为空)")

    new_keys = {row_key(r) for r in new_rows}
    replaced = sum(1 for r in old_rows if row_key(r) in new_keys)
    out_rows = [r for r in old_rows if row_key(r) not in new_keys] + new_rows

    # Convert numeric columns back to native types for summarize (DictReader
    # returns strings only; int columns in old CSV files may carry a .0 suffix,
    # so convert to float first and then to int)
    def _typed(r):
        for k in ("cost", "time_ms", "reward", "gap_pct"):
            if r.get(k) not in (None, ""):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = float("nan")
        for k in ("feasible", "completed", "steps", "n_targets",
                  "invalid", "window_viol"):
            if r.get(k) not in (None, ""):
                try:
                    r[k] = int(float(r[k]))
                except (ValueError, TypeError):
                    pass
        return r

    out_rows = [_typed(r) for r in out_rows]

    ec.write_csv(dest_per, out_rows)
    print(f"拼接: 替换 {replaced} 行, 追加 {len(new_rows) - max(replaced, 0)} 行, "
          f"共 {len(out_rows)} 行 -> {dest_per}")

    if args.exp5:
        from exp5_ablation import summarize_variants
        summary = summarize_variants(out_rows)
    else:
        summary = ec.summarize(out_rows)
    ec.write_csv(os.path.join(args.dest, "summary.csv"), summary)

    if args.exp2:
        pivot = os.path.join(args.dest, "table_quality_fair.txt")
        with open(pivot, "w", encoding="utf-8") as f:
            f.write(f"{'scale':>5} {'method':>22} {'gap%':>8} {'cost(feas)':>11} "
                    f"{'pass':>6} {'time(s)':>8}\n")
            for s in summary:
                f.write(f"{s['condition']:>5} {s['protocol']:>22} "
                        f"{s['mean_gap_pct']:>8.2f} "
                        f"{s['mean_cost_feasible']:>11.2f} "
                        f"{s['pass_rate']:>6.3f} "
                        f"{s['mean_time_ms'] / 1000.0:>8.3f}\n")
        print(f"透视表: {pivot}")


if __name__ == "__main__":
    main()
