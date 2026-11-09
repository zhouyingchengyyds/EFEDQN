# -*- coding: utf-8 -*-
"""exp0_training_curves.py — periodic evaluation curves recorded during training.

Data sources (either one, xlsx preferred):
1. EFEDQN/output_seed1/{off,On}_reward_{1..6}.xlsx — periodic evaluation data
   exported by train.py at the end of training (one file per test condition
   group, rows = one evaluation every 100 episodes, columns = the 15 fixed test
   environments; off = greedy, i.e. EFEDQN; On = best-of-10 over 10
   permutations, i.e. EFEDQN-E);
2. EFEDQN/logs/train_seed1_*.log — lines of the form
   `eval@<episode> mean_off_reward per group: g1 .. g6` /
   `eval@<episode> mean_on_reward per group: g1 .. g6` in the training log
   (line-by-line equivalent to the xlsx: mean reward over the 15 test
   environments of that condition group).

Outputs (results/exp0_curves/):
- fig4-1/2/3.{pdf,png,svg} — 25 / 50 / 100 nodes (groups 1/2/3);
- fig5-1/2/3.{pdf,png,svg} — 32 / 64 / 128 time windows (groups 4/2/6; the
   64-window figure reuses group 3);
- curve_data.csv — one row per evaluation point (figure, condition, episode,
   EFEDQN, EFEDQN-E).

The x-axis is the actual training episode (0-3000, evaluated every 100
episodes, 30 evaluation points in total); EFEDQN and EFEDQN-E share the same
training run and differ only in the inference protocol used at evaluation.

Usage:
  python exp0_training_curves.py
  python exp0_training_curves.py --xlsx-dir EFEDQN/output_seed1
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
EFEDQN_DIR = os.path.join(TEST_DIR, "EFEDQN")

# 6 condition groups, ordered as test_envs in train.py (0-based)
CONDITIONS = [
    "25 nodes, 16 tw",
    "50 nodes, 32 tw",
    "100 nodes, 64 tw",
    "100 nodes, 32 tw",
    "100 nodes, 64 tw (repeat)",
    "100 nodes, 128 tw",
]

# Figure mapping (0-based group index):
# fig4-1..3 = 25/50/100 nodes; fig5-1..3 = 32/64/128 time windows
FIGURE_MAP = [
    ("fig4-1", 0, "25 nodes"),
    ("fig4-2", 1, "50 nodes"),
    ("fig4-3", 2, "100 nodes"),
    ("fig5-1", 3, "32 time windows"),
    ("fig5-2", 2, "64 time windows"),
    ("fig5-3", 5, "128 time windows"),
]

EVAL_RE = re.compile(
    r"eval@(\d+) mean_(off|on)_reward per group:\s*(.*)")


def load_from_xlsx(xlsx_dir):
    """xlsx -> {group: {'off': arr, 'on': arr}} (each row is the environment
    mean of one evaluation)."""
    out = {}
    for g in range(6):
        rec = {}
        for prefix, key in (("off", "off"), ("On", "on")):
            p = os.path.join(xlsx_dir, f"{prefix}_reward_{g + 1}.xlsx")
            if not os.path.exists(p):
                continue
            df = pd.read_excel(p)
            cols = [c for c in df.columns if str(c).startswith("Column_")]
            rec[key] = df[cols].mean(axis=1).to_numpy()
        if rec:
            out[g] = rec
    return out


def load_from_log(log_path):
    """eval@ lines of the training log -> {group: {'off': [(ep, val)..], 'on': [...]}}."""
    rows = {"off": {}, "on": {}}
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if not m:
                continue
            ep, kind, vals = int(m.group(1)), m.group(2), m.group(3)
            try:
                per_group = [float(x) for x in vals.split()]
            except ValueError:
                continue
            if len(per_group) < 6:
                continue
            rows[kind][ep] = per_group
    eps = sorted(rows["off"].keys())
    if not eps:
        return {}
    out = {}
    for g in range(6):
        rec = {}
        for key in ("off", "on"):
            if rows[key]:
                rec[key] = np.array([rows[key][ep][g] for ep in eps])
        if rec:
            out[g] = rec
    return out, eps


def main():
    parser = argparse.ArgumentParser(description="图4/图5 训练评估曲线(C1)")
    parser.add_argument('--xlsx-dir', type=str,
                        default=os.path.join(EFEDQN_DIR, "output_seed1"))
    parser.add_argument('--log', type=str, default=None,
                        help='训练日志(默认自动查找 EFEDQN/logs/train_seed1_*.log)')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(TEST_DIR, "results", "exp0_curves"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.xlsx_dir) and \
            os.path.exists(os.path.join(args.xlsx_dir, "off_reward_1.xlsx")):
        curves = load_from_xlsx(args.xlsx_dir)
        episodes = np.arange(1, len(next(iter(curves.values()))["off"]) + 1) \
            * args.eval_every
        print(f"数据来源: {args.xlsx_dir} (xlsx)")
    else:
        log = args.log
        if log is None:
            cands = sorted(glob.glob(os.path.join(
                EFEDQN_DIR, "logs", "train_seed1_*.log")))
            if not cands:
                raise SystemExit("未找到 xlsx 数据与训练日志")
            log = cands[-1]
        curves, eps = load_from_log(log)
        episodes = np.array(eps, dtype=float)
        print(f"数据来源: {log} (eval@ 日志行, {len(eps)} 个评估点)")

    csv_rows = []
    for fig_name, g, label in FIGURE_MAP:
        if g not in curves:
            print(f"[跳过] {fig_name}: 组 {g + 1} 无数据")
            continue
        rec = curves[g]
        plt.figure(figsize=(5, 3.4))
        if "off" in rec:
            plt.plot(episodes[:len(rec["off"])], rec["off"], linestyle='-',
                     linewidth=1.8, color='#1f77b4', label='EFEDQN (greedy)')
        if "on" in rec:
            plt.plot(episodes[:len(rec["on"])], rec["on"], linestyle='--',
                     linewidth=1.8, color='#d62728',
                     label='EFEDQN-E (best of 10)')
        plt.title(f"Evaluation reward ({label})")
        plt.xlabel('Training episode')
        plt.ylabel('Average evaluation reward')
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        for ext in ("pdf", "png", "svg"):
            plt.savefig(os.path.join(args.output_dir, f"{fig_name}.{ext}"),
                        dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[保存] {fig_name}.pdf/.png/.svg  ({CONDITIONS[g]})")
        for i, ep in enumerate(episodes):
            csv_rows.append({
                "figure": fig_name, "condition": CONDITIONS[g],
                "episode": int(ep),
                "reward_efedqn": rec["off"][i] if "off" in rec else "",
                "reward_efedqn_e": rec["on"][i] if "on" in rec else "",
            })

    csv_path = os.path.join(args.output_dir, "curve_data.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"写出: {csv_path} ({len(csv_rows)} 行)")


if __name__ == "__main__":
    main()
