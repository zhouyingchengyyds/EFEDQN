import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Six conditions in the same order as test_envs in train.py
CONDITIONS = [
    ("25 nodes, 16 tw",   25,  16),
    ("50 nodes, 32 tw",   50,  32),
    ("100 nodes, 64 tw", 100,  64),
    ("100 nodes, 32 tw", 100,  32),
    ("100 nodes, 64 tw (repeat)", 100, 64),
    ("100 nodes, 128 tw", 100, 128),
]

# Alias-to-group mapping for the exported curve names:
# aliases 1..3 = 25/50/100 nodes (groups 1/2/3), aliases 4..6 = 32/64/128 windows (groups 4/3/6, 1-based)
FIGURE_MAP = {
    "fig4-1": 0, "fig4-2": 1, "fig4-3": 2,
    "fig5-1": 3, "fig5-2": 2, "fig5-3": 5,
}


def load_curve(input_dir, prefix, group_index, test_envs):
    path = os.path.join(input_dir, f"{prefix}_reward_{group_index + 1}.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    cols = [c for c in df.columns if str(c).startswith("Column_")][:test_envs]
    # Each row is one periodic evaluation (mean over environments); x is the true episode index
    return df[cols].mean(axis=1).to_numpy()


def plot_group(input_dir, out_dir, group_index, eval_every, test_envs):
    off = load_curve(input_dir, "off", group_index, test_envs)
    on = load_curve(input_dir, "On", group_index, test_envs)
    if off is None and on is None:
        print(f"跳过组 {group_index + 1}: 无数据文件")
        return None

    title = CONDITIONS[group_index][0]
    plt.figure(figsize=(10, 6))
    x = np.arange(1, (len(off) if off is not None else len(on)) + 1) * eval_every
    if off is not None:
        plt.plot(x, off, linestyle='-', linewidth=2, label='EFEDQN')
    if on is not None:
        plt.plot(x, on, linestyle='--', linewidth=2, label='EFEDQN-E')
    plt.title(f'Reward curve ({title})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    stem = os.path.join(out_dir, f"reward_curve_{group_index + 1}")
    plt.savefig(stem + ".png", dpi=200, bbox_inches='tight')
    plt.savefig(stem + ".svg", format='svg', bbox_inches='tight')
    plt.close()
    print(f"已保存: {stem}.png / .svg")
    return stem


def main():
    parser = argparse.ArgumentParser(description="从 output/*.xlsx 原始数据重绘训练曲线（无插值拉伸）")
    parser.add_argument('--input-dir', type=str, default='output')
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--eval-every', type=int, default=30, help='评估间隔（episodes）')
    parser.add_argument('--test-envs', type=int, default=50, help='每条件评估环境数（xlsx 列数）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stems = {}
    for g in range(len(CONDITIONS)):
        stems[g] = plot_group(args.input_dir, args.output_dir, g, args.eval_every, args.test_envs)

    # Copy each saved curve to its alias file name
    print("\n论文图对应关系：")
    for fig_name, g in FIGURE_MAP.items():
        src = stems.get(g)
        if src:
            for ext in (".png", ".svg"):
                dst = os.path.join(args.output_dir, fig_name + ext)
                with open(src + ext, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
            print(f"  {fig_name}  <-  组 {g + 1} ({CONDITIONS[g][0]})")


if __name__ == "__main__":
    main()
