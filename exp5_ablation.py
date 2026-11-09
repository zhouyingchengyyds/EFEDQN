# -*- coding: utf-8 -*-
"""exp5_ablation.py — component ablation and permutation-equivariance control.

Ablation scope (four families):
  Full model          — EFEDQN full model, 3 training seeds (model_seed{1,2,3}.pth)
  w/o gating          — gating disabled, test/Ablation/Gate_control/model_seed{1,2,3}.pth
  plain decoder       — decoder architecture (plain MLP vs residual MLP),
                        test/Ablation/Decoder_arch/model_seed{1,2,3}.pth
  uniform replay      — experience replay (uniform sampling vs PER),
                        test/Ablation/Replay_strategy/model_seed{1,2,3}.pth

Inference-time variations (same trained weights model_seed1.pth, no extra
model needed):
  EFEDQN-K{k}         — number of enhancements K in {1, 2, 5, 10, 20}
  EFormer-K{k}        — EFormer best-of-K sampling with the same enhancement
                        count as EFEDQN (control against permutation-equivariant
                        or symmetry-exploiting alternatives)

Condition: 100 nodes / 64 windows / 32 targets (consistent with the other
experiments), --num-instances defaults to 100.
Reporting: per-variant mean±std across seeds (pass rate / condition cost); K
rows are single-weight inference variations. Variants whose weights are missing
are skipped with a message.

Usage:
  python exp5_ablation.py
  python exp5_ablation.py --variants full,K   # only the full model and K ablation
"""

import argparse
import os

import numpy as np

import exp_common as ec

ABLATION_DIR = os.path.join(ec.TEST_DIR, "Ablation")

# (variant name, weight directory, weight file template)
SEED_VARIANTS = [
    ("Full model", ec.EFEDQN_DIR, "model_seed{s}.pth"),
    ("w/o gating", os.path.join(ABLATION_DIR, "Gate_control"),
     "model_seed{s}.pth"),
    ("Plain MLP decoder", os.path.join(ABLATION_DIR, "Decoder_arch"),
     "model_seed{s}.pth"),
    ("Uniform replay", os.path.join(ABLATION_DIR, "Replay_strategy"),
     "model_seed{s}.pth"),
]
SEEDS = [1, 2, 3]
K_GRID = [1, 2, 5, 10, 20]


def summarize_variants(rows):
    """Two-level summary: per-variant mean±std across seeds; K rows hold a
    single value. Kept as a function so splice_results.py can recompute the
    summary after splicing per_instance."""
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
    return summary


def main():
    parser = argparse.ArgumentParser(description="表11 消融实验(C11+C12)")
    parser.add_argument('--num-instances', type=int, default=100)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--k-v', type=int, default=64)
    parser.add_argument('--targets', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument('--variants', type=str, default="all",
                        help="all 或逗号子集: full,gating,decoder,replay,K,eformer")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp5_ablation'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    want = None if args.variants == "all" else \
        {v.strip() for v in args.variants.split(",")}

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build the instance pool (fixed seeds, materialized) ----
    inst_dir = os.path.join(args.output_dir, "instances")
    instances = []
    for i in range(args.num_instances):
        seed = ec.derive_seed("exp5", i)
        inst = ec.gen_random_instance(args.n, args.k_v, args.targets, seed)
        inst["id"] = f"abl_i{i:03d}"
        instances.append(inst)
    ec.save_instances(instances, inst_dir)
    cond = f"{args.n}n_{args.k_v}tw"

    # ---- Build the method table ----
    methods = {}
    tag_map = {"Full model": "full", "w/o gating": "gating",
               "Plain MLP decoder": "decoder", "Uniform replay": "replay"}
    for (name, folder, tmpl) in SEED_VARIANTS:
        tag = tag_map[name]
        if want is not None and tag not in want:
            continue
        for s in SEEDS:
            path = os.path.join(folder, tmpl.format(s=s))
            if not os.path.isfile(path):
                print(f"[跳过] {name} seed{s}: 缺权重 {path} (补齐后重跑)")
                continue
            methods[f"{name}[seed{s}]"] = ec.EFEDQNSolver(
                path, device, k=1, name=f"{name}[seed{s}]",
                extra_sys_path=folder if folder != ec.EFEDQN_DIR else None)

    if want is None or "K" in want:
        base = os.path.join(ec.EFEDQN_DIR, "model_seed1.pth")
        for k in K_GRID:
            methods[f"EFEDQN-K{k}"] = ec.EFEDQNSolver(
                base, device, k=k, name=f"EFEDQN-K{k}")

    if want is None or "eformer" in want:
        ckpt = os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt")
        for k in K_GRID:
            methods[f"EFormer-K{k}"] = ec.EFormerSolver(
                ckpt, device, k=k, name=f"EFormer-K{k}",
                max_steps=args.max_steps)

    if not methods:
        raise SystemExit("未构造出任何方法(权重均缺失?)")
    print(f"方法: {sorted(methods)} | 设备: {device}")

    rows = ec.run_methods_on_instances(
        methods, instances, cond, args.max_steps)
    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), rows)

    # ---- Two-level summary: per-variant mean±std across seeds; K single ----
    summary = summarize_variants(rows)
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"), summary)
    print("\n=== 消融汇总(mean±std, 跨种子) ===")
    for s in summary:
        print(f"{s['variant']:>20} ({s['n_seeds']} runs): "
              f"pass={s['pass_rate_mean']:.3f}±{s['pass_rate_std']:.3f} | "
              f"cost(feas)={s['cost_feas_mean']:.2f}±{s['cost_feas_std']:.2f}")


if __name__ == "__main__":
    main()
