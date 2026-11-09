# -*- coding: utf-8 -*-
"""exp3_temporal_diag.py — temporal-reasoning diagnosis on the certified pool.

Diagnosis subset: the benchmark certified pool (n10x50 / n20x50 / n30x50 /
n50x50, all proven OPTIMAL by Gurobi). Pool generation (instances.py) already
enforces certificate (i): the optimal route without any time window is
infeasible under the window constraints, and the windowed optimum is strictly
more expensive than the free optimum. The static optimum of every instance is
thus proven infeasible, so this subset serves as a temporal diagnosis set and
needs no separate construction.

Compared variants (all on the same instance pool, paired comparison):
  Full (EFEDQN)      — full model, greedy
  Full (EFEDQN-E)    — full model, best-of-10 over 10 permutations
  (a) No-TW          — static encoder without time-window features
                       (model_seed1_no_tw.pth, window features set to -2),
                       no dynamic mask (purely static model)
  (c) Mask-only      — same weights as (a), with a dynamic action mask applied
                       at inference (only the mask carries window information)
  (b) Frozen         — frozen encoder (model_seed1_no_dy.pth: encoded once and
                       not re-encoded per step; the decoder input takes a
                       4-dim elapsed-time feature; a dynamic action mask is
                       applied at inference)

Metrics: pass rate, condition cost, gap% (relative to the certified
time-aware optimum). The gap is measured on feasible routes; when a static
variant is infeasible, the pass rate column carries that information.

Usage:
  python exp3_temporal_diag.py            # all variants
  python exp3_temporal_diag.py --scales 10,20
"""

import argparse
import os

import exp_common as ec


def build_variants(device, max_steps):
    solvers = {}
    s = ec.make_efedqn_solvers(device=device)
    solvers["Full(EFEDQN)"] = s["EFEDQN"]
    solvers["Full(EFEDQN-E)"] = s["EFEDQN-E"]
    # (a) no-TW without mask: purely static model
    solvers["(a)No-TW"] = (ec.EFEDQNSolver(
        os.path.join(ec.TEST_DIR, "No_timwin", "model_seed1_no_tw.pth"),
        device, k=1, use_mask=False, no_tw=True, name="(a)No-TW")
        if s["EFEDQN-noTW"] is not None else None)
    # (c) mask-only: no-TW features + dynamic mask
    solvers["(c)Mask-only"] = (ec.EFEDQNSolver(
        os.path.join(ec.TEST_DIR, "No_timwin", "model_seed1_no_tw.pth"),
        device, k=1, use_mask=True, no_tw=True, name="(c)Mask-only")
        if s["EFEDQN-noTW"] is not None else None)
    # (b) frozen encoding
    solvers["(b)Frozen"] = s["EFEDQN-Frozen"]
    return {k: v for k, v in solvers.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(
        description="表9 时间诊断: 认证池(证书(i)子集)上的时间推理能力")
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument('--scales', type=str, default="10,20,30,50")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp3_diag'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pool = ec.load_benchmark_pool()
    scales = {int(s) for s in args.scales.split(",")}
    pool = [(i, r) for (i, r) in pool if i["nodes"] in scales]
    if not pool:
        raise SystemExit(f"认证池为空: {ec.BENCHMARK_DIR}")

    methods = build_variants(device, args.max_steps)
    # Report certificate (i) statistics (already enforced at pool generation)
    n_cert = sum(1 for (i, r) in pool if r and r["status"] == "OPTIMAL")
    print(f"诊断子集: {len(pool)} 实例(全部满足证书(i): 静态最优在有窗下不可行; "
          f"{n_cert} 个含认证最优) | 设备: {device}")

    by_scale = {}
    for inst, ref in pool:
        by_scale.setdefault(inst["nodes"], []).append((inst, ref))

    all_rows = []
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
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"),
                 ec.summarize(all_rows))

    # Persist certificate (i) statistics
    cert_path = os.path.join(args.output_dir, "certificate_summary.txt")
    with open(cert_path, "w", encoding="utf-8") as f:
        f.write("certificate (i): static optimum infeasible under windows\n")
        f.write(f"pool_size={len(pool)}\n")
        f.write("note: enforced at pool generation (instances.py), "
                "window_gap = tw_opt - free_opt > 0 certified per instance\n")
    print(f"证书统计: {cert_path}")


if __name__ == "__main__":
    main()
