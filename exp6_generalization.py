# -*- coding: utf-8 -*-
"""exp6_generalization.py — generalization experiments.

Generalization is validated in two parts:

Block A — scale extrapolation (larger scales drawn from the same generator):
  150/200/300 nodes (windows/targets follow the standard condition table),
  7 methods (TD-ACO/GA/PSO/EFormer/EFormer-K10/EFEDQN/EFEDQN-E).

Block B — distribution shift (4 dimensions differing from the training
distribution, 100 nodes):
  (v)   target ratio: 0.1 / 0.5 (training uses 1/3)
  (vi)  restriction density: 16 / 192 windows (reference is 64)
  (vii) window duration: short 15-60 min / long 13-24 h (training uses 1-13 h)
  (iii) average degree: sparse connected out-deg≈2 (training mean out-degree≈3.2)
  Methods: EFEDQN / EFEDQN-E / EFormer.
  A no-shift reference row is included. Real road network validation is done
  in exp7.

Usage:
  python exp6_generalization.py                      # both blocks
  python exp6_generalization.py --blocks extrapolate # scale extrapolation only
  python exp6_generalization.py --blocks distshift --num-instances 20
"""

import argparse
import os

import exp_common as ec

# Distribution-shift conditions: (name, n, k_v, targets, window_mode, out_deg)
DISTSHIFT = [
    ("no-shift(ref)",  100, 64, 32, "default", None),
    ("target-ratio-0.1", 100, 64, 10, "default", None),
    ("target-ratio-0.5", 100, 64, 50, "default", None),
    ("restr-dense-16",  100, 16, 32, "default", None),
    ("restr-dense-192", 100, 192, 32, "default", None),
    ("win-short",       100, 64, 32, "short", None),
    ("win-long",        100, 64, 32, "long", None),
    ("degree-thin",     100, 64, 32, "default", 2.0),
]


def gen_block_instances(name_list, num_instances, out_dir):
    """Generate and materialize instances from a condition table; name_list is
    either the DISTSHIFT or the SCALE_EXTRAP tuples."""
    all_instances = {}
    for spec in name_list:
        if len(spec) == 4:  # SCALE_EXTRAP: (name, n, k_v, targets)
            name, n, k_v, t_num = spec
            wmode, odeg = "default", None
        else:               # DISTSHIFT: 6-tuple
            name, n, k_v, t_num, wmode, odeg = spec
        insts = []
        for i in range(num_instances):
            seed = ec.derive_seed("exp6", name, i)
            inst = ec.gen_random_instance(n, k_v, t_num, seed,
                                          window_mode=wmode, out_deg=odeg)
            inst["id"] = f"{name}_i{i:03d}"
            insts.append(inst)
        ec.save_instances(insts, os.path.join(out_dir, name))
        all_instances[name] = insts
    return all_instances


def main():
    parser = argparse.ArgumentParser(description="表12/13 泛化实验(C10)")
    parser.add_argument('--num-instances', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument('--blocks', type=str, default="extrapolate,distshift")
    parser.add_argument('--methods', type=str, default=None,
                        help='可选: 逗号分隔方法子集(默认各块全集)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp6_general'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    blocks = {b.strip() for b in args.blocks.split(",")}

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    inst_dir = os.path.join(args.output_dir, "instances")

    # Block A: scale extrapolation
    if "extrapolate" in blocks:
        methods = {}
        if args.methods:
            want = {m.strip() for m in args.methods.split(",")}
        else:
            # Default method set; EFormer-K10 is not enabled by default and can
            # be switched on with --methods EFormer-K10
            want = {"TD-ACO", "GA", "PSO", "EFormer", "EFEDQN"}
        if "TD-ACO" in want:
            methods["TD-ACO"] = ec.HeuristicSolver("aco")
        if "GA" in want:
            methods["GA"] = ec.HeuristicSolver("ga")
        if "PSO" in want:
            methods["PSO"] = ec.HeuristicSolver("pso")
        if "EFormer" in want:
            methods["EFormer"] = ec.EFormerSolver(
                os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, 1,
                max_steps=args.max_steps)
        if "EFormer-K10" in want:
            methods["EFormer-K10"] = ec.EFormerSolver(
                os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device,
                10, name="EFormer-K10", max_steps=args.max_steps)
        if "EFEDQN" in want:
            s = ec.make_efedqn_solvers(device=device)
            methods["EFEDQN"] = s["EFEDQN"]
            methods["EFEDQN-E"] = s["EFEDQN-E"]
        if not methods:
            raise SystemExit("块 A 未选择方法")

        print(f"\n########## 块 A: 规模外推 ({list(methods)}) ##########")
        pool = gen_block_instances(ec.SCALE_EXTRAP, args.num_instances,
                                   inst_dir)
        rows = []
        for name, insts in pool.items():
            print(f"\n=== [{name}] ===")
            r = ec.run_methods_on_instances(methods, insts, name,
                                            args.max_steps)
            rows.extend(r)
            ec.print_summary(ec.summarize(r))
        ec.write_csv(os.path.join(args.output_dir, "extrap_per_instance.csv"),
                     rows)
        ec.write_csv(os.path.join(args.output_dir, "extrap_summary.csv"),
                     ec.summarize(rows))

    # Block B: distribution shift
    if "distshift" in blocks:
        methods = {}
        if args.methods:
            want = {m.strip() for m in args.methods.split(",")}
        else:
            want = {"EFEDQN", "EFormer"}
        s = ec.make_efedqn_solvers(device=device)
        if "EFEDQN" in want:
            methods["EFEDQN"] = s["EFEDQN"]
            methods["EFEDQN-E"] = s["EFEDQN-E"]
        if "EFormer" in want:
            methods["EFormer"] = ec.EFormerSolver(
                os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, 1,
                max_steps=args.max_steps)
        if not methods:
            raise SystemExit("块 B 未选择方法")

        print(f"\n########## 块 B: 分布偏移 ({list(methods)}) ##########")
        pool = gen_block_instances(DISTSHIFT, args.num_instances, inst_dir)
        rows = []
        for spec in DISTSHIFT:
            name = spec[0]
            insts = pool[name]
            print(f"\n=== [{name}] ===")
            r = ec.run_methods_on_instances(methods, insts, name,
                                            args.max_steps)
            rows.extend(r)
            ec.print_summary(ec.summarize(r))
        ec.write_csv(os.path.join(args.output_dir, "distshift_per_instance.csv"),
                     rows)
        ec.write_csv(os.path.join(args.output_dir, "distshift_summary.csv"),
                     ec.summarize(rows))


if __name__ == "__main__":
    main()
