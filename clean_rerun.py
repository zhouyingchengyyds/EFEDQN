# -*- coding: utf-8 -*-
"""clean_rerun.py — single-process timing re-measurement without parallel contention.

Background: when run_all_experiments.py runs jobs in parallel, the GPU methods
(EFEDQN/EFormer) are slowed down by roughly 1.5-2x because several processes
share the same device, whereas the CPU heuristics are almost unaffected, which
makes per-method timings incomparable. This script re-measures serially after
all parallel jobs have finished:
1. exp1 timing: 5 conditions x 7 methods x first --n1 (default 10) instances of
   each pool (loaded from the cached instance pools); only time_ms is recorded,
   the other metrics come from the parallel run and are unaffected by contention;
2. exp6ext timing: 3 scales x 6 methods x first --n2 (default 5) instances of
   each pool;
3. exp7 full rerun: real road network, 4 departure times x 7 methods (both
   results and timings are used);
4. exp4 latency: 30 fresh seeds on 100-node instances, replaying the
   "plan -> perturb at 50% -> continue" scenario to measure the single-forward
   latency of EFEDQN and the full re-solve latency of RollingHorizon cleanly;
5. exp2 full rerun: certified pool of 200 instances x (K=1/K=10/wall) for all
   methods (both results and timings are used, so the tables stay internally
   consistent).

Writes CSVs under results/clean_timing/; used to replace the timing columns and
full tables when filling in results.
Usage:  python clean_rerun.py [--skip-exp2]
"""

import argparse
import copy
import json
import os
import time

import numpy as np

import exp_common as ec

OUT = os.path.join(ec.RESULTS_DIR, "clean_timing")
JOBS = os.path.join(ec.RESULTS_DIR, "_jobs")


def write_rows(name, rows, append=False):
    import csv
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    if rows:
        mode = "a" if append else "w"
        write_header = not (append and os.path.exists(p))
        with open(p, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
    print(f"写出: {p} ({len(rows)} 行, {'追加' if append else '覆盖'})", flush=True)


def load_pool(job_dir, cond, n):
    import glob
    insts = []
    for p in sorted(glob.glob(os.path.join(JOBS, job_dir, "instances",
                                           cond, "*.json")))[:n]:
        with open(p, encoding="utf-8") as f:
            insts.append(json.load(f))
    if not insts:
        raise RuntimeError(f"实例池为空: {job_dir}/{cond}")
    return insts


def time_exp1(device, n1, methods=None, append=False):
    import torch
    all_methods = {}
    all_methods["TD-ACO"] = ec.HeuristicSolver("aco", restarts=1)
    all_methods["GA"] = ec.HeuristicSolver("ga", restarts=1)
    all_methods["PSO"] = ec.HeuristicSolver("pso", restarts=1)
    all_methods["EFormer"] = ec.EFormerSolver(
        os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=1,
        max_steps=200)
    all_methods["EFormer-K10"] = ec.EFormerSolver(
        os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=10,
        name="EFormer-K10", max_steps=200)
    s = ec.make_efedqn_solvers(device=device)
    all_methods["EFEDQN"] = s["EFEDQN"]
    all_methods["EFEDQN-E"] = s["EFEDQN-E"]
    methods = {k: v for k, v in all_methods.items()
               if methods is None or k in methods}

    conds = ["25n_16tw", "50n_32tw", "100n_64tw", "100n_32tw", "100n_128tw"]
    # Pre-warm the process pool before timing EFEDQN-E (keeps worker weight
    # loading out of the timed window)
    if "EFEDQN-E" in methods:
        ec.prewarm_efedqn_pool(
            os.path.join(ec.EFEDQN_DIR, "model_seed1.pth"), None,
            device, k=10)
    rows = []
    for cond in conds:
        insts = load_pool(f"exp1_{cond}", cond, n1)
        print(f"[exp1-timing] {cond}: {len(insts)} 实例", flush=True)
        for name, solver in methods.items():
            for inst in insts:
                env = ec.env_from_inst(inst)
                seed = ec.derive_seed(cond, inst["id"], name)
                t0 = time.perf_counter()
                solver.solve(env, seed=seed, max_steps=200)
                rows.append({"condition": cond, "method": name,
                             "instance": inst["id"],
                             "time_ms": (time.perf_counter() - t0) * 1000.0})
            ts = [r["time_ms"] for r in rows
                  if r["condition"] == cond and r["method"] == name]
            print(f"  {name:<12} mean={np.mean(ts)/1000:.2f}s", flush=True)
    write_rows("exp1_timing.csv", rows, append=append)


def time_exp6ext(device, n2, methods=None, append=False):
    import torch
    all_methods = {}
    all_methods["TD-ACO"] = ec.HeuristicSolver("aco", restarts=1)
    all_methods["GA"] = ec.HeuristicSolver("ga", restarts=1)
    all_methods["PSO"] = ec.HeuristicSolver("pso", restarts=1)
    all_methods["EFormer"] = ec.EFormerSolver(
        os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, k=1,
        max_steps=200)
    s = ec.make_efedqn_solvers(device=device)
    all_methods["EFEDQN"] = s["EFEDQN"]
    all_methods["EFEDQN-E"] = s["EFEDQN-E"]
    methods = {k: v for k, v in all_methods.items()
               if methods is None or k in methods}

    conds = ["150n_96tw", "200n_128tw", "300n_192tw"]
    # Pre-warm the process pool before timing EFEDQN-E (same as time_exp1)
    if "EFEDQN-E" in methods:
        ec.prewarm_efedqn_pool(
            os.path.join(ec.EFEDQN_DIR, "model_seed1.pth"), None,
            device, k=10)
    rows = []
    for cond in conds:
        insts = load_pool("exp6ext_TD-ACO", cond, n2)
        print(f"[exp6ext-timing] {cond}: {len(insts)} 实例", flush=True)
        for name, solver in methods.items():
            for inst in insts:
                env = ec.env_from_inst(inst)
                seed = ec.derive_seed(cond, inst["id"], name)
                t0 = time.perf_counter()
                solver.solve(env, seed=seed, max_steps=200)
                rows.append({"condition": cond, "method": name,
                             "instance": inst["id"],
                             "time_ms": (time.perf_counter() - t0) * 1000.0})
            ts = [r["time_ms"] for r in rows
                  if r["condition"] == cond and r["method"] == name]
            print(f"  {name:<12} mean={np.mean(ts)/1000:.2f}s", flush=True)
    write_rows("exp6ext_timing.csv", rows, append=append)


def rerun_exp7(device):
    import subprocess, sys
    print("[exp7] 完整重跑(单进程)", flush=True)
    r = subprocess.run([sys.executable, "exp7_real_scene.py",
                        "--output-dir", os.path.join(OUT, "exp7_real")],
                       cwd=ec.TEST_DIR)
    if r.returncode != 0:
        print("[exp7] 重跑失败", flush=True)


def time_exp4_latency(device, n_scen=30):
    """Measure replanning latency cleanly.

    Uses fresh seeds on 100-node instances: plan with greedy to the state at
    25%/50%/75% of the steps, then measure (a) EFEDQN single-forward latency
    (re-encode + decode), 3 repeats, median; (a') frozen-encoder single-forward
    latency, 3 repeats, median; (b) full RH re-solve latency. All three
    perturbation points are covered.
    """
    import torch
    import exp4_disruption as e4
    s = ec.make_efedqn_solvers(device=device)
    full = s["EFEDQN"]
    frozen = s["EFEDQN-Frozen"]
    rh = ec.RollingHorizonAdapter(max_steps=200, two_opt=False)
    rows = []
    for i in range(n_scen):
        seed = ec.derive_seed("clean_lat", i)
        inst = ec.gen_random_instance(100, 64, 32, seed)
        inst["id"] = f"lat_i{i:03d}"
        env = ec.env_from_inst(inst)
        plan = full.solve(env, seed=ec.derive_seed(seed, "plan"), max_steps=200)
        if not plan["feasible"] or not plan.get("actions"):
            continue
        actions = plan["actions"]
        for p in (0.25, 0.5, 0.75):
            step_idx = max(1, int(len(actions) * p))
            pos, t, goals, _ = e4.plan_prefix_state(copy.deepcopy(env),
                                                    actions, step_idx, 200)
            init = {"pos": pos, "time": t, "goals": goals, "start_pos": 0}
            row = {"instance": inst["id"], "point": p, "step_idx": step_idx}
            # (a) EFEDQN single forward pass, 3 repeats
            lats = [e4.single_forward_latency_ms(full, env, init)
                    for _ in range(3)]
            row["efedqn_fwd_ms_med"] = float(np.median(lats))
            row["efedqn_fwd_ms_min"] = min(lats)
            # (a') frozen-encoder single forward pass, 3 repeats
            if frozen is not None:
                lats_f = [e4.single_forward_latency_ms(frozen, env, init)
                          for _ in range(3)]
                row["frozen_fwd_ms_med"] = float(np.median(lats_f))
                row["frozen_fwd_ms_min"] = min(lats_f)
            # (b) full RH re-solve
            t0 = time.perf_counter()
            rh.solve(copy.deepcopy(env), seed=ec.derive_seed(seed, "rh", p),
                     max_steps=200 - step_idx, init=init)
            row["rh_resolve_ms"] = (time.perf_counter() - t0) * 1000.0
            rows.append(row)
            print(f"[latency] {inst['id']} p={p} "
                  f"fwd={row['efedqn_fwd_ms_med']:.1f}ms "
                  f"frozen={row.get('frozen_fwd_ms_med', float('nan')):.1f}ms "
                  f"rh={row['rh_resolve_ms']/1000:.2f}s", flush=True)
    write_rows("exp4_latency.csv", rows)
    if rows:
        print("EFEDQN 前向中位: "
              f"{np.median([r['efedqn_fwd_ms_med'] for r in rows]):.1f}ms; "
              "RH 重解中位: "
              f"{np.median([r['rh_resolve_ms'] for r in rows])/1000:.2f}s",
              flush=True)


def rerun_exp2(device):
    import subprocess, sys
    print("[exp2] 完整重跑(单进程, 全预算档)", flush=True)
    r = subprocess.run([sys.executable, "exp2_quality_fair.py",
                        "--output-dir", os.path.join(OUT, "exp2_quality")],
                       cwd=ec.TEST_DIR)
    if r.returncode != 0:
        print("[exp2] 重跑失败", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n1', type=int, default=10, help='exp1 每条件计时实例数')
    parser.add_argument('--n2', type=int, default=5, help='exp6ext 每规模计时实例数')
    parser.add_argument('--parts', type=str, default="exp1,exp6ext,exp7,exp4lat,exp2")
    parser.add_argument('--methods', type=str, default=None,
                        help='可选: 逗号分隔方法子集(计时部分)')
    parser.add_argument('--append', action='store_true', help='计时CSV追加而非覆盖')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    parts = {p.strip() for p in args.parts.split(",")}
    methods = None
    if args.methods:
        methods = {m.strip() for m in args.methods.split(",")}

    if "exp1" in parts:
        time_exp1(device, args.n1, methods, append=args.append)
    if "exp6ext" in parts:
        time_exp6ext(device, args.n2, methods, append=args.append)
    if "exp7" in parts:
        rerun_exp7(device)
    if "exp4lat" in parts:
        time_exp4_latency(device)
    if "exp2" in parts:
        rerun_exp2(device)
    print("\nclean_rerun 完成", flush=True)


if __name__ == "__main__":
    main()
