# -*- coding: utf-8 -*-
"""run_all_experiments.py — parallel orchestrator that runs all experiments (frozen-weights runs excepted).

Concurrency control:
- global concurrency MAX_TOTAL (default 10);
- this machine is CPU-only (16 cores / 22 threads), so MAX_GPU is only a
  placeholder (= MAX_TOTAL);
- each subprocess sets OMP/MKL_NUM_THREADS=2 to avoid thread contention.

Each parallel job uses its own --output-dir (results/_jobs/<name>); once a job
finishes, merge_results.py merges it into the standard location
results/<experiment>/ (instance pools are seed-derived and identical across jobs,
so merging equals a single sequential run).

Note: exp2 is kept out of the parallel schedule -- its wall-clock restart counts
depend on real time and parallel contention would change the results themselves;
clean_rerun.py reruns it in a single process after all parallel jobs finish, and
its output (results/clean_timing/exp2_quality) is the final data.

Usage:  python run_all_experiments.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS = os.path.join(TEST_DIR, "results")

MAX_TOTAL = 10
MAX_GPU = 10

# (job name, argument list, uses GPU)
JOBS = [
    # ---- exp1: main comparison (5 conditions x 7 methods, 50 instances/condition) ----
    *[(f"exp1_{c}",
       ["exp1_main_comparison.py", "--conditions", c, "--num-instances", "50",
        "--output-dir", os.path.join(RESULTS, "_jobs", f"exp1_{c}")], True)
      for c in ["25n_16tw", "50n_32tw", "100n_64tw", "100n_32tw",
                "100n_128tw"]],
    # ---- exp2: not parallel -- see the module docstring; clean_rerun finalizes
    # ---- it in a single process ----
    # ---- exp3: temporal diagnosis (certified pool, 200 instances x 5 variants,
    # ---- including Frozen) ----
    ("exp3", ["exp3_temporal_diag.py"], True),
    # ---- exp4: online disruption and replanning (100 instances, incl. the
    # ---- Frozen arm) ----
    ("exp4", ["exp4_disruption.py", "--num-instances", "100"], True),
    # ---- exp5: ablation and K grid (100 instances; split into 3 jobs by
    # ---- variant family) ----
    ("exp5_seeds",
     ["exp5_ablation.py", "--variants", "full,gating,decoder,replay",
      "--num-instances", "100",
      "--output-dir", os.path.join(RESULTS, "_jobs", "exp5_seeds")], True),
    ("exp5_K",
     ["exp5_ablation.py", "--variants", "K", "--num-instances", "100",
      "--output-dir", os.path.join(RESULTS, "_jobs", "exp5_K")], True),
    ("exp5_eformer",
     ["exp5_ablation.py", "--variants", "eformer", "--num-instances", "100",
      "--output-dir", os.path.join(RESULTS, "_jobs", "exp5_eformer")], True),
    # ---- exp6 block A: scale extrapolation (30 instances/scale; split into 5
    # ---- jobs by method) ----
    *[(f"exp6ext_{m}",
       ["exp6_generalization.py", "--blocks", "extrapolate", "--methods", m,
        "--num-instances", "30",
        "--output-dir", os.path.join(RESULTS, "_jobs", f"exp6ext_{m}")],
       m in ("EFormer", "EFEDQN"))
      for m in ["TD-ACO", "GA", "PSO", "EFormer", "EFEDQN"]],
    # ---- exp6 block B: distribution shift (50 instances/condition; split into
    # ---- 2 jobs by method) ----
    ("exp6ds_ef",
     ["exp6_generalization.py", "--blocks", "distshift", "--methods", "EFEDQN",
      "--num-instances", "50",
      "--output-dir", os.path.join(RESULTS, "_jobs", "exp6ds_ef")], True),
    ("exp6ds_eformer",
     ["exp6_generalization.py", "--blocks", "distshift",
      "--methods", "EFormer", "--num-instances", "50",
      "--output-dir", os.path.join(RESULTS, "_jobs", "exp6ds_eformer")], True),
    # ---- exp6 step-cap sensitivity (depends on the exp6ext instance pools, so it
    # ---- must run after them) ----
    ("exp6_horizon",
     ["exp6_horizon_analysis.py", "--max-steps", "400",
      "--output-dir", os.path.join(RESULTS, "exp6_horizon")], True),
    # ---- exp7: real road network (timings are taken from the clean_rerun rerun) ----
    ("exp7", ["exp7_real_scene.py"], True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        for name, cmd, gpu in JOBS:
            print(f"{'GPU' if gpu else 'CPU':>3} {name:>16}: {PY} {' '.join(cmd)}")
        return

    os.makedirs(os.path.join(RESULTS, "_logs"), exist_ok=True)
    pending = list(JOBS)
    running = {}  # name -> (Popen, gpu, logfile)
    done, failed = [], []

    while pending or running:
        # Launch new jobs
        while pending:
            n_total = len(running)
            n_gpu = sum(1 for (_, g, _) in running.values() if g)
            name, cmd, gpu = pending[0]
            if n_total >= MAX_TOTAL or (gpu and n_gpu >= MAX_GPU):
                break
            pending.pop(0)
            log = open(os.path.join(RESULTS, "_logs", f"{name}.log"), "w",
                       encoding="utf-8")
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["PYTHONIOENCODING"] = "utf-8"
            p = subprocess.Popen([PY] + cmd, cwd=TEST_DIR, stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
            running[name] = (p, gpu, log)
            print(f"[{time.strftime('%H:%M:%S')}] 启动 {name} "
                  f"(pid {p.pid}, {'GPU' if gpu else 'CPU'}; "
                  f"运行中 {len(running)}, 待启 {len(pending)})", flush=True)

        time.sleep(15)
        for name in list(running):
            p, gpu, log = running[name]
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            del running[name]
            (done if rc == 0 else failed).append(name)
            print(f"[{time.strftime('%H:%M:%S')}] 结束 {name} (rc={rc}; "
                  f"完成 {len(done)} 失败 {len(failed)} 运行中 {len(running)} "
                  f"待启 {len(pending)})", flush=True)

    print(f"\n全部结束: 成功 {len(done)}, 失败 {len(failed)}")
    if failed:
        print("失败作业:", ", ".join(failed))
    print("合并命令: python merge_results.py")


if __name__ == "__main__":
    main()
