"""EFormer tester for S-TSP-TVRA (greedy evaluation protocol).

Two modes:
1. Pool mode (default, --instances-dir): load the certified benchmark/*.json
   instance pool and run a greedy rollout per instance, reporting
   cost / feasible / gap (vs Gurobi) / time on the same instances and metrics,
   so runs can be paired.
2. Random mode (--mode random): generate instances on the fly for the five
   conditions (25n_16tw, ...) of the standard evaluation protocol.

Optional --aug-k > 1: take the best of K rollouts per instance (the best-of-K
protocol used for equal search-budget comparisons); the default is one greedy
rollout.
"""

import csv
import glob
import logging
import os
import random
import time

import numpy as np
import torch

from STSPEnv import STSPEnv as Env
from STSPModel import STSPModel as Model
from STSPProblemDef import (generate_test_instance, instance_from_json,
                            instance_to_tensors)

# Same conditions as evaluate.py
CONDITIONS = [
    ("25n_16tw", 25, 16, 8),
    ("50n_32tw", 50, 32, 16),
    ("100n_64tw", 100, 64, 32),
    ("100n_32tw", 100, 32, 32),
    ("100n_128tw", 100, 128, 32),
]


class STSPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.tester_params = tester_params
        self.logger = logging.getLogger(name="tester")
        self.device = torch.device(tester_params.get("device",
                                    "cuda" if torch.cuda.is_available() else "cpu"))

        # Seed all random sources so evaluation is reproducible in both modes.
        # Greedy rollouts also consume the GPU RNG: the MatNet-style col_emb of
        # the Precoder calls torch.rand on every pre_forward, and best-of-K
        # sampling uses multinomial. Instances follow a fixed sorted(glob) order,
        # so the RNG consumption sequence is deterministic and a rerun with the
        # same seed gives identical results. Seeding precedes model construction,
        # so randomly initialized weights are reproducible as well.
        seed = tester_params.get("seed", 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)          # GPU-side random source (col_emb / sampling)
        torch.backends.cudnn.deterministic = True  # deterministic cuDNN algorithms
        torch.backends.cudnn.benchmark = False     # disable autotuning (it adds randomness)

        self.model = Model(**model_params).to(self.device)
        model_load = tester_params["model_load"]
        if model_load.get("enable"):
            checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], False)
            self.logger.info("Loaded model from %s", checkpoint_fullname)
        self.model.eval()

        self.env = Env(**env_params)

    def _greedy_rollout(self, inst):
        """Single greedy rollout for one instance; returns (cost, success, steps, time_ms)."""
        self.model.eval()
        t0 = time.perf_counter()
        tensors = instance_to_tensors([inst], self.device)
        self.env.load_problems(tensors)
        reset_state, _, _ = self.env.reset()
        with torch.no_grad():
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
        cost = float(self.env.cost[0, 0].item())
        success = bool(self.env.success[0, 0].item())
        return cost, success, self.env.step_count, (time.perf_counter() - t0) * 1000.0

    def _sampled_rollouts(self, inst, k):
        """K sampled rollouts (the pomo dimension is reused as K parallel
        samples); returns (best feasible cost or minimum cost, any feasible, ms)."""
        t0 = time.perf_counter()
        tensors = instance_to_tensors([inst], self.device)
        # Reuse the pomo dimension as K parallel sampled rollouts
        old_pomo = self.env.pomo_size
        self.env.pomo_size = k
        self.env.load_problems(tensors)
        reset_state, _, _ = self.env.reset()
        # forward takes the multinomial sampling branch: temporarily switch
        # eval_type to softmax (stay in eval mode so BatchNorm running stats are
        # not polluted by sampled data)
        was_training = self.model.training
        self.model.eval()
        old_eval_type = self.model.model_params["eval_type"]
        self.model.model_params["eval_type"] = "softmax"
        with torch.no_grad():
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
        self.model.model_params["eval_type"] = old_eval_type
        if was_training:
            self.model.train()
        costs = self.env.cost[0].cpu().numpy()
        succs = self.env.success[0].cpu().numpy()
        feas_costs = costs[succs]
        best_cost = float(feas_costs.min()) if feas_costs.size else float(costs.min())
        any_feasible = bool(succs.any())
        self.env.pomo_size = old_pomo
        return best_cost, any_feasible, (time.perf_counter() - t0) * 1000.0

    def _seed_reencoded_rollouts(self, inst, k):
        """Instance augmentation by seed resampling (--aug-mode seed): copy one
        instance K times along the batch dimension and let every pre_forward
        redraw the MatNet-style random one-hot column seeds of the Precoder
        (torch.rand is independent per batch element). Decoding stays greedy
        argmax without switching to softmax, so all randomness lies on the
        encoding side and decoding is deterministic for a given encoding."""
        t0 = time.perf_counter()
        tensors = instance_to_tensors([inst] * k, self.device)
        old_pomo = self.env.pomo_size
        self.env.pomo_size = 1
        self.env.load_problems(tensors)
        reset_state, _, _ = self.env.reset()
        with torch.no_grad():
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
        costs = self.env.cost[:, 0].cpu().numpy()
        succs = self.env.success[:, 0].cpu().numpy()
        feas_costs = costs[succs]
        best_cost = float(feas_costs.min()) if feas_costs.size else float(costs.min())
        any_feasible = bool(succs.any())
        self.env.pomo_size = old_pomo
        return best_cost, any_feasible, (time.perf_counter() - t0) * 1000.0

    def run_pool_mode(self):
        """Evaluation on the fixed instance pool (same protocol as EFEDQN evaluate.py --instances-dir)."""
        instances_dir = self.tester_params["instances_dir"]
        files = sorted(glob.glob(os.path.join(instances_dir, "n*", "n*.json")))
        if not files:
            raise SystemExit("no instances found under {}".format(instances_dir))

        rows = []
        for path in files:
            inst = instance_from_json(path)
            solver = inst.get("solver") or {}
            cost, success, steps, t_ms = self._greedy_rollout(inst)

            gap_pct, gap_kind = float("nan"), ""
            if success and solver.get("status") == "OPTIMAL":
                gap_pct = 100.0 * (cost - solver["cost"]) / solver["cost"]
                gap_kind = "opt"
            elif success and solver.get("bound"):
                gap_pct = 100.0 * (cost - solver["bound"]) / solver["bound"]
                gap_kind = "bound"

            rows.append({
                "instance_id": inst["id"], "nodes": inst["n"],
                "tw": int((inst["win_s"] >= 0).sum()),
                "targets": len(inst["targets"]),
                "protocol": "EFormer",
                "cost": cost, "feasible": int(success),
                "steps": steps, "time_ms": t_ms,
                "solver_status": solver.get("status", ""),
                "solver_opt_cost": solver.get("cost", ""),
                "solver_bound": solver.get("bound", ""),
                "gap_pct": gap_pct, "gap_kind": gap_kind,
            })
            self.logger.info("%s: feasible=%d cost=%.1f gap=%s steps=%d t=%.1fms",
                             inst["id"], success, cost,
                             "{:.2f}%".format(gap_pct) if gap_pct == gap_pct else "n/a",
                             steps, t_ms)

            # Optional best-of-K protocol (EFEDQN-E).
            # aug_mode=seed (default): seed resampling augmentation (the copies
            #   redraw col_emb seeds and decode with argmax); sample: step-wise
            #   sampled decoding.
            k = self.tester_params.get("aug_k", 1)
            if k > 1:
                if self.tester_params.get("aug_mode", "seed") == "seed":
                    bcost, bfeas, bt_ms = self._seed_reencoded_rollouts(inst, k)
                    proto = "EFormer-K{}-seed".format(k)
                else:
                    bcost, bfeas, bt_ms = self._sampled_rollouts(inst, k)
                    proto = "EFormer-K{}".format(k)
                bgap, bkind = float("nan"), ""
                if bfeas and solver.get("status") == "OPTIMAL":
                    bgap = 100.0 * (bcost - solver["cost"]) / solver["cost"]
                    bkind = "opt"
                elif bfeas and solver.get("bound"):
                    bgap = 100.0 * (bcost - solver["bound"]) / solver["bound"]
                    bkind = "bound"
                rows.append({
                    "instance_id": inst["id"], "nodes": inst["n"],
                    "tw": int((inst["win_s"] >= 0).sum()),
                    "targets": len(inst["targets"]),
                    "protocol": proto,
                    "cost": bcost, "feasible": int(bfeas),
                    "steps": -1, "time_ms": bt_ms,
                    "solver_status": solver.get("status", ""),
                    "solver_opt_cost": solver.get("cost", ""),
                    "solver_bound": solver.get("bound", ""),
                    "gap_pct": bgap, "gap_kind": bkind,
                })
        return rows

    def run_random_mode(self):
        """Generate instances on the fly for CONDITIONS (main mode of evaluate.py)."""
        # Reseed so the random-mode instance stream does not depend on the RNG
        # consumed by model construction after __init__, making runs identical
        # with or without a loaded checkpoint.
        num_instances = self.tester_params.get("num_instances", 100)
        seed = self.tester_params.get("seed", 2024)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        rows = []
        for name, n, k_v, t_num in CONDITIONS:
            for i in range(num_instances):
                inst = generate_test_instance(n, k_v, t_num)
                cost, success, steps, t_ms = self._greedy_rollout(inst)
                rows.append({
                    "condition": name, "nodes": n, "tw": k_v, "targets": t_num,
                    "instance_id": i, "protocol": "EFormer",
                    "cost": cost, "feasible": int(success),
                    "steps": steps, "time_ms": t_ms,
                    "gap_pct": float("nan"), "gap_kind": "",
                })
        return rows


def summarize(rows):
    """Aggregate by condition / protocol: pass rate, cost, gap and time (as evaluate.py summarize)."""
    groups = {}
    for r in rows:
        key = (r.get("condition", "n{}".format(r["nodes"])), r["protocol"])
        groups.setdefault(key, []).append(r)

    summary = []
    for (condition, protocol), rs in sorted(groups.items()):
        n = len(rs)
        feas = [r["cost"] for r in rs if r["feasible"]]
        z = [0 if r["feasible"] else 1 for r in rs]
        fitness = [6 * r["targets"] / (r["cost"] + 1e4 * zi) for r, zi in zip(rs, z)]
        gaps = [r["gap_pct"] for r in rs
                if r["gap_pct"] == r["gap_pct"]]  # drop NaN
        summary.append({
            "condition": condition, "protocol": protocol, "n_instances": n,
            "pass_rate": z.count(0) / n if n else 0.0,
            "mean_cost_all": sum(r["cost"] for r in rs) / n,
            "mean_cost_feasible": sum(feas) / len(feas) if feas else float("nan"),
            "n_feasible": len(feas),
            "mean_gap_pct": sum(gaps) / len(gaps) if gaps else float("nan"),
            "n_gap": len(gaps),
            "mean_fitness": sum(fitness) / n,
            "mean_time_ms": sum(r["time_ms"] for r in rs) / n,
            "median_time_ms": sorted(r["time_ms"] for r in rs)[n // 2],
        })
    return summary


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_tester(env_params, model_params, tester_params):
    tester = STSPTester(env_params, model_params, tester_params)
    mode = tester_params.get("mode", "pool")

    if mode == "pool":
        rows = tester.run_pool_mode()
    else:
        rows = tester.run_random_mode()

    out_dir = tester_params.get("output_dir", "results")
    detail_path = os.path.join(out_dir, "per_instance_eformer.csv")
    write_csv(detail_path, rows)

    summary = summarize(rows)
    summary_path = os.path.join(out_dir, "summary_eformer.csv")
    write_csv(summary_path, summary)

    tester.logger.info("逐实例明细: %s", detail_path)
    tester.logger.info("汇总结果:   %s", summary_path)
    tester.logger.info("=== 汇总（pass rate / 条件成本 / gap / 平均时间） ===")
    for s in summary:
        tester.logger.info(
            "%10s %12s | pass=%.3f | cost(feas)=%.2f | gap=%.2f%% (n=%d) | time=%.1fms",
            s["condition"], s["protocol"], s["pass_rate"],
            s["mean_cost_feasible"], s["mean_gap_pct"], s["n_gap"],
            s["mean_time_ms"])
    return summary
