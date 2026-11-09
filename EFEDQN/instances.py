"""Benchmark instance pool generation + exact-solver certification.

Protocol (mirrors evaluate.py's make_env exactly, with minute-precision windows):
- generate_directed_graph(n) -> graph_to_dict -> random arc lengths/durations in [1,50]
- k_num restricted arcs with random windows [s, s + U(1h, 13h)] mod 24h, quantised to
  whole minutes so the Gurobi model (integer minutes) is exact.
- An instance is kept only if Gurobi proves the windowed problem OPTIMAL within
  max_steps (the evaluate.py protocol limit); the solve result IS the certification.
- Window-relevance property: the optimal walk of the relaxed problem WITHOUT time
  windows must become infeasible once the windows are enforced. Certified by
  requiring the windowed optimum to strictly exceed the free-window optimum; the
  free-optimal path is also replayed as a guard.
- Attempts failing any check are regenerated with fresh seeds until exactly
  cfg["count"] instances are kept, so the final pool size never shrinks.

Scales: 50 instances each at 10/20/30/50 nodes; solver horizons
50/50/50/60 steps; time limits 60/600/600/3600 s.

Run:  python instances.py --output-dir benchmark [--scales 10,20,30,50]
      python instances.py --output-dir benchmark --scales 30  (resume/extend)

Logs (all attempts incl. rejected ones) go to console AND {output-dir}/gen_log.txt.
"""

import argparse
import copy
import csv
import json
import logging
import os
import random

import numpy as np

from Env import Sample_Env
from Gragh import generate_directed_graph, graph_to_dict
from exact_solver import replay_path, solve_env

SCALES = {
    10: {"k_v": 6,  "targets": 3,  "count": 50, "time_limit": 60.0,  "max_steps": 50},
    20: {"k_v": 16, "targets": 8,  "count": 50, "time_limit": 600.0, "max_steps": 50},
    30: {"k_v": 16, "targets": 8,  "count": 50, "time_limit": 600.0, "max_steps": 50},
    50: {"k_v": 32, "targets": 16, "count": 50, "time_limit": 3600.0, "max_steps": 60},
}

MAX_STEPS = 49
MASTER_SEED = 1


def derive_seed(n, attempt):
    """Seed derivation entry point: MASTER_SEED, scale n and attempt number uniquely determine a seed"""
    return MASTER_SEED * 1000003 + n * 1009 + attempt


def gen_instance(n, k_v, target_num, seed):
    """One instance, same construction as evaluate.make_env (seeded)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))

    G1 = generate_directed_graph(n)
    map_, node_size, edge_size = graph_to_dict(G1)
    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    env = Sample_Env(map=map_, nodes_num=node_size, edges_num=edge_size,
                     edges=edge, target=target_nodes, times=times, k_num=k_v)

    # quantise window ends to whole minutes (solver works in integer minutes)
    q_limits = []
    for (u, v, s, e) in env.edge_limits:
        s_min = int(round(float(s) * 60))
        e_min = int(round(float(e) * 60))
        if e_min == s_min:
            e_min = (e_min + 60) % 1440
        q_limits.append((u, v, s_min, e_min))
    env.edge_limits = [(u, v, s / 60.0, e / 60.0) for (u, v, s, e) in q_limits]
    return env, q_limits


def instance_to_json(env, q_limits, meta):
    n_valid = sum(1 for row in env.map for slot in row if slot[0] != -1)
    return {
        "id": meta["id"],
        "nodes": env.space_size,
        "num_windows": len(q_limits),
        "num_targets": len(env.goal_states1),
        "seed": meta["seed"],
        "attempt": meta["attempt"],
        "max_steps": meta["max_steps"],
        "map": [[list(slot) for slot in row] for row in env.map],
        "targets": list(env.goal_states1),
        "edge_limits": [list(w) for w in q_limits],
        "n_valid_arcs": n_valid,
        "solver": meta["solver"],
    }


def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        inst = json.load(f)
    env = env_from_dict(inst)
    return inst, env


def env_from_dict(inst):
    map_ = [[tuple(slot) for slot in row] for row in inst["map"]]
    limits = [(u, v, s / 60.0, e / 60.0)
              for (u, v, s, e) in inst["edge_limits"]]
    return Sample_Env(map=map_, nodes_num=inst["nodes"],
                      edges_num=inst.get("n_valid_arcs", 0),
                      edges=None, target=list(inst["targets"]),
                      edge_limits=limits, random_limits=False)


def run_scale(n, cfg, output_dir, workers, max_steps=MAX_STEPS):
    scale_dir = os.path.join(output_dir, f"n{n}")
    os.makedirs(scale_dir, exist_ok=True)

    rows = []
    kept = 0
    attempt = 0
    while kept < cfg["count"]:
        seed = derive_seed(n, attempt)
        env, q_limits = gen_instance(n, cfg["k_v"], cfg["targets"], seed)
        env.reset_agent_time(0.0)
        env.reset_env()
        env.reset_agent_pos(0)
        attempt += 1

        env_free = copy.deepcopy(env)
        env_free.edge_limits = []
        free_res = solve_env(env_free, time_limit=cfg["time_limit"],
                             workers=workers, max_steps=max_steps,
                             seed=seed % (2 ** 31))
        if free_res["status"] != "OPTIMAL" or not free_res["replay_match"]:
            logging.info("[n=%d] attempt %d: free-window solve %s -> regenerate",
                         n, attempt, free_res["status"])
            continue
        free_cost = free_res["cost"]

        rep_free = replay_path(env, free_res["path"], max_steps=max_steps)
        if rep_free is not None and rep_free["feasible"]:
            logging.info("[n=%d] attempt %d: free-window optimum stays "
                         "feasible under windows -> regenerate", n, attempt)
            continue

        res = solve_env(env, time_limit=cfg["time_limit"], workers=workers,
                        max_steps=max_steps, seed=seed % (2 ** 31))
        if res["status"] != "OPTIMAL":
            logging.info("[n=%d] attempt %d: %s (not proven optimal) -> "
                         "regenerate", n, attempt, res["status"])
            continue
        if not res["replay_match"]:
            logging.info("[n=%d] attempt %d: %s but replay mismatch -> "
                         "regenerate", n, attempt, res["status"])
            continue

        tw_ref = res["cost"]
        if tw_ref is None or tw_ref <= free_cost + 1e-6:
            logging.info("[n=%d] attempt %d: window gap not certified "
                         "(free=%.0f, tw_ref=%s) -> regenerate",
                         n, attempt, free_cost, tw_ref)
            continue

        kept += 1
        inst_id = f"n{n}_i{kept:02d}"
        meta = {"id": inst_id, "seed": seed, "attempt": attempt,
                "max_steps": max_steps, "solver": {
            "status": res["status"],
            "cost": res["cost"],
            "free_opt_cost": free_cost,
            "window_gap": res["cost"] - free_cost,
            "bound": res["bound"],
            "gap_to_bound_pct": res["gap_to_bound"],
            "wall_time_s": res["wall_time"],
            "replay_feasible": res["replay_feasible"],
            "replay_match": res["replay_match"],
            "path": res["path"],
        }}
        inst = instance_to_json(env, q_limits, meta)
        with open(os.path.join(scale_dir, inst_id + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(inst, f, indent=1)

        rows.append({
            "instance_id": inst_id, "nodes": n, "tw": cfg["k_v"],
            "targets": cfg["targets"], "seed": seed, "attempt": attempt,
            "status": res["status"], "opt_cost": res["cost"],
            "free_opt_cost": free_cost,
            "window_gap": res["cost"] - free_cost,
            "bound": res["bound"], "gap_to_bound_pct": res["gap_to_bound"],
            "wall_time_s": res["wall_time"],
            "replay_match": res["replay_match"],
        })
        logging.info("[n=%d] kept %d/%d (attempt %d): %s cost=%.0f "
                     "free=%.0f bound=%.1f t=%.1fs",
                     n, kept, cfg["count"], attempt, res["status"],
                     res["cost"], free_cost, res["bound"], res["wall_time"])

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="benchmark")
    parser.add_argument("--scales", type=str, default="10,20,30,50")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="step horizon used to certify instances; "
                             "None = use the per-scale value in SCALES")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "gen_log.txt"),
                                encoding="utf-8"),
        ],
    )

    all_rows = []
    for n in [int(s) for s in args.scales.split(",")]:
        cfg = SCALES[n]
        ms = args.max_steps if args.max_steps is not None else cfg["max_steps"]
        logging.info("=== scale n=%d: %d instances, k=%d windows, "
                     "%d targets, time limit %.0fs, max steps %d ===",
                     n, cfg["count"], cfg["k_v"], cfg["targets"],
                     cfg["time_limit"], ms)
        rows = run_scale(n, cfg, args.output_dir, args.workers, max_steps=ms)
        all_rows.extend(rows)

    out_csv = os.path.join(args.output_dir, "solver_results.csv")
    exists = os.path.isfile(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(all_rows)
    logging.info("wrote %d rows -> %s", len(all_rows), out_csv)


if __name__ == "__main__":
    main()
