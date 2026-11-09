# -*- coding: utf-8 -*-
"""exp7_real_scene.py — evaluation on a real road network (Xigong district, Luoyang).

The network is taken from OpenStreetMap (20 nodes: 1 depot + 6 delivery points
+ 13 transit junctions, 29 undirected segments / 58 directed arcs, 8 restricted
segments):
  - fig3-1..4: route maps of EFEDQN-E (best-of-10 over 10 permutations) for the
    4 real departure times (06:00/08:00/12:00/18:00) in lon/lat layout, with
    restricted edges highlighted;
  - pass rate / condition cost / time per instance for the same 4 departures x
    7 methods (TD-ACO/GA/PSO/EFormer/EFormer-K10/EFEDQN/EFEDQN-E).

Notes:
- data source: real_scene/luoyang_stsp_20node.json, built from OSM network
  geometry and the public freight control periods of Luoyang;
- cost c = segment length (m) / 100; travel time follows the operating speed of
  the road class; an arc whose departure time falls inside a restricted window
  [a,b] cannot be entered (closed interval, as in the model);
- the service time (5 min at delivery points) is folded into the travel time of
  every directed arc entering that node (arrival advances the clock by the
  service duration, matching the departure-time semantics D = T + s·z of the
  formulation; service time is 0 on synthetic instances, so the fold is a no-op
  there);
- shared evaluation step cap --max-steps (default 200);
- the cost/time magnitudes differ from the training distribution (1-50), so the
  run is a generalization test under distribution shift and is reported as
  such.

Usage:
  python exp7_real_scene.py
  python exp7_real_scene.py --methods EFEDQN,EFormer   # method subset
"""

import argparse
import copy
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import exp_common as ec

SCENE_JSON = os.path.join(ec.REAL_SCENE_DIR, "luoyang_stsp_20node.json")
DEPART_TIMES = [6.0, 8.0, 12.0, 18.0]
DEPART_LABELS = ["06:00", "08:00", "12:00", "18:00"]


def load_real_scene():
    """luoyang_stsp_20node.json -> benchmark-compatible inst dict + metadata."""
    with open(SCENE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["nodes"]
    arcs = data["arcs"]
    n = len(nodes)
    assert [nd["id"] for nd in nodes] == list(range(n)), "节点 id 需为 0..n-1"

    # Out-adjacency (validates out-degree <= 4)
    out_arcs = {i: [] for i in range(n)}
    for a in arcs:
        out_arcs[a["u"]].append(a)
    for i, lst in out_arcs.items():
        if len(lst) > 4:
            raise RuntimeError(f"节点 {i} 出度 {len(lst)} > 4, 超出动作维")

    # map: fill the out-edge slots of each node in order (same [v, cost,
    # time_min] structure as the generator)
    map_ = [[[-1, 0, 0]] * 4 for _ in range(n)]
    for i in range(n):
        for k, a in enumerate(out_arcs[i]):
            map_[i][k] = [a["v"], float(a["cost"]), float(a["time_min"])]

    # Service-time folding: add the service_min of each node to the travel time
    # of every directed arc entering it — reaching that node advances the clock
    # by s_v minutes, matching the departure-time semantics D = T_u + s_u*z_u
    # of the formulation (revisiting a delivery point is charged again, a
    # conservative rule applied to all methods equally; service time is 0 on
    # synthetic instances, so the fold is a no-op there).
    # Note: only the time changes, the edge cost (length) does not.
    svc = {nd["id"]: float(nd.get("service_min") or 0.0) for nd in nodes}
    for i in range(n):
        for k in range(4):
            v, c, t = map_[i][k]
            if v != -1 and svc.get(v, 0.0) > 0.0:
                map_[i][k] = [v, c, t + svc[v]]

    targets = [nd["id"] for nd in nodes if nd["kind"] == "delivery"]
    limits_min = []
    for a in arcs:
        if a.get("restricted") and a.get("a") is not None:
            limits_min.append((a["u"], a["v"], int(round(a["a"] * 60)),
                               int(round(a["b"] * 60))))

    coords = [(nd["lon"], nd["lat"]) for nd in nodes]
    kinds = [nd["kind"] for nd in nodes]
    # Undirected edges for plotting: (u, v, restricted)
    undirected = []
    seen = set()
    for a in arcs:
        key = (min(a["u"], a["v"]), max(a["u"], a["v"]))
        if key in seen:
            continue
        seen.add(key)
        undirected.append((a["u"], a["v"], bool(a.get("restricted"))))

    inst = {
        "id": "luoyang_20node", "nodes": n,
        "num_windows": len(limits_min), "num_targets": len(targets),
        "seed": 0, "max_steps": None,
        "map": map_, "targets": targets,
        "edge_limits": [list(w) for w in limits_min],
        "n_valid_arcs": len(arcs), "start_time": 0.0,
        "node_coords": coords,
    }
    return inst, kinds, undirected


def actions_to_node_path(env, actions, t0, max_steps):
    env = copy.deepcopy(env)
    env.reset_env()
    env.reset_agent_pos(0)
    env.reset_agent_time(t0)
    path = [env.agent_pos]
    done = False
    for a in actions[:max_steps]:
        if done:
            break
        _, _, done, _ = env.one_step(a, Q=False)
        path.append(env.agent_pos)
    return path


def plot_route(inst, kinds, undirected, node_path, t0, label, out_stem):
    coords = inst["node_coords"]
    mean_lat = np.mean([c[1] for c in coords])
    fig, ax = plt.subplots(figsize=(7, 6))

    # Segments (restricted edges in a highlighted color)
    for (u, v, restricted) in undirected:
        x = [coords[u][0], coords[v][0]]
        y = [coords[u][1], coords[v][1]]
        if restricted:
            ax.plot(x, y, color="#d62728", linewidth=1.6, alpha=0.55,
                    zorder=1)
        else:
            ax.plot(x, y, color="#888888", linewidth=1.0, alpha=0.6, zorder=1)

    # Route (arrows)
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        if u == v:
            continue
        ax.annotate("", xy=coords[v], xytext=coords[u],
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                    lw=2.2, shrinkA=10, shrinkB=10),
                    zorder=3)

    # Nodes
    for i, (x, y) in enumerate(coords):
        if kinds[i] == "depot":
            ax.scatter(x, y, marker="^", s=200, c="#2ca02c",
                       edgecolors="k", zorder=4)
            ax.annotate("depot", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, zorder=5)
        elif kinds[i] == "delivery":
            ax.scatter(x, y, marker="s", s=110, c="#ff7f0e",
                       edgecolors="k", zorder=4)
        else:
            ax.scatter(x, y, marker="o", s=50, c="#bbbbbb",
                       edgecolors="k", zorder=4)

    ax.set_title(f"EFEDQN-E route, departure {label} (t0={t0:g}h)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="^", ls="", color="#2ca02c", label="depot"),
        Line2D([], [], marker="s", ls="", color="#ff7f0e", label="delivery"),
        Line2D([], [], marker="o", ls="", color="#bbbbbb", label="transit"),
        Line2D([], [], color="#d62728", label="restricted road"),
        Line2D([], [], color="#1f77b4", label="planned route"),
    ], loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_stem}.png/.svg/.pdf")


def main():
    parser = argparse.ArgumentParser(description="真实路网评估(洛阳, C9/C10)")
    parser.add_argument('--max-steps', type=int, default=ec.DEFAULT_MAX_STEPS)
    parser.add_argument('--methods', type=str,
                        default="TD-ACO,GA,PSO,EFormer,EFormer-K10,EFEDQN")
    parser.add_argument('--times', type=str, default=None,
                        help='出发时刻覆盖(小时, 逗号分隔; 默认 6.5,8,12,18)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(ec.RESULTS_DIR, 'exp7_real'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    times = ([float(t) for t in args.times.split(",")] if args.times
             else DEPART_TIMES)
    labels = [f"{int(t):02d}:{int(round((t % 1) * 60)):02d}" for t in times]

    inst, kinds, undirected = load_real_scene()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"真实路网: {inst['nodes']} 节点({inst['num_targets']} 配送点), "
          f"{inst['n_valid_arcs']} 有向弧, {inst['num_windows']} 条限行有向弧")

    methods = {}
    want = {m.strip() for m in args.methods.split(",")}
    if "TD-ACO" in want:
        methods["TD-ACO"] = ec.HeuristicSolver("aco")
    if "GA" in want:
        methods["GA"] = ec.HeuristicSolver("ga")
    if "PSO" in want:
        methods["PSO"] = ec.HeuristicSolver("pso")
    if "EFormer" in want:
        methods["EFormer"] = ec.EFormerSolver(
            os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"),
            device, 1, max_steps=args.max_steps)
    if "EFormer-K10" in want:
        methods["EFormer-K10"] = ec.EFormerSolver(
            os.path.join(ec.EFORMER_DIR, "checkpoint-best.pt"), device, 10,
            name="EFormer-K10", max_steps=args.max_steps)
    if "EFEDQN" in want:
        s = ec.make_efedqn_solvers(device=device)
        methods["EFEDQN"] = s["EFEDQN"]
        methods["EFEDQN-E"] = s["EFEDQN-E"]
    if not methods:
        raise SystemExit("未选择方法")

    all_rows = []
    for t0, label in zip(times, labels):
        cond = f"depart_{label.replace(':', '')}"
        print(f"\n=== 出发时刻 {label} ===")
        inst_t = dict(inst)
        inst_t["id"] = f"luoyang_{label.replace(':', '')}"
        rows = ec.run_methods_on_instances(
            methods, [inst_t], cond, args.max_steps, start_time=t0)
        all_rows.extend(rows)
        ec.print_summary(ec.summarize(rows))

        # fig3: EFEDQN-E route map. The seed matches run_methods_on_instances
        # exactly (derive_seed(cond, inst_id, "EFEDQN-E")), so the figure shows
        # the very solution reported in the results. The route is taken from the
        # node_path recorded during rollout whenever possible: it is consistent
        # with the slot permutation of the best round, which avoids the path
        # misalignment caused by replaying action indices across accumulated
        # permutations (identical when k=1, but with k=10 such a replay points
        # at the wrong slots).
        if "EFEDQN-E" in methods:
            env = ec.env_from_inst(inst_t)
            seed = ec.derive_seed(cond, inst_t["id"], "EFEDQN-E")
            m = methods["EFEDQN-E"].solve(env, seed=seed,
                                          max_steps=args.max_steps,
                                          start_time=t0)
            node_path = (m.get("node_path")
                         or actions_to_node_path(env, m.get("actions", []),
                                                 t0, args.max_steps))
            n_hit = sum(1 for t in inst_t["targets"] if t in node_path)
            print(f"[fig3-check] {label}: solver steps={m['steps']}, "
                  f"route steps={len(node_path) - 1}, "
                  f"delivery covered={n_hit}/{len(inst_t['targets'])}, "
                  f"return-to-depot={node_path[-1] == 0}, "
                  f"path={'->'.join(map(str, node_path))}")
            plot_route(inst_t, kinds, undirected, node_path, t0, label,
                       os.path.join(args.output_dir,
                                    f"fig3-{times.index(t0) + 1}"))

    ec.write_csv(os.path.join(args.output_dir, "per_instance.csv"), all_rows)
    ec.write_csv(os.path.join(args.output_dir, "summary.csv"),
                 ec.summarize(all_rows))


if __name__ == "__main__":
    main()
