"""Instance definition and feature construction for S-TSP-TVRA.

This module converts instances of the time-windowed Steiner TSP setting into the
edge-centric tensor representation expected by EFormer:

- Instance generation uses the generators in Gragh.py and Env.py (sparse
  directed graph, out-degree <= 4, edge lengths and speeds i.i.d. U{1..50},
  edge-level restricted time windows), so the training and test distributions
  are consistent with the evaluation protocol.
- Training distribution: n=100, target_num=n/3, random non-target start,
  departure time U(0,24), number of time windows ~ N(edge/4, edge/8) truncated
  to [16, edge/3].
- Test protocol: a fixed instance pool (benchmark/*.json), start = node 0 (the
  depot), departure time 0, a single greedy rollout, step budget 100.

Time-window semantics: closed interval [s, e] in hours that may wrap past
midnight; edges are blocked inside the window. Waiting is not allowed.
"""

import json
import math
import os
import random
import sys

import numpy as np
import torch

# Directory of the generator modules (Gragh.py / Env.py); single source of instances
_LONG_REWARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "long_reward"))
if _LONG_REWARD_DIR not in sys.path:
    sys.path.insert(0, _LONG_REWARD_DIR)

from Gragh import generate_directed_graph, graph_to_dict  # noqa: E402
from Env import Sample_Env  # noqa: E402

# Number of edge feature channels (see instance_to_tensors)
EDGE_FEAT_DIM = 8
# Normalization constants: edge lengths and speeds divided by 50
NORM_LEN = 50.0
NORM_SPEED = 50.0


def build_instance_from_env(env, start=None, start_time=0.0):
    """Convert a Sample_Env into an instance dict with node-level dense arrays.

    Args:
        env: Sample_Env (already holds map / targets / edge_limits)
        start: start node (None keeps env.agent_pos; random non-target node in training)
        start_time: departure time in hours (U(0,24) in training, 0 in testing)
    Returns:
        dict(n, start, start_time, targets(np.ndarray), adj, length, speed,
             win_s, win_e); the last five are (n, n) numpy arrays, win_* = -1
             where no window exists
    """
    n = env.space_size
    adj = np.zeros((n, n), dtype=bool)
    length = np.zeros((n, n), dtype=np.float32)
    speed = np.zeros((n, n), dtype=np.float32)
    for u in range(n):
        for slot in env.map[u]:
            v, l, s = slot[0], slot[1], slot[2]
            if v != -1:
                adj[u, v] = True
                length[u, v] = float(l)
                speed[u, v] = float(s)

    win_s = np.full((n, n), -1.0, dtype=np.float32)
    win_e = np.full((n, n), -1.0, dtype=np.float32)
    for (u, v, s, e) in env.edge_limits:
        # At most one window per directed edge (generator guarantee); keep the first
        if win_s[u, v] < 0:
            win_s[u, v] = float(s)
            win_e[u, v] = float(e)

    if start is None:
        start = env.agent_pos if env.agent_pos is not None else 0
    return {
        "n": n,
        "start": int(start),
        "start_time": float(start_time),
        "targets": np.array(sorted(env.goal_states1), dtype=np.int64),
        "adj": adj,
        "length": length,
        "speed": speed,
        "win_s": win_s,
        "win_e": win_e,
    }


def generate_train_instance(problem_size, k_num=None, target_num=None):
    """Generate an instance from the training distribution."""
    G1 = generate_directed_graph(problem_size)
    map_, node_size, edge_size = graph_to_dict(G1)
    if target_num is None:
        target_num = node_size // 3
    # train.py: targets may include node 0; the start is a random non-target node
    target_nodes = random.sample(range(0, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    env = Sample_Env(map=map_, nodes_num=node_size, edges_num=edge_size,
                     edges=edge, target=target_nodes, times=times, k_num=k_num)
    start = random.choice([x for x in range(node_size) if x not in env.goal_states1])
    start_time = float(np.random.uniform(0.0, 24.0))
    return build_instance_from_env(env, start=start, start_time=start_time)


def generate_test_instance(problem_size, k_num, target_num):
    """Generate an instance from the test distribution."""
    G1 = generate_directed_graph(problem_size)
    map_, node_size, edge_size = graph_to_dict(G1)
    # Test protocol: targets exclude node 0 (the depot), start = 0, departure time = 0
    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    env = Sample_Env(map=map_, nodes_num=node_size, edges_num=edge_size,
                     edges=edge, target=target_nodes, times=times, k_num=k_num)
    return build_instance_from_env(env, start=0, start_time=0.0)


def instance_from_json(path):
    """Load a JSON instance from the benchmark pool (certified by instances.py).

    Time windows are stored as integer minutes in the JSON and converted back to
    hours here, matching env_from_dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        inst = json.load(f)
    n = inst["nodes"]
    adj = np.zeros((n, n), dtype=bool)
    length = np.zeros((n, n), dtype=np.float32)
    speed = np.zeros((n, n), dtype=np.float32)
    for u, row in enumerate(inst["map"]):
        for slot in row:
            v, l, s = slot[0], slot[1], slot[2]
            if v != -1:
                adj[u, v] = True
                length[u, v] = float(l)
                speed[u, v] = float(s)
    win_s = np.full((n, n), -1.0, dtype=np.float32)
    win_e = np.full((n, n), -1.0, dtype=np.float32)
    for (u, v, s_min, e_min) in inst["edge_limits"]:
        if win_s[u, v] < 0:
            win_s[u, v] = s_min / 60.0
            win_e[u, v] = e_min / 60.0
    return {
        "id": inst["id"],
        "n": n,
        "start": 0,
        "start_time": 0.0,
        "max_steps": inst.get("max_steps", 100),
        "targets": np.array(sorted(inst["targets"]), dtype=np.int64),
        "adj": adj,
        "length": length,
        "speed": speed,
        "win_s": win_s,
        "win_e": win_e,
        "solver": inst.get("solver"),
    }


def time_sincos(t_hour, periods=(0.5, 2.0, 8.0, 24.0)):
    """Multi-period sin/cos time encoding in hours (as in Env.time_to_multi_sincos)."""
    feats = []
    for T in periods:
        rad = 2.0 * math.pi * t_hour / T
        feats.append(math.sin(rad))
        feats.append(math.cos(rad))
    return feats


def instance_to_tensors(instances, device):
    """Convert a batch of equal-size instances into EFormer input tensors.

    Returns a dict:
        x_edges     (B,N,N)  long   edge existence tag: 0 = none, 1 = directed edge, 2 = diagonal
        edge_feats  (B,N,N,8) float edge features (see below)
        cost_mat    (B,N,N)  float  Precoder cost matrix: normalized length, 1.5 sentinel for absent edges, 0 on the diagonal
        node_flags  (B,N,2)  float  [is start / depot, is target]
        adj         (B,N,N)  bool   adjacency (used by the dynamic mask)
        length      (B,N,N)  float  raw edge length (cost accumulation)
        travel_time (B,N,N)  float  travel time in hours (speed / 60)
        win_s/win_e (B,N,N)  float  restricted window bounds in hours, -1 if none
        start       (B,)     long
        start_time  (B,)     float
        target_mask (B,N)    bool

    Eight edge feature channels (window encoding relative to departure time t0,
    the one-shot way of encoding time):
        0: length/50           1: speed/50            2: edge exists
        3: has restricted window  4/5: sin/cos((s-t0)%24 / 24 * 2pi)
        6/7: sin/cos((e-t0)%24 / 24 * 2pi)
    """
    bsz = len(instances)
    n = instances[0]["n"]
    assert all(inst["n"] == n for inst in instances), "一批实例的节点数必须一致"

    adj = np.stack([inst["adj"] for inst in instances])
    length = np.stack([inst["length"] for inst in instances])
    speed = np.stack([inst["speed"] for inst in instances])
    win_s = np.stack([inst["win_s"] for inst in instances])
    win_e = np.stack([inst["win_e"] for inst in instances])

    # x_edges tags: diagonal = 2
    x_edges = np.where(adj, 1, 0).astype(np.int64)
    idx = np.arange(n)
    x_edges[:, idx, idx] = 2

    # Cost matrix: normalized lengths; absent edges use 1.5 (out-of-range sentinel,
    # distinct from the longest edge 1.0); diagonal 0
    cost_mat = np.where(adj, length / NORM_LEN, 1.5).astype(np.float32)
    cost_mat[:, idx, idx] = 0.0

    # Time-window features (relative to t0, sin/cos with a 24 h period)
    has_win = win_s >= 0
    rel_s = np.where(has_win, (win_s - np.array([inst["start_time"] for inst in instances])[:, None, None]) % 24.0 / 24.0, 0.0)
    rel_e = np.where(has_win, (win_e - np.array([inst["start_time"] for inst in instances])[:, None, None]) % 24.0 / 24.0, 0.0)
    edge_feats = np.stack([
        np.where(adj, length / NORM_LEN, 0.0),
        np.where(adj, speed / NORM_SPEED, 0.0),
        adj.astype(np.float32),
        has_win.astype(np.float32),
        np.sin(2 * np.pi * rel_s),
        np.cos(2 * np.pi * rel_s),
        np.sin(2 * np.pi * rel_e),
        np.cos(2 * np.pi * rel_e),
    ], axis=-1).astype(np.float32)

    # Travel time in hours
    travel_time = np.where(adj, speed / 60.0, 0.0).astype(np.float32)

    starts = np.array([inst["start"] for inst in instances], dtype=np.int64)
    start_times = np.array([inst["start_time"] for inst in instances], dtype=np.float32)
    target_mask = np.zeros((bsz, n), dtype=bool)
    for b, inst in enumerate(instances):
        target_mask[b, inst["targets"]] = True
    node_flags = np.stack([
        (np.arange(n)[None, :] == starts[:, None]).astype(np.float32),
        target_mask.astype(np.float32),
    ], axis=-1)

    to = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    return {
        "x_edges": to(x_edges, torch.long),
        "edge_feats": to(edge_feats, torch.float32),
        "cost_mat": to(cost_mat, torch.float32),
        "node_flags": to(node_flags, torch.float32),
        "adj": to(adj, torch.bool),
        "length": to(length, torch.float32),
        "travel_time": to(travel_time, torch.float32),
        "win_s": to(win_s, torch.float32),
        "win_e": to(win_e, torch.float32),
        "start": to(starts, torch.long),
        "start_time": to(start_times, torch.float32),
        "target_mask": to(target_mask, torch.bool),
    }
