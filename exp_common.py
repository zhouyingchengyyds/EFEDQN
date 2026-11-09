# -*- coding: utf-8 -*-
"""exp_common.py — shared framework for all exp*.py experiments.

Responsibilities:
1. Paths and module aliases: injects the EFEDQN / EFormer / Heuristic
   directories into sys.path and registers the sys.modules['ENV'] / ['Gragh']
   aliases (Heuristic code does `from ENV import *`).
2. Reproducible instance generation: gen_random_instance(...) is deterministic
   per seed and instance and emits JSON dicts compatible with the benchmark pool
   (map/targets/edge_limits in minutes) that can be materialized to disk.
3. Unified replay scoring: replay_actions() — the output of every method (action
   sequence or node sequence) is scored by the same replayer with a shared
   max_steps (default 200, adjustable):
   feasible = all targets visited and return to depot, 0 invalid actions and 0
   window violations.
4. Method adapters: EFEDQN / EFEDQN-E / EFEDQN-noTW / EFEDQN-Frozen /
   component-variant models / EFormer (greedy and best-of-K sampling) /
   TD-ACO / GA / PSO / RollingHorizon, all exposing the same interface
   solve(env, seed) -> metrics dict.
5. Summary output: summarize() / write_csv() / load_solver_results().

Conventions:
- The experiment seed defaults to 1 and is derived per instance, so methods do
  not interfere with each other and results do not depend on execution order.
- The max_steps field of a benchmark instance JSON is only a solver
  certification parameter (50/50/60) and does not constrain other methods; all
  evaluated methods use the shared --max-steps (default 200).
- No_timwin/model_seed1_no_tw.pth: same architecture as GAT_DQN, whole model
  saved; window features are filled with -2 during training, and at inference
  the same input is built by temporarily clearing edge_limits.
- Freeze_code/model_seed1_no_dy.pth: whole model saved; the class definition
  GAT_Frozen.py sits next to the weights in Freeze_code (resolved at load time
  via extra_sys_path, the same loading pattern as the component-variant models).
  The frozen encoder encodes once at the episode start (cached during inference
  under a static key) and feeds a 4-dim elapsed-time feature to the decoder;
  inference applies the dynamic action mask and uses the frozen encoding
  together with the elapsed-time channel.
"""

import copy
import csv
import glob
import hashlib
import io
import json
import os
import random
import sys
import time
from contextlib import redirect_stdout

import numpy as np

# 1. Paths and module aliases
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
EFEDQN_DIR = os.path.join(TEST_DIR, "EFEDQN")
EFORMER_DIR = os.path.join(TEST_DIR, "EFormer")
HEURISTIC_DIR = os.path.join(TEST_DIR, "Heuristic")
BENCHMARK_DIR = os.path.join(TEST_DIR, "benchmark")
REAL_SCENE_DIR = os.path.join(TEST_DIR, "real_scene")
FREEZE_DIR = os.path.join(TEST_DIR, "Freeze_code")
RESULTS_DIR = os.path.join(TEST_DIR, "results")

for _d in (EFEDQN_DIR, EFORMER_DIR, HEURISTIC_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import Env as _EnvMod            # noqa: E402
import Gragh as _GraghMod        # noqa: E402

# Heuristic code references the env via `from ENV import *` (the lowercase
# module resolves on case-insensitive Windows, but the alias must be registered
# explicitly on Linux)
sys.modules.setdefault("ENV", _EnvMod)
sys.modules.setdefault("Gragh", _GraghMod)

Sample_Env = _EnvMod.Sample_Env
is_edge_active = _EnvMod.is_edge_active
generate_directed_graph = _GraghMod.generate_directed_graph
graph_to_dict = _GraghMod.graph_to_dict

# Shared evaluation step cap (overridable by each experiment's --max-steps)
DEFAULT_MAX_STEPS = 200
# Shared experiment seed (reproducibility requirement)
DEFAULT_SEED = 1

# The 5 condition groups of the main comparison (same as CONDITIONS in
# evaluate_random.py)
CONDITIONS = [
    ("25n_16tw",   25,  16,  8, "Table 6: 25 nodes"),
    ("50n_32tw",   50,  32, 16, "Table 6: 50 nodes"),
    ("100n_64tw", 100,  64, 32, "Table 6: 100 nodes / Table 7: 64 tw"),
    ("100n_32tw", 100,  32, 32, "Table 7: 32 tw"),
    ("100n_128tw", 100, 128, 32, "Table 7: 128 tw"),
]

# Scale-extrapolation conditions; windows/targets follow the default table
SCALE_EXTRAP = [
    ("150n_96tw",  150,  96, 48),
    ("200n_128tw", 200, 128, 64),
    ("300n_192tw", 300, 192, 96),
]


# 2. Seed utilities
def derive_seed(*parts):
    """Derive a per-instance / per-method seed: independent of execution order,
    replayable on its own, stable across processes.

    Note: built-in hash(str) cannot be used -- it is salted per process
    (PYTHONHASHSEED), and inconsistent values across processes would break
    reproducibility; MD5 is used instead.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    h = int(hashlib.md5(key).hexdigest()[:8], 16)
    return (DEFAULT_SEED * 1000003 + h) % (2 ** 31)


def seed_everything(seed):
    """Seed all random sources, including the GPU side (needed by the EFormer
    Precoder and multinomial sampling)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# 3. Instance generation / loading / env construction
def _valid_arcs(map_):
    return sum(1 for row in map_ for slot in row if slot[0] != -1)


def prune_edges_to_degree(map_, target_out_deg, seed, max_tries=200):
    """Randomly remove edges (clear slots) until the mean out-degree approaches
    target_out_deg, keeping the graph weakly connected.

    Used for the average-degree dimension of the distribution-shift study (the
    training distribution has mean out-degree ≈3.2, the sparse level ≈2).
    """
    import networkx as nx
    rng = random.Random(seed)
    n = len(map_)
    for _ in range(max_tries):
        cand = copy.deepcopy(map_)
        # Collect removable arcs (preferring those that do not zero out an
        # endpoint's out/in degree; simplified here to all candidates)
        slots = [(u, k) for u in range(n) for k in range(4) if cand[u][k][0] != -1]
        rng.shuffle(slots)
        target_arcs = int(target_out_deg * n)
        cur_arcs = len(slots)
        for u, k in slots:
            if cur_arcs <= target_arcs:
                break
            cand[u][k] = [-1, 0, 0]
            cur_arcs -= 1
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for u in range(n):
            for k in range(4):
                if cand[u][k][0] != -1:
                    g.add_edge(u, cand[u][k][0])
        if nx.is_weakly_connected(g):
            return cand
    raise RuntimeError("prune_edges_to_degree: 无法在删边后保持弱连通")


def _quantize_limits_hour(limits):
    """Quantize windows to whole minutes (matching instances.py):
    [(u,v,s_h,e_h)] -> list in minutes."""
    out = []
    for (u, v, s, e) in limits:
        s_min = int(round(float(s) * 60))
        e_min = int(round(float(e) * 60))
        if e_min == s_min:
            e_min = (e_min + 60) % 1440
        out.append((u, v, s_min, e_min))
    return out


def gen_random_instance(n, k_v, target_num, seed, window_mode="default",
                        out_deg=None, minute_quant=True):
    """Generate one instance with the standard protocol; returns a
    benchmark-compatible inst dict.

    window_mode:
      'default' — window duration 1+U(0,12) hours (native Env protocol, k_num passed in)
      'short'   — window duration 15-60 minutes (distribution shift)
      'long'    — window duration 13-24 hours (distribution shift)
    out_deg: when not None, remove edges until the mean out-degree ≈ out_deg
             (distribution shift)
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))

    G1 = generate_directed_graph(n)
    map_, node_size, _edge_size = graph_to_dict(G1)
    if out_deg is not None:
        map_ = prune_edges_to_degree(map_, out_deg, seed)
        map_ = [[list(s) for s in row] for row in map_]
    edge_size = _valid_arcs(map_)

    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]

    if window_mode == "default":
        env = Sample_Env(map=copy.deepcopy(map_), nodes_num=node_size,
                         edges_num=edge_size, edges=edge, target=target_nodes,
                         times=times, k_num=k_v)
    else:
        env = Sample_Env(map=copy.deepcopy(map_), nodes_num=node_size,
                         edges_num=edge_size, edges=edge, target=target_nodes,
                         times=times, k_num=0)  # k_num=0 -> no windows, defined below
        # Custom windows: sample k_v valid arcs
        arcs = [(u, v) for u in range(node_size) for v in
                [map_[u][k][0] for k in range(4)] if v != -1]
        chosen = random.sample(arcs, min(k_v, len(arcs)))
        limits = []
        for (u, v) in chosen:
            s = random.uniform(0.0, 24.0)
            if window_mode == "short":
                dur = random.uniform(0.25, 1.0)      # 15-60 min
            elif window_mode == "long":
                dur = random.uniform(13.0, 24.0)     # 13–24 h
            else:
                raise ValueError(window_mode)
            limits.append((u, v, s, (s + dur) % 24.0))
        env.edge_limits = limits

    # Quantize windows to whole minutes (as in the certified pool; env and inst
    # share the same quantized values)
    q = _quantize_limits_hour(env.edge_limits)
    env.edge_limits = [(u, v, s / 60.0, e / 60.0) for (u, v, s, e) in q]

    return env_to_inst(env, inst_id=None, seed=seed, minute_limits=q,
                       node_coords=None)


def env_to_inst(env, inst_id, seed, minute_limits, node_coords=None,
                max_steps_cert=None, start_time=0.0):
    """Sample_Env -> benchmark-compatible inst dict (edge_limits in minutes)."""
    return {
        "id": inst_id,
        "nodes": env.space_size,
        "num_windows": len(minute_limits),
        "num_targets": len(env.goal_states1),
        "seed": seed,
        "max_steps": max_steps_cert,  # solver certification parameter only
        "map": [[list(slot) for slot in row] for row in env.map],
        "targets": list(env.goal_states1),
        "edge_limits": [list(w) for w in minute_limits],
        "n_valid_arcs": _valid_arcs(env.map),
        "start_time": start_time,
        "node_coords": node_coords,
    }


def env_from_inst(inst):
    """inst dict -> Sample_Env (edge_limits converted to hours; map converted to
    tuples)."""
    map_ = [[tuple(slot) for slot in row] for row in inst["map"]]
    limits = [(u, v, s / 60.0, e / 60.0)
              for (u, v, s, e) in inst["edge_limits"]]
    env = Sample_Env(map=map_, nodes_num=inst["nodes"],
                     edges_num=inst.get("n_valid_arcs", 0), edges=None,
                     target=list(inst["targets"]), edge_limits=limits,
                     random_limits=False)
    env.reset_agent_time(float(inst.get("start_time", 0.0)))
    env.reset_env()
    env.reset_agent_pos(0)
    return env


def load_benchmark_pool(benchmark_dir=BENCHMARK_DIR):
    """Load the certified instance pool: [(inst_dict, solver_ref)]. solver_ref
    holds the certified optimum / bound."""
    out = []
    for jp in sorted(glob.glob(os.path.join(benchmark_dir, "n*", "n*.json"))):
        with open(jp, encoding="utf-8") as f:
            inst = json.load(f)
        s = inst.get("solver") or {}
        ref = None
        if s:
            ref = {"status": s.get("status"),
                   "opt_cost": float(s["cost"]) if s.get("cost") else None,
                   "bound": float(s["bound"]) if s.get("bound") else None,
                   "wall_time_s": float(s.get("wall_time_s", 0.0))}
        out.append((inst, ref))
    return out


# 4. Unified replay scoring
def node_path_to_actions(env, node_path):
    """Node sequence -> action sequence. Adjacent duplicates are skipped (the
    first EFormer step is a placeholder); a jump without a matching edge is
    recorded as 'INVALID' (counted as a violation on replay)."""
    actions = []
    prev = node_path[0] if node_path else 0
    for nxt in node_path[1:]:
        if nxt == prev:
            continue
        slot_found = "INVALID"
        for k in range(env.action_size):
            if env.map[prev][k][0] == nxt:
                slot_found = k
                break
        actions.append(slot_found)
        prev = nxt
    return actions


def replay_actions(env, actions, max_steps=DEFAULT_MAX_STEPS, init=None,
                   start_time=None):
    """Unified replay scorer (same semantics as evaluate_pool.rollout).

    init: dict(pos, time, goals, start_pos) -- partial initial state for
    scenarios such as continuing after a disruption.
    start_time: overrides the departure time (defaults to the current
    start_time of env, or 0).
    Returns a metrics dict: cost/feasible/completed/steps/invalid/window_viol/
    reward.
    """
    env.reset_env()  # goal_states = goal_states1.copy()
    if init is not None:
        env.agent_pos = init["pos"]
        env.start_pos = init.get("start_pos", 0)
        env.agent_time = init["time"]
        if init.get("goals") is not None:
            env.goal_states = list(init["goals"])
    else:
        env.reset_agent_pos(0)
        env.reset_agent_time(start_time if start_time is not None else 0.0)

    cost = 0.0
    viol = [0, 0]
    steps = 0
    done = False
    total_reward = 0.0
    for a in actions:
        if done or steps >= max_steps:
            break
        if a == "INVALID":
            cost += 1000.0
            viol[0] += 1
            steps += 1
            continue
        cur = env.agent_pos
        nxt = env.map[cur][a][0]
        if nxt == -1:
            cost += 1000.0
        else:
            cost += env.map[cur][a][1]
        _, r, d, v = env.one_step(a, Q=False)
        total_reward += r
        viol[0] += v[0]
        viol[1] += v[1]
        steps += 1
        done = d
    feasible = bool(done and viol[0] == 0 and viol[1] == 0)
    return {
        "cost": cost, "feasible": feasible, "completed": bool(done),
        "steps": steps, "invalid": viol[0], "window_viol": viol[1],
        "reward": total_reward, "n_targets": len(env.goal_states1),
    }


def better_result(a, b):
    """best-of-K selection rule: lexicographic max of (feasible, -cost); None is
    treated as the worst result."""
    if a is None:
        return b
    if b is None:
        return a
    ka = (1 if a["feasible"] else 0, -a["cost"])
    kb = (1 if b["feasible"] else 0, -b["cost"])
    return a if ka >= kb else b


# 5. EFEDQN rollouts (own implementation: action logging / no-TW feature
# separation / partial initial state)
def _load_efedqn_model(path, device):
    import torch
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"模型权重不存在: {path}\n(请先补充对应权重文件后再运行该方法的实验)")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _qnet_forward(qnet, state, nodes_size, device, elapsed=None):
    import torch
    t = [s.to(device) for s in state]
    ns = torch.tensor([nodes_size], dtype=torch.int32).to(device)
    with torch.no_grad():
        if getattr(qnet, "needs_time", False):
            # Frozen encoder (GAT_DQN_Frozen): additionally pass the elapsed
            # time (hours) since the encoding moment, plus the mask of the
            # remaining targets (column 1 of the get_state node features, for
            # goal-aware decoders; only passed when the model declares
            # needs_goals, older frozen models do not receive it)
            gm = t[0][:, 1] if getattr(qnet, "needs_goals", False) else None
            qs, _ = qnet(t[0], t[1], t[2], t[3], ns, elapsed=elapsed,
                         goal_mask=gm)
        else:
            qs, _ = qnet(t[0], t[1], t[2], t[3], ns)
    return qs.view(-1)


def _efedqn_reset_for_rollout(env, init, start_time):
    """Reset before a single rollout (shared by the serial and parallel paths)."""
    env.reset_env()
    if init is not None:
        env.agent_pos = init["pos"]
        env.start_pos = init.get("start_pos", 0)
        env.agent_time = init["time"]
        if init.get("goals") is not None:
            env.goal_states = list(init["goals"])
    else:
        env.reset_agent_pos(0)
        env.reset_agent_time(start_time if start_time is not None else 0.0)


def _efedqn_rollout_body(env, qnet, device, use_mask, no_tw_feat, max_steps,
                         needs_time, encode_t0):
    """Step loop of a single rollout (without reset or permutation): shared by
    the serial path and the parallel workers so both stay bitwise identical.
    The loop consumes no RNG (argmax is deterministic)."""
    import torch
    actions = []
    node_path = [env.agent_pos]
    steps = 0
    done = False
    viol = [0, 0]
    cost = 0.0
    total_reward = 0.0
    while not done and steps < max_steps:
        real_limits = env.edge_limits
        if no_tw_feat:
            env.edge_limits = []
        state, nsize = env.get_state()
        if no_tw_feat:
            env.edge_limits = real_limits
        qs = _qnet_forward(qnet, state, nsize, device,
                           elapsed=((float(env.agent_time) - encode_t0) % 24.0)
                           if needs_time else None)
        if use_mask:
            mask = env.get_action_mask()
            # Skip the mask when every action is masked; the violation count
            # then fails the run
            if any(mask):
                mask_t = torch.tensor(mask, dtype=torch.bool, device=qs.device)
                qs = qs.masked_fill(~mask_t, float("-inf"))
        a = int(np.argmax(qs.detach().cpu().numpy()))
        actions.append(a)
        cur = env.agent_pos
        nxt = env.map[cur][a][0]
        if nxt == -1:
            cost += 1000.0
        else:
            cost += env.map[cur][a][1]
        _, r, d, v = env.one_step(a, Q=False)
        total_reward += r
        viol[0] += v[0]
        viol[1] += v[1]
        steps += 1
        done = d
        node_path.append(env.agent_pos)
    return {
        "cost": cost, "feasible": bool(done and viol[0] == 0 and viol[1] == 0),
        "completed": bool(done), "steps": steps,
        "invalid": viol[0], "window_viol": viol[1],
        "reward": total_reward, "n_targets": len(env.goal_states1),
        "actions": actions, "node_path": node_path,
    }


def rollout_efedqn(env, qnet, device, k=1, use_mask=True, no_tw_feat=False,
                   max_steps=DEFAULT_MAX_STEPS, init=None, permute=True,
                   start_time=None, seed=1):
    """EFEDQN inference (serial reference implementation). k=1 is standard
    greedy with one slot permutation, matching the training-time evaluation
    protocol; k>1 is best-of-K over K permutations (EFEDQN-E).
    Note: the slot permutation is applied in place within map rows and
    reset_env does not restore map, so permutations accumulate across j (and
    across solve calls, e.g. the repeated restarts of wall-clock modes such as
    BudgetWrap) -- this is exactly the diversity source of EFEDQN-E. The
    parallel version (rollout_efedqn_parallel) resets and permutes in the main
    process with the identical RNG consumption sequence, so the accumulated
    history is reproducible.
    no_tw_feat=True: edge_limits are temporarily cleared while building the
    state (window features filled with -2, matching the training input of the
    no-TW model), while the mask and execution still use the real windows.
    Frozen-encoder models (needs_time=True, e.g. GAT_DQN_Frozen): every step
    additionally receives the elapsed time since the encoding moment (episode
    start or continuation point), in hours mod 24.
    init: partial initial state dict(pos, time, goals, start_pos) for
    continuing after a disruption."""
    best = None
    needs_time = getattr(qnet, "needs_time", False)
    for j in range(max(k, 1)):
        seed_everything(derive_seed(seed, "efedqn_rollout", j))
        _efedqn_reset_for_rollout(env, init, start_time)
        if permute:
            env.shuffle_space()
        # Frozen encoder (needs_time): record the encoding moment (episode
        # start or continuation point)
        encode_t0 = float(env.agent_time) if env.agent_time is not None else 0.0
        res = _efedqn_rollout_body(env, qnet, device, use_mask, no_tw_feat,
                                   max_steps, needs_time, encode_t0)
        best = better_result(best, res)
    return best


# 5b. EFEDQN-E parallel worker infrastructure (multiprocessing, bitwise identical
# to the serial path)
_EFEDQN_WORKER_STATE = {}   # inside a worker process: {"qnet", "device"}
_EFEDQN_POOLS = {}          # in the main process: (model_path, gat_dir, device) -> pool


def _efedqn_worker_init(model_path, gat_dir, device_str):
    """Worker process initializer: loads its own qnet (reusing the GAT sys.path
    injection) and sets torch to a single thread so 10 workers do not
    oversubscribe the CPU."""
    import torch
    torch.set_num_threads(1)
    device = torch.device(device_str)
    saved_mod = sys.modules.pop("GAT", None)
    saved_path = sys.path[:]
    if gat_dir:
        sys.path[:] = [gat_dir] + [p for p in sys.path if p != gat_dir]
    try:
        qnet = _load_efedqn_model(model_path, device)
    finally:
        sys.path[:] = saved_path
        if saved_mod is not None:
            sys.modules["GAT"] = saved_mod
    qnet.eval()
    _EFEDQN_WORKER_STATE["qnet"] = qnet
    _EFEDQN_WORKER_STATE["device"] = device


def _efedqn_worker_rollout(task):
    """Worker task: run the deterministic step loop on an env copy that has
    already been reset and permuted."""
    env, use_mask, no_tw_feat, max_steps, encode_t0 = task
    qnet = _EFEDQN_WORKER_STATE["qnet"]
    device = _EFEDQN_WORKER_STATE["device"]
    needs_time = getattr(qnet, "needs_time", False)
    return _efedqn_rollout_body(env, qnet, device, use_mask, no_tw_feat,
                                max_steps, needs_time, encode_t0)


def _efedqn_worker_ping(_):
    """Pre-warm helper: returns only after the worker has finished its
    initializer (weight loading)."""
    return os.getpid()


def get_efedqn_pool(model_path, gat_dir=None, device=None, k=10):
    """Global singleton process pool (spawn context), cached by
    (weights, GAT directory, device) and reused across instances and repeated
    BudgetWrap restarts, so the startup cost is paid only once."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import torch
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (model_path, gat_dir, str(device))
    pool = _EFEDQN_POOLS.get(key)
    if pool is None:
        ctx = mp.get_context("spawn")
        n = max(2, min(int(k), os.cpu_count() or 8))
        pool = ProcessPoolExecutor(
            max_workers=n, mp_context=ctx,
            initializer=_efedqn_worker_init,
            initargs=(model_path, gat_dir, str(device)))
        _EFEDQN_POOLS[key] = pool
    return pool


def prewarm_efedqn_pool(model_path, gat_dir=None, device=None, k=10):
    """Pre-warm the pool: submit k ping tasks and collect them, ensuring every
    worker has loaded the weights -- this keeps the pool startup out of the
    timed window."""
    pool = get_efedqn_pool(model_path, gat_dir, device, k)
    list(pool.map(_efedqn_worker_ping, range(int(k))))


def rollout_efedqn_parallel(env, model_path, gat_dir, device, k=10,
                            use_mask=True, no_tw_feat=False,
                            max_steps=DEFAULT_MAX_STEPS, init=None,
                            permute=True, start_time=None, seed=1):
    """Parallel EFEDQN-E version (bitwise identical to serial rollout_efedqn).

    Key design: the main process replays exactly the same RNG consumption
    sequence as the serial version -- seed_everything -> reset -> shuffle_space
    per round (cheap, an in-place pure-Python row shuffle; and since reset_env
    does not restore map, permutations accumulate across rounds and across solve
    calls because the main process holds the same env object, which matches the
    serial accumulated history including repeated BudgetWrap restarts). Each
    round deep-copies the reset and permuted env and dispatches it to the
    persistent workers; workers only run the deterministic step loop (argmax, no
    RNG). Results are submitted and collected in j order and folded with
    better_result (ties keep the smaller j), exactly as in the serial version.

    Timing: the wall clock of the caller (EFEDQNSolver.solve) covers all K
    rounds, including permutation, deep copies, dispatch and result transfer, the
    same measurement as the serial path."""
    pool = get_efedqn_pool(model_path, gat_dir, device, k)
    futures = []
    for j in range(k):
        seed_everything(derive_seed(seed, "efedqn_rollout", j))
        _efedqn_reset_for_rollout(env, init, start_time)
        if permute:
            env.shuffle_space()
        encode_t0 = float(env.agent_time) if env.agent_time is not None else 0.0
        env_copy = copy.deepcopy(env)
        futures.append(pool.submit(_efedqn_worker_rollout,
                                   (env_copy, use_mask, no_tw_feat,
                                    max_steps, encode_t0)))
    best = None
    for f in futures:
        best = better_result(best, f.result())
    return best


class EFEDQNSolver:
    """Unified adapter for EFEDQN / EFEDQN-E / no-TW / Frozen / component variants.

    With k>1 and parallel=True (default), the K permutation rollouts of best-of-K
    are dispatched to the persistent process pool via
    rollout_efedqn_parallel (bitwise identical to the serial path);
    parallel=False falls back to the serial reference implementation (used for
    consistency checks)."""

    def __init__(self, model_path, device, k=1, use_mask=True, no_tw=False,
                 name="EFEDQN", extra_sys_path=None, parallel=True):
        import torch
        self.device = torch.device(device)
        # torch.load resolves the GAT_DQN class of the whole-model pickle via
        # the sys.modules['GAT'] cache; the variant directories hold GAT
        # modules of the same name, so the cache is cleared and the directory
        # defining the class is temporarily put first on sys.path to resolve it
        # from the intended directory, then the previous state is restored
        gat_dir = extra_sys_path or EFEDQN_DIR
        saved_mod = sys.modules.pop("GAT", None)
        saved_path = sys.path[:]
        sys.path[:] = [gat_dir] + [p for p in sys.path if p != gat_dir]
        try:
            self.qnet = _load_efedqn_model(model_path, self.device)
        finally:
            sys.path[:] = saved_path
            if saved_mod is not None:
                sys.modules["GAT"] = saved_mod
        self.qnet.eval()
        self.k = k
        self.use_mask = use_mask
        self.no_tw = no_tw
        self.name = name
        self.model_path = model_path
        self.extra_sys_path = extra_sys_path
        self.parallel = parallel

    def solve(self, env, seed=1, max_steps=DEFAULT_MAX_STEPS, init=None,
              start_time=None):
        t0 = time.perf_counter()
        if self.k > 1 and self.parallel:
            res = rollout_efedqn_parallel(
                env, self.model_path, self.extra_sys_path, self.device,
                k=self.k, use_mask=self.use_mask, no_tw_feat=self.no_tw,
                max_steps=max_steps, init=init, seed=seed,
                start_time=start_time)
        else:
            res = rollout_efedqn(env, self.qnet, self.device, k=self.k,
                                 use_mask=self.use_mask, no_tw_feat=self.no_tw,
                                 max_steps=max_steps, init=init, seed=seed,
                                 start_time=start_time)
        res["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return res


def make_efedqn_solvers(max_steps_note="", device=None):
    """Pre-build the commonly used EFEDQN-family solvers (with a clear message
    when weights are missing)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    solvers = {}
    solvers["EFEDQN"] = EFEDQNSolver(
        os.path.join(EFEDQN_DIR, "model_seed1.pth"), device, k=1)
    solvers["EFEDQN-E"] = EFEDQNSolver(
        os.path.join(EFEDQN_DIR, "model_seed1.pth"), device, k=10)
    no_tw_path = os.path.join(TEST_DIR, "No_timwin", "model_seed1_no_tw.pth")
    try:
        solvers["EFEDQN-noTW"] = EFEDQNSolver(
            no_tw_path, device, k=1, no_tw=True, name="EFEDQN-noTW")
    except FileNotFoundError as e:
        print(f"[警告] {e} — EFEDQN-noTW 本次留空")
        solvers["EFEDQN-noTW"] = None
    frozen_path = os.path.join(FREEZE_DIR, "model_seed1_no_dy.pth")
    try:
        # The class definition GAT_Frozen.py lives in the Freeze_code directory;
        # the dynamic action mask is enabled (use_mask defaults to True) and the
        # model uses the frozen encoding plus the elapsed-time feature.
        solvers["EFEDQN-Frozen"] = EFEDQNSolver(
            frozen_path, device, k=1, name="EFEDQN-Frozen",
            extra_sys_path=FREEZE_DIR)
    except FileNotFoundError as e:
        print(f"[警告] {e} — EFEDQN-Frozen 本次留空, 补齐权重后重跑即可恢复")
        solvers["EFEDQN-Frozen"] = None
    return solvers


# 6. EFormer adapter (greedy and best-of-K sampling; records node paths for the
# unified replay)
EFORMER_MODEL_PARAMS = {
    "embedding_dim": 256, "sqrt_embedding_dim": 256 ** (1 / 2),
    "head_num": 16, "encoder_layer_num": 6, "qkv_dim": 16,
    "sqrt_qkv_dim": 16 ** (1 / 2), "logit_clipping": 10,
    "ff_hidden_dim": 512, "eval_type": "argmax", "hidden_dim": 256,
    "GCN_dim": 6, "mlp_layers": 3, "aggregation": "mean",
    "ms_hidden_dim": 16, "ms_layer1_init": (1 / 2) ** (1 / 2),
    "ms_layer2_init": (1 / 16) ** (1 / 2), "edge_feat_dim": 8,
}


class EFormerSolver:
    def __init__(self, ckpt_path, device, k=1, name="EFormer",
                 max_steps=DEFAULT_MAX_STEPS, aug_mode="sample"):
        import torch
        from STSPModel import STSPModel
        from STSPEnv import STSPEnv
        self.device = torch.device(device)
        self.model = STSPModel(**EFORMER_MODEL_PARAMS).to(self.device)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"EFormer 权重不存在: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], False)
        self.model.eval()
        self.env = STSPEnv(pomo_size=1, max_steps=max_steps, fail_penalty=1000.0)
        self.k = k
        self.name = name
        self.max_steps = max_steps
        self.aug_mode = aug_mode

    def _build_inst(self, env, start_time):
        from STSPProblemDef import build_instance_from_env
        st = start_time if start_time is not None else float(
            getattr(env, "_inst_start_time", 0.0) or 0.0)
        return build_instance_from_env(env, start=0, start_time=st)

    def solve(self, env, seed=1, max_steps=None, init=None, start_time=None):
        """greedy (k=1) or best-of-K (k>1). Output is scored by the unified
        replayer.

        aug_mode("sample", default) -- step-wise multinomial sampling decoding
        (the pomo dimension is reused for K parallel samples that share one
        encoding).
        aug_mode("seed") -- the instance augmentation used by EFormer: the same
        instance is replicated K times along the batch dimension and each copy
        independently resamples the MatNet-style random one-hot column embedding
        of the Precoder inside one batched pre_forward, while decoding stays
        greedy argmax throughout (no softmax switch). Diversity comes from the
        encoding side: the same instance fed several times yields a unique
        embedding each time, and decoding is deterministic for a given
        encoding."""
        import torch
        from STSPProblemDef import instance_to_tensors
        ms = max_steps or self.max_steps
        self.env.max_steps = ms
        t0 = time.perf_counter()
        seed_everything(derive_seed(seed, "eformer", self.k))
        inst = self._build_inst(env, start_time)
        k = self.k
        seed_mode = (self.aug_mode == "seed") or k == 1
        old_pomo = self.env.pomo_size
        if seed_mode:
            # Seed-style augmentation: batch of K copies (each with its own
            # col_emb seed), pomo=1
            self.env.pomo_size = 1
            tensors = instance_to_tensors([inst] * k, self.device)
        else:
            self.env.pomo_size = k
            tensors = instance_to_tensors([inst], self.device)
        self.env.load_problems(tensors)
        reset_state, _, _ = self.env.reset()
        old_eval = self.model.model_params["eval_type"]
        if not seed_mode and k > 1:
            self.model.model_params["eval_type"] = "softmax"
        paths = [[inst["start"]] for _ in range(k)]
        with torch.no_grad():
            self.model.pre_forward(reset_state)
            state, _, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
                # shape of current: seed mode (K,1) -> [:,0]; sample mode (1,K) -> [0]
                cur = (self.env.current[:, 0] if seed_mode
                       else self.env.current[0]).cpu().tolist()
                for p in range(k):
                    paths[p].append(cur[p])
        self.model.model_params["eval_type"] = old_eval
        self.env.pomo_size = old_pomo

        # Each candidate path is replayed and scored independently, best-of-K
        best = None
        for p in range(k):
            actions = node_path_to_actions(env, paths[p])
            m = replay_actions(env, actions, max_steps=ms, init=init,
                               start_time=start_time)
            m["actions"] = actions
            best = better_result(best, m)
        best["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return best


# 7. Heuristic adapters (TD-ACO / GA / PSO); internal prints are suppressed and
# multiple restarts are supported
class HeuristicSolver:
    """kind in {'aco', 'ga', 'pso'}. restarts: number of independent restarts
    (equal-rollout budget); time_budget_s: equal wall-clock budget (takes
    precedence over restarts, always runs at least once)."""

    def __init__(self, kind, restarts=1, time_budget_s=None, name=None,
                 aco_params=None, ga_params=None, pso_params=None,
                 verbose=False):
        self.kind = kind
        self.restarts = restarts
        self.time_budget_s = time_budget_s
        self.name = name or {"aco": "TD-ACO", "ga": "GA", "pso": "PSO"}[kind]
        # Baseline parameters: TD-ACO 50 ants x 100 iterations; GA/PSO
        # population 200 x 100 iterations. Tuning results (measured with
        # tune_heuristics.py): raising the GA population from 100 to 200 lifts
        # the 50-node pass rate from 20% to 60%; PSO population 100 -> 200 with
        # w 0.7 -> 0.5 lifts 50 nodes from 20% to 50%; TD-ACO alpha 0.1 -> 0.05
        # (stronger window avoidance, same budget) lifts 50 nodes from 60% to
        # 70%. Fair comparisons under larger budgets are covered by the equal
        # forward-trajectory count (K=10 independent restarts) and equal
        # wall-clock budget levels of exp2.
        self.aco_params = dict(ant_num=50, max_iter=100, beta=2.0, gamma=3.0,
                               rho1=0.1, rho2=0.2, alpha=0.05)
        self.aco_params.update(aco_params or {})
        self.ga_params = dict(pop_size=200, max_iter=100, crossover_rate=0.8,
                              mutation_rate=0.1, tour_size=2)
        self.ga_params.update(ga_params or {})
        self.pso_params = dict(pop_size=200, max_iter=100, w=0.5, c1=1.5,
                               c2=1.5)
        self.pso_params.update(pso_params or {})
        self.verbose = verbose

    def _solve_once(self, env, seed, max_steps, start_time):
        seed_everything(derive_seed(seed, self.kind, "run"))
        sink = io.StringIO()
        ctx = redirect_stdout(sink) if not self.verbose else _nullctx()
        with ctx:
            if self.kind == "aco":
                from ASO import ACO_ShortestTime
                aco = ACO_ShortestTime(env=env, start_pos=0,
                                       start_time=start_time or 0.0,
                                       **self.aco_params)
                aco.max_ant_steps = max_steps  # shared step cap
                node_path, _, _ = aco.run()
                actions = node_path_to_actions(env, node_path or [])
            elif self.kind == "ga":
                from GA import GA
                ga = GA(env=env, start_pos=0, start_time=start_time or 0.0,
                        **self.ga_params)
                sol = ga.optimize()
                actions = list(sol.actions) if sol is not None else []
            elif self.kind == "pso":
                from PSO import PSO
                pso = PSO(env=env, start_pos=0, start_time=start_time or 0.0,
                          **self.pso_params)
                sol = pso.optimize()
                actions = list(sol.actions) if sol is not None else []
            else:
                raise ValueError(self.kind)
        return replay_actions(env, actions, max_steps=max_steps,
                              start_time=start_time)

    def solve(self, env, seed=1, max_steps=DEFAULT_MAX_STEPS, init=None,
              start_time=None):
        """The heuristics cannot continue from a partial initial state (init is
        only used to keep the replay start consistent)."""
        t0 = time.perf_counter()
        best = None
        if self.time_budget_s is not None:
            n_runs = 0
            while n_runs == 0 or (time.perf_counter() - t0) < self.time_budget_s:
                r = self._solve_once(env, derive_seed(seed, n_runs),
                                     max_steps, start_time)
                r["restart_idx"] = n_runs
                best = better_result(best, r)
                n_runs += 1
            best["n_restarts"] = n_runs
        else:
            for j in range(max(self.restarts, 1)):
                r = self._solve_once(env, derive_seed(seed, j), max_steps,
                                     start_time)
                r["restart_idx"] = j
                best = better_result(best, r)
            best["n_restarts"] = max(self.restarts, 1)
        best["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return best


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# 8. RollingHorizon adapter (implementation in Re_planning/rolling_horizon.py)
class RollingHorizonAdapter:
    def __init__(self, name="RollingHorizon", max_steps=DEFAULT_MAX_STEPS,
                 two_opt=True, label_cap=20):
        sys.path.insert(0, os.path.join(TEST_DIR, "Re_planning"))
        from rolling_horizon import RollingHorizonSolver
        self.solver = RollingHorizonSolver(max_steps=max_steps,
                                           two_opt=two_opt,
                                           label_cap=label_cap)
        self.name = name

    def solve(self, env, seed=1, max_steps=DEFAULT_MAX_STEPS, init=None,
              start_time=None):
        seed_everything(derive_seed(seed, "rh"))
        t0 = time.perf_counter()
        actions = self.solver.solve(env, init=init,
                                    start_time=start_time, max_steps=max_steps)
        m = replay_actions(env, actions, max_steps=max_steps, init=init,
                           start_time=start_time)
        m["actions"] = actions
        m["time_ms"] = (time.perf_counter() - t0) * 1000.0
        return m


# 9. Summary and output
def summarize(rows, gap_ref=None):
    """Summarize by (condition, protocol): pass/fitness/condition cost/gap/time.
    gap_ref: {instance_id: {'status','opt_cost','bound'}} certified reference
    (optional)."""
    groups = {}
    for r in rows:
        groups.setdefault((r.get("condition", ""), r["protocol"]), []).append(r)

    summary = []
    for (condition, protocol), rs in sorted(groups.items()):
        n = len(rs)
        feas_costs = [r["cost"] for r in rs if r["feasible"]]
        z = [0 if r["feasible"] else 1 for r in rs]
        fitness = [6 * r["n_targets"] / (r["cost"] + 1e4 * zi)
                   for r, zi in zip(rs, z)]
        gaps = [r["gap_pct"] for r in rs
                if r.get("gap_pct") is not None and r["gap_pct"] == r["gap_pct"]]
        summary.append({
            "condition": condition, "protocol": protocol, "n_instances": n,
            "pass_rate": z.count(0) / n if n else 0.0,
            "n_feasible": len(feas_costs),
            "mean_cost_all": sum(r["cost"] for r in rs) / n,
            "mean_cost_feasible": (sum(feas_costs) / len(feas_costs))
                                  if feas_costs else float("nan"),
            "mean_fitness": sum(fitness) / n,
            "mean_gap_pct": (sum(gaps) / len(gaps)) if gaps else float("nan"),
            "n_gap": len(gaps),
            "mean_time_ms": sum(r["time_ms"] for r in rs) / n,
            "median_time_ms": sorted(r["time_ms"] for r in rs)[n // 2],
        })
    return summary


def attach_gap(row, ref):
    """Attach the gap relative to the certified optimum or bound to feasible
    results."""
    row["gap_pct"] = float("nan")
    row["gap_kind"] = ""
    if ref and row["feasible"]:
        if ref["status"] == "OPTIMAL" and ref["opt_cost"]:
            row["gap_pct"] = 100.0 * (row["cost"] - ref["opt_cost"]) / ref["opt_cost"]
            row["gap_kind"] = "opt"
        elif ref.get("bound"):
            row["gap_pct"] = 100.0 * (row["cost"] - ref["bound"]) / ref["bound"]
            row["gap_kind"] = "bound"
    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"写出: {path}")


def save_instances(instances, out_dir):
    """Materialize the generated instance pool to disk (keeps all methods
    strictly paired and reproducible)."""
    os.makedirs(out_dir, exist_ok=True)
    for inst in instances:
        p = os.path.join(out_dir, f"{inst['id']}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(inst, f, indent=1, ensure_ascii=False)
    print(f"实例池已固化: {out_dir} ({len(instances)} 个)")


def print_summary(summary):
    print("\n=== 汇总（pass rate / fitness / 条件成本 / gap / 平均时间） ===")
    for s in summary:
        print(f"{s['condition']:>12} {s['protocol']:>16} | "
              f"pass={s['pass_rate']:.3f} | fit={s['mean_fitness']:.4f} | "
              f"cost(feas)={s['mean_cost_feasible']:.2f} | "
              f"gap={s['mean_gap_pct']:.2f}% (n={s['n_gap']}) | "
              f"time={s['mean_time_ms']:.1f}ms")


def standard_row(condition, inst, protocol, m, extra=None):
    """Build a standard per-instance row."""
    row = {
        "condition": condition, "nodes": inst["nodes"],
        "tw": inst["num_windows"], "targets": inst["num_targets"],
        "instance_id": inst["id"], "protocol": protocol,
        "cost": m["cost"], "feasible": int(m["feasible"]),
        "completed": int(m.get("completed", m["feasible"])),
        "steps": m.get("steps", -1), "n_targets": m["n_targets"],
        "invalid": m.get("invalid", 0), "window_viol": m.get("window_viol", 0),
        "time_ms": m["time_ms"],
    }
    if extra:
        row.update(extra)
    return row


def run_methods_on_instances(methods, instances, condition, max_steps,
                             start_time=None, gap_refs=None, verbose=True):
    """Evaluate all methods sequentially on the instance pool; returns
    per-instance rows.
    methods: {name: solver}; a solver of None skips that method;
    the seed is re-derived before every method and instance (reproducible)."""
    rows = []
    for inst in instances:
        env_master = env_from_inst(inst)
        ref = (gap_refs or {}).get(inst["id"])
        for name, solver in methods.items():
            if solver is None:
                continue
            env = copy.deepcopy(env_master)  # isolate state between methods
            seed = derive_seed(condition, inst["id"], name)
            try:
                m = solver.solve(env, seed=seed, max_steps=max_steps,
                                 start_time=start_time)
            except FileNotFoundError as e:
                print(f"[跳过] {name}: {e}")
                continue
            row = standard_row(condition, inst, name, m)
            attach_gap(row, ref)
            rows.append(row)
            if verbose:
                print(f"[{condition}] {inst['id']} {name}: "
                      f"feasible={row['feasible']} cost={row['cost']:.1f} "
                      f"steps={row['steps']} t={row['time_ms']:.0f}ms")
    return rows
