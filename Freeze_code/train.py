# -*- coding: utf-8 -*-
"""train.py — training script for the frozen-encoding model (model_seed{seed}_no_dy.pth).

Training protocol: training instances are generated up front, three
environment slots run in parallel, epsilon is scheduled per episode, the agent
uses PER + Double DQN with a soft target update (tau = 0.01), periodic
evaluation writes reward xlsx files, and seeds are fixed for random / numpy /
torch.

Model: GAT_DQN_Frozen (GAT_Frozen.py) encodes the graph once at the episode
start (static-key cache reuse) and feeds a four-dimensional elapsed-time
feature to the decoder.

Experience state: [s0_node, s0_edge_idx, s0_edge_attr, pos, elapsed,
goal_mask], where the s0* tensors are the episode-start snapshot used as
encoder input and consistent with the inference-time cached encoding. pos,
elapsed and goal_mask change every step, so the training forward
Q(Encode(s0), pos, elapsed, goal_mask) is the same function that the inference
cache reproduces, with no train/inference mismatch. goal_mask is the 0/1 mask
of remaining targets, shrinking with env.goal_states each step and consumed by
the goal-aware decoder.

Action masking: exploration samples only feasible actions; greedy selection,
TD targets and periodic evaluation all apply the dynamic action mask, matching
inference with use_mask=True.

Output: model_seed{seed}_no_dy.pth stores the whole model; the class definition
lives in GAT_Frozen.py in this directory and is resolved from here at load time.

Episode step cap: MAX_EPISODE_STEPS = 1500. With partial observability a
mid-training greedy policy can wander for a very long time and inflate single
episodes. A slot that reaches the cap is truncated and replaced by a new
environment; the truncation transition enters the buffer with the environment's
true done=False (normal bootstrap, no pseudo-termination) and still counts
towards the completion statistics, so episode counting, epsilon scheduling and
evaluation triggers keep their usual timing.

--train-n (default 100) and --eval-interval (default 500) expose the training
node count and the evaluation interval for small smoke runs.

Usage (from this directory):
  python train.py --seed 1
  python train.py --seed 1 --episodes 1 --test-envs 1 --train-n 10 \
      --eval-interval 1 --model-out _smoke.pth --output-dir _smoke_out   # smoke test
"""

import argparse
import gc
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

# Depend on Env / Gragh from the EFEDQN directory
_EFEDQN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EFEDQN")
if _EFEDQN_DIR not in sys.path:
    sys.path.insert(0, _EFEDQN_DIR)

from Env import Sample_Env                               # noqa: E402
from Gragh import generate_directed_graph, graph_to_dict  # noqa: E402
from Agent_frozen import FrozenDQNAgent                  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1,
                    help='global random seed (random/numpy/torch)')
parser.add_argument('--episodes', type=int, default=3000,
                    help='number of training episodes (also number of '
                         'pre-generated training instances)')
parser.add_argument('--test-envs', type=int, default=15,
                    help='number of test environments per condition for '
                         'periodic evaluation')
parser.add_argument('--train-n', type=int, default=100,
                    help='number of nodes per training instance '
                         '(original code hardcodes 100)')
parser.add_argument('--eval-interval', type=int, default=500,
                    help='periodic evaluation interval in episodes '
                         '(original code hardcodes 100)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='directory for reward xlsx outputs '
                         '(default: output_seed{seed})')
parser.add_argument('--model-out', type=str, default=None,
                    help='path of the saved checkpoint '
                         '(default: model_seed{seed}_no_dy.pth)')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = f'output_seed{args.seed}'
if args.model_out is None:
    args.model_out = f'model_seed{args.seed}_no_dy.pth'

# Guard: Env.normal_int_sample requires 16 <= k <= edge_size/3 when sampling the
# number of time windows. With too few nodes that interval is empty and sampling
# loops forever. At n >= 25 the actual edge count (~140+) is far above the lower
# bound, so the interval is never empty.
if args.train_n < 25:
    raise SystemExit(
        f"--train-n >= 25 required (got {args.train_n}): "
        f"Env 窗口数采样区间 [16, edge_size/3] 在过小图上为空会死循环")

SEED = args.seed
EPISODES = args.episodes
TEST_ENVS_PER_GROUP = args.test_envs
N_ENVS = 3  # Parallel environments: three envs step once per round, three transitions are stored, one agent update follows
MAX_EPISODE_STEPS = 1500  # Per-episode step cap: reaching it truncates the slot and starts a new environment

# Logging: console plus logs/train_frozen_seed{seed}_{timestamp}.log
os.makedirs('logs', exist_ok=True)
LOG_FILE = os.path.join(
    'logs', f'train_frozen_seed{SEED}_{time.strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f'log file: {LOG_FILE}')
logger.info(f'args: seed={SEED}, episodes={EPISODES}, '
            f'test_envs={TEST_ENVS_PER_GROUP}, n_envs={N_ENVS}, '
            f'train_n={args.train_n}, eval_interval={args.eval_interval}, '
            f'output_dir={args.output_dir}, model_out={args.model_out}')
logger.info(f'pytorch={torch.__version__}, '
            f'device={"cuda" if torch.cuda.is_available() else "cpu"}')
TRAIN_WALL_START = time.time()

# Instance pool generation
envs = []
test_envs = [[], [], [], [], [], []]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

test_num = 6
for i in range(test_num):
    for _ in range(TEST_ENVS_PER_GROUP):
        n = None
        k_v = None
        target_num = None
        if i == 0:
            n = 25
            k_v = 16
            target_num = 8
        elif i == 1:
            n = 50
            k_v = 32
            target_num = 16
        elif i == 2:
            n = 100
            k_v = 64
            target_num = 32
        elif i == 3:
            n = 100
            k_v = 32
            target_num = 32
        elif i == 4:
            n = 100
            k_v = 64
            target_num = 32
        elif i == 5:
            n = 100
            k_v = 128
            target_num = 32

        G1 = generate_directed_graph(n)
        map, node_size, edge_size = graph_to_dict(G1)
        target_nodes = random.sample(range(1, node_size), target_num)
        edge = [random.randint(1, 50) for _ in range(edge_size)]
        times = [random.randint(1, 50) for _ in range(edge_size)]
        test_envs[i].append(Sample_Env(map=map, nodes_num=node_size,
                                       edges_num=edge_size, edges=edge,
                                       target=target_nodes, times=times,
                                       k_num=k_v))

for i in range(EPISODES * N_ENVS):
    n = args.train_n
    G1 = generate_directed_graph(n)
    map, node_size, edge_size = graph_to_dict(G1)
    target_num = int(node_size / 3)
    target_nodes = random.sample(range(0, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    envs.append(Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size,
                           edges=edge, target=target_nodes, times=times))

train_envs = envs[:]

# Agent and training loop
agent = FrozenDQNAgent(4)

train_start_times = np.random.uniform(low=0, high=24, size=1)
t_start_time = train_start_times[0]  # all environments share the same initial time

off_reward = [[[] for _ in range(TEST_ENVS_PER_GROUP)] for _ in range(test_num)]
On_reward = [[[] for _ in range(TEST_ENVS_PER_GROUP)] for _ in range(test_num)]

count = 0            # incremented once per N_ENVS completed environments (one episode count)
env_done_total = 0   # total number of completed training environments
agent.reset_qnet()
Astep = 0            # gradient update counter: target network syncs every 10 updates


def init_slot(env):
    """Initialize one environment slot: reset the environment and capture the
    start snapshot s0, the encoding time t0, the initial action mask and the
    initial remaining-target mask.

    Frozen semantics: the encoder input is pinned to the start snapshot (the
    first three tensors of s0); later steps only update pos / elapsed /
    goal_mask / action_mask, which is exactly what inference sees once the
    encoding cache is hit.
    """
    env.reset_agent_time(t_start_time)
    env.reset_env()
    env.reset_agent_pos(random.choice(
        [x for x in range(env.space_size) if x not in env.goal_states1]))
    state, nodes_size = env.get_state()
    s0 = [state[0].clone(), state[1].clone(), state[2].clone()]
    # Action mask: exclude actions that would violate road time windows and directions that cannot be taken
    action_mask = env.get_action_mask()
    return {'env': env, 's0': s0, 't0': float(env.agent_time),
            'pos': int(env.agent_pos), 'nodes_size': nodes_size,
            'goal_mask': goal_flags_of(env), 'action_mask': action_mask,
            'episode_reward': 0.0, 'episode_steps': 0, 'done': False}


def goal_flags_of(env):
    """0/1 mask of the remaining target nodes (env.goal_states shrinks as targets are visited)."""
    mask = torch.zeros(env.space_size, dtype=torch.float32)
    for g in env.goal_states:
        mask[int(g)] = 1.0
    return mask


def make_frozen_state(s0, pos, elapsed, goal_mask):
    """Build the frozen-encoding state: [start snapshot x3, current position, elapsed hours, remaining-target mask]."""
    return [s0[0], s0[1], s0[2],
            torch.tensor([pos], dtype=torch.int),
            torch.tensor(float(elapsed), dtype=torch.float32),
            goal_mask]


def elapsed_of(s):
    return (float(s['env'].agent_time) - s['t0']) % 24.0


pool_idx = 0  # cursor into the training environment pool
slots = []
for _ in range(min(N_ENVS, len(train_envs))):
    slots.append(init_slot(train_envs[pool_idx]))
    pool_idx += 1

# Aggregation window: one summary log line per N_ENVS completed environments
window_rewards = []    # total return of each episode, in completion order
window_steps = []      # step count of each episode, in completion order
window_loss_sum = 0.0  # sum of the losses of all gradient updates in this window
window_updates = 0     # number of gradient updates performed in this window
window_wall_start = time.time()

while slots:  # active environment slots remain
    # (1) Select the actions of all active environments in one batch with the
    #     current network, so every transition comes from the same network
    #     version. States use frozen semantics; exploration and greedy
    #     selection both pick only feasible actions.
    states_now = [make_frozen_state(s['s0'], s['pos'], elapsed_of(s),
                                    s['goal_mask'])
                  for s in slots]
    actions = agent.get_actions(states_now,
                                 [s['nodes_size'] for s in slots],
                                 [s['action_mask'] for s in slots])

    # (2) Step each environment and record the transition
    #     (state_t, a, r, state_{t+1}, current mask, next mask)
    transitions = []
    for s, action, state_t in zip(slots, actions, states_now):
        cur_mask = s['action_mask']  # mask used for the action, kept before the update below
        _, reward, done, _ = s['env'].one_step(action, Q=False)
        s['episode_reward'] += reward
        s['episode_steps'] += 1
        s['done'] = done
        # Mask of the next state, used when computing the target Q value
        next_action_mask = s['env'].get_action_mask()
        if not done:
            s['pos'] = int(s['env'].agent_pos)
            s['goal_mask'] = goal_flags_of(s['env'])
            state_next = make_frozen_state(s['s0'], s['pos'], elapsed_of(s),
                                           s['goal_mask'])
            s['action_mask'] = next_action_mask
        else:
            state_next = state_t  # next_state of a done transition is not used in the TD target
        transitions.append((state_t, action, reward, state_next, done,
                            s['nodes_size'], cur_mask, next_action_mask))

    # (3) Store all transitions of this round (one per active environment)
    for tr in transitions:
        agent.store(*tr)

    # (4) One agent update
    a_loss = agent.train_step()
    if a_loss is not None:
        window_loss_sum += float(a_loss)
        window_updates += 1
    Astep += 1
    if Astep >= 10:
        agent.sync_qnet()
        Astep = 0

    # (5) Once an episode passes 50 steps while epsilon sits at its floor, raise epsilon again
    for s in slots:
        if s['episode_steps'] >= 50 and agent.epsilon <= 0.1:
            agent.epsilon += 0.9

    # (6) Finished slots are recorded and immediately replaced by a new
    #     environment; the slot is dropped once the pool is exhausted. Slots that
    #     reach MAX_EPISODE_STEPS are truncated the same way: their last
    #     transition was stored in (2)/(3) with the environment's true
    #     done=False (normal bootstrap, no pseudo-termination), and statistics
    #     keep advancing.
    active_slots = []
    for s in slots:
        if s['done'] or s['episode_steps'] >= MAX_EPISODE_STEPS:
            env_done_total += 1
            window_rewards.append(s['episode_reward'])
            window_steps.append(s['episode_steps'])
            if pool_idx < len(train_envs):
                active_slots.append(init_slot(train_envs[pool_idx]))
                pool_idx += 1
        else:
            active_slots.append(s)
    slots = active_slots

    # (7) Every N_ENVS completed environments (one episode count): refresh
    #     epsilon, log the summary, run gc and periodic evaluation (evaluation
    #     uses frozen semantics with the inference cache, no mask)
    while env_done_total >= (count + 1) * N_ENVS:
        count += 1
        agent.current_step += 1
        agent.epsilon = 0.1 + (1 - 0.1) * np.exp(-count / 2000)
        group_rewards = window_rewards[:N_ENVS]
        group_steps = window_steps[:N_ENVS]
        window_rewards = window_rewards[N_ENVS:]
        window_steps = window_steps[N_ENVS:]
        avg_loss = window_loss_sum / max(window_updates, 1)
        window_loss_sum = 0.0
        window_updates = 0
        logger.info(f'episode {count}/{EPISODES} | '
                    f'reward={np.mean(group_rewards):.4f} | '
                    f'avg_loss={avg_loss:.6f} | '
                    f'steps={np.mean(group_steps):.2f} | '
                    f'epsilon={agent.epsilon:.4f} | '
                    f'time={time.time() - window_wall_start:.2f}s')
        window_wall_start = time.time()
        if count % 1000 == 0:
            gc.collect()
        if count % args.eval_interval == 0:
            agent.qnet.eval()  # enable the inference encoding cache
            for g in range(len(test_envs)):
                envs_ = test_envs[g]
                for e in range(len(envs_)):
                    env = envs_[e]
                    env.reset_agent_time(0.00)
                    env.reset_env()
                    env.reset_agent_pos(0)
                    env.shuffle_space()
                    # Encode the start snapshot once after reset and shuffle
                    state0, nodes_size0 = env.get_state()
                    s0 = [state0[0].clone(), state0[1].clone(),
                          state0[2].clone()]
                    t0 = float(env.agent_time)

                    done = False
                    total_reward = 0
                    step = 0
                    v = [0, 0]
                    actions = []
                    while not done:
                        fstate = make_frozen_state(s0, int(env.agent_pos),
                                                   (float(env.agent_time)
                                                    - t0) % 24.0,
                                                   goal_flags_of(env))
                        action, weight = agent.get_action(
                            fstate, nodes_size0, is_test=True,
                            action_mask=env.get_action_mask())
                        actions.append((action, weight))
                        _, reward, done, violation = env.one_step(
                            action, Q=False)
                        total_reward += reward
                        step += 1
                        v[0] += violation[0]
                        v[1] += violation[1]
                        if total_reward <= -100 or step >= 100:
                            break
                    off_reward[g][e].append(total_reward)

            for g in range(len(test_envs)):
                envs_ = test_envs[g]
                for e in range(len(envs_)):
                    max_reward = -10000
                    for _ in range(10):
                        env = envs_[e]
                        env.reset_agent_time(0.00)
                        env.reset_env()
                        env.reset_agent_pos(0)
                        env.shuffle_space()
                        state0, nodes_size0 = env.get_state()
                        s0 = [state0[0].clone(), state0[1].clone(),
                              state0[2].clone()]
                        t0 = float(env.agent_time)

                        done = False
                        total_reward = 0
                        step = 0
                        v = [0, 0]
                        actions = []
                        while not done:
                            fstate = make_frozen_state(
                                s0, int(env.agent_pos),
                                (float(env.agent_time) - t0) % 24.0,
                                goal_flags_of(env))
                            action, weight = agent.get_action(
                                fstate, nodes_size0, is_test=True,
                                action_mask=env.get_action_mask())
                            actions.append((action, weight))
                            _, reward, done, violation = env.one_step(
                                action, Q=False)
                            total_reward += reward
                            step += 1
                            v[0] += violation[0]
                            v[1] += violation[1]
                            if total_reward <= -100 or step >= 100:
                                break
                        if total_reward > max_reward:
                            max_reward = total_reward
                    On_reward[g][e].append(max_reward)
            agent.qnet.train()  # back to training mode, clears the cache

            mean_off = [np.mean([off_reward[g][e][-1]
                                 for e in range(TEST_ENVS_PER_GROUP)])
                        for g in range(test_num)]
            mean_on = [np.mean([On_reward[g][e][-1]
                                for e in range(TEST_ENVS_PER_GROUP)])
                       for g in range(test_num)]
            logger.info(f'eval@{count} mean_off_reward per group: '
                        + ' '.join(f'{x:.2f}' for x in mean_off))
            logger.info(f'eval@{count} mean_on_reward per group: '
                        + ' '.join(f'{x:.2f}' for x in mean_on))

# Saving and export
agent.qnet.clear_cache()  # keep cache tensors out of the checkpoint
torch.save(agent.qnet, args.model_out)
logger.info(f'model saved: {args.model_out}')
logger.info(f'training finished in '
            f'{(time.time() - TRAIN_WALL_START) / 3600:.2f} h')

save_dir = os.path.join(os.getcwd(), args.output_dir)
os.makedirs(save_dir, exist_ok=True)

# Export the off/on reward curves
for file_index in range(test_num):
    data_dict = {}
    for col_index in range(TEST_ENVS_PER_GROUP):
        data_dict[f"Column_{col_index + 1}"] = off_reward[file_index][col_index]
    df = pd.DataFrame(data_dict)
    filename = os.path.join(save_dir, f"off_reward_{file_index + 1}.xlsx")
    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")

for file_index in range(test_num):
    data_dict = {}
    for col_index in range(TEST_ENVS_PER_GROUP):
        data_dict[f"Column_{col_index + 1}"] = On_reward[file_index][col_index]
    df = pd.DataFrame(data_dict)
    filename = os.path.join(save_dir, f"On_reward_{file_index + 1}.xlsx")
    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")
