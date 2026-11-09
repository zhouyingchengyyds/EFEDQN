import torch
from Env import Sample_Env
from Agent import DQNAgent
import time
import gc
from Gragh import *
import pandas as pd
import os
import random
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='global random seed (random/numpy/torch)')
parser.add_argument('--episodes', type=int, default=3000, help='number of training episodes (each episode closes after N_ENVS completed environment rollouts; total pre-generated training instances = episodes * N_ENVS, default 3000*3 = 9000)')
parser.add_argument('--test-envs', type=int, default=15, help='number of test environments per condition for periodic evaluation')
parser.add_argument('--output-dir', type=str, default=None, help='directory for reward xlsx outputs (default: output_seed{seed})')
parser.add_argument('--model-out', type=str, default=None, help='path of the saved checkpoint (default: model_seed{seed}.pth)')
args = parser.parse_args()

# Append the seed to the default names so that results of different seeds do not overwrite each other
if args.output_dir is None:
    args.output_dir = f'output_seed{args.seed}'
if args.model_out is None:
    args.model_out = f'model_seed{args.seed}.pth'

SEED = args.seed
EPISODES = args.episodes
TEST_ENVS_PER_GROUP = args.test_envs
N_ENVS = 3  # parallel environments: one step each per round -> one agent update per round

# Log to both the console and logs/train_seed{seed}_{timestamp}.log
os.makedirs('logs', exist_ok=True)
LOG_FILE = os.path.join('logs', f'train_seed{SEED}_{time.strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f'log file: {LOG_FILE}')
logger.info(f'args: seed={SEED}, episodes={EPISODES}, test_envs={TEST_ENVS_PER_GROUP}, n_envs={N_ENVS}, '
            f'output_dir={args.output_dir}, model_out={args.model_out}')
logger.info(f'pytorch={torch.__version__}, device={"cuda" if torch.cuda.is_available() else "cpu"}')
TRAIN_WALL_START = time.time()

def normal_int_sample(mean, std, min_val, max_val):
    num = random.gauss(mean, std)
    int_num = round(num)
    # Clamp to the allowed range
    if min_val > int_num:
        int_num = min_val
    if max_val < int_num:
        int_num = max_val
    return int_num

def get_average(recodes):
    average_test_rewards = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    for m1 in range(len(average_test_rewards)):
        for n1 in range(len(average_test_rewards[m1])):
            temp = []
            for recode in recodes:
                reward_list = recode[m1][n1][:]
                temp.append(reward_list)
            transposed = zip(*temp)
            average_temp = [sum(col) / len(col) for col in transposed]
            average_test_rewards[m1][n1] = average_temp
    return average_test_rewards

def plot_rewards(All_reward, output_prefix="reward_plot_"):
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    num_plots = len(All_reward)

    for i in range(num_plots):
        plt.figure(figsize=(10, 6))
        data = All_reward[i]
        x = np.arange(len(data))

        plt.plot(x, data, linestyle='-', linewidth=2)

        plt.title(f'Reward curve')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')

        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        output_filename = f"{output_prefix}{i + 1}.svg"
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"saved picture: {output_filename}")

        plt.close()

envs = []
check_envs = []
test_envs = [[],[],[],[],[],[]]
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
        test_envs[i].append(Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes,times
        = times,k_num=k_v))

for i in range(EPISODES * N_ENVS):
    n = 100
    G1 = generate_directed_graph(n)
    map, node_size, edge_size = graph_to_dict(G1)
    target_num = int(node_size / 3)
    target_nodes = random.sample(range(0, node_size), target_num)
    edge = [random.randint(1, 50) for _ in range(edge_size)]
    times = [random.randint(1, 50) for _ in range(edge_size)]
    envs.append(Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes,times = times))

train_envs = envs[:]
sync_interval = 50
agent = DQNAgent(4)

train_start_times = np.random.uniform(low=0, high=24, size=1)

off_reward = [[[] for _ in range(TEST_ENVS_PER_GROUP)] for _ in range(test_num)]
On_reward = [[[] for _ in range(TEST_ENVS_PER_GROUP)] for _ in range(test_num)]

count = 0            # increments once every N_ENVS finished environments, ends at EPISODES
env_done_total = 0   # total finished training environments, ends at EPISODES * N_ENVS
agent.reset_qnet()
Astep = 0            # gradient update counter; the target network is synced every 10 updates

t_start_time = train_start_times[0]  # all environments share the same start time

def init_slot(env):
    """Initialise one environment slot and take its first state and action mask"""
    env.reset_agent_time(t_start_time)
    env.reset_env()
    env.reset_agent_pos(random.choice([x for x in range(env.space_size) if x not in env.goal_states1]))
    state, nodes_size = env.get_state()
    # Action mask: exclude moves that violate a road time window or leave the graph
    action_mask = env.get_action_mask()
    return {'env': env, 'state': state, 'nodes_size': nodes_size,
            'action_mask': action_mask,
            'episode_reward': 0.0, 'episode_steps': 0, 'done': False}

pool_idx = 0  # next unused environment of the training pool
slots = []
for _ in range(min(N_ENVS, len(train_envs))):
    slots.append(init_slot(train_envs[pool_idx]))
    pool_idx += 1

# Aggregation window: one summary log line per N_ENVS finished environments
window_rewards = []    # total return of each episode, in completion order
window_steps = []      # number of steps of each episode, in completion order
window_loss_sum = 0.0  # sum of the losses of this window
window_updates = 0     # number of gradient updates in this window
window_wall_start = time.time()

while slots:  # active environment slots remain
    # (1) Select the actions of all active environments in one batched forward pass, so that
    #     every transition of the round comes from the same network version; each environment
    #     uses its own action mask
    actions = agent.get_actions([s['state'] for s in slots],
                                [s['nodes_size'] for s in slots],
                                [s['action_mask'] for s in slots])

    # (2) Step each environment in turn
    transitions = []
    for s, action in zip(slots, actions):
        next_state, reward, done, _ = s['env'].one_step(action)
        # Mask of the next state (used for the target Q value)
        next_action_mask = s['env'].get_action_mask()
        transitions.append((s['state'], action, reward, next_state, done, s['nodes_size'],
                            s['action_mask'], next_action_mask))
        s['episode_reward'] += reward
        s['episode_steps'] += 1
        s['done'] = done
        if not done:
            s['state'] = copy.deepcopy(next_state)
            s['action_mask'] = next_action_mask

    # (3) Store all transitions of this round (one per active environment) in the replay buffer
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

    # (5) Raise the global epsilon again once an episode exceeds 50 steps while exploration has decayed
    for s in slots:
        if s['episode_steps'] >= 50 and agent.epsilon <= 0.1:
            agent.epsilon += 0.9

    # (6) Finished slots: record their statistics and start a new environment; drop the slot when the pool is exhausted
    active_slots = []
    for s in slots:
        if s['done']:
            env_done_total += 1
            window_rewards.append(s['episode_reward'])
            window_steps.append(s['episode_steps'])
            if pool_idx < len(train_envs):
                active_slots.append(init_slot(train_envs[pool_idx]))
                pool_idx += 1
        else:
            active_slots.append(s)
    slots = active_slots

    # (7) Every N_ENVS finished environments: refresh epsilon, log one summary line, run gc and evaluation
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
        logger.info(f'episode {count}/{EPISODES} | reward={np.mean(group_rewards):.4f} | '
                    f'avg_loss={avg_loss:.6f} | steps={np.mean(group_steps):.2f} | '
                    f'epsilon={agent.epsilon:.4f} | time={time.time() - window_wall_start:.2f}s')
        window_wall_start = time.time()
        if count % 1000 == 0:
            gc.collect()  # explicit garbage collection
        if count % 500 == 0:
            for g in range(len(test_envs)):
                envs = test_envs[g]
                for e in range(len(envs)):
                    env=envs[e]
                    env.reset_agent_time(0.00)
                    env.reset_env()
                    env.reset_agent_pos(0)
                    env.shuffle_space()

                    done = False
                    total_reward = 0
                    step = 0
                    v = [0, 0]
                    actions = []
                    while not done:
                        state, nodes_size = env.get_state()
                        action, weight = agent.get_action(state, nodes_size, is_test=True,
                                                          action_mask=env.get_action_mask())
                        actions.append((action, weight))
                        next_state, reward, done, violation = env.one_step(action)
                        total_reward += reward
                        step += 1
                        v[0] += violation[0]
                        v[1] += violation[1]
                        if total_reward <= -100 or step >= 100:
                            break
                    off_reward[g][e].append(total_reward)

            for g in range(len(test_envs)):
                envs = test_envs[g]
                for e in range(len(envs)):
                    max_reward = -10000
                    for _ in range(10):
                        env=envs[e]
                        env.reset_agent_time(0.00)
                        env.reset_env()
                        env.reset_agent_pos(0)
                        env.shuffle_space()

                        done = False
                        total_reward = 0
                        step = 0
                        v = [0, 0]
                        actions = []
                        while not done:
                            state, nodes_size = env.get_state()
                            action, weight = agent.get_action(state, nodes_size, is_test=True,
                                                              action_mask=env.get_action_mask())
                            actions.append((action, weight))
                            next_state, reward, done, violation = env.one_step(action)
                            total_reward += reward
                            step += 1
                            v[0] += violation[0]
                            v[1] += violation[1]
                            if total_reward <= -100 or step >= 100:
                                break
                        if total_reward > max_reward:
                            max_reward = total_reward
                    On_reward[g][e].append(max_reward)

            mean_off = [np.mean([off_reward[g][e][-1] for e in range(TEST_ENVS_PER_GROUP)]) for g in range(test_num)]
            mean_on = [np.mean([On_reward[g][e][-1] for e in range(TEST_ENVS_PER_GROUP)]) for g in range(test_num)]
            logger.info(f'eval@{count} mean_off_reward per group: ' + ' '.join(f'{x:.2f}' for x in mean_off))
            logger.info(f'eval@{count} mean_on_reward per group: ' + ' '.join(f'{x:.2f}' for x in mean_on))

torch.save(agent.qnet, args.model_out)
logger.info(f'model saved: {args.model_out}')
logger.info(f'training finished in {(time.time() - TRAIN_WALL_START) / 3600:.2f} h')

save_dir = os.path.join(os.getcwd(), args.output_dir)

# Create the output directory if it does not exist yet
os.makedirs(save_dir, exist_ok=True)

# Export the off-reward table
for file_index in range(test_num):
    data_dict = {}
    for col_index in range(TEST_ENVS_PER_GROUP):
        data_dict[f"Column_{col_index + 1}"] = off_reward[file_index][col_index]

    df = pd.DataFrame(data_dict)

    # Full path of the output file
    filename = os.path.join(save_dir, f"off_reward_{file_index + 1}.xlsx")

    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")

# Export the on-reward table
for file_index in range(test_num):
    data_dict = {}
    for col_index in range(TEST_ENVS_PER_GROUP):
        data_dict[f"Column_{col_index + 1}"] = On_reward[file_index][col_index]

    df = pd.DataFrame(data_dict)

    # Full path of the output file
    filename = os.path.join(save_dir, f"On_reward_{file_index + 1}.xlsx")

    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")