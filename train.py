import torch
from Env import Sample_Env
from Agent import DQNAgent
import time
import gc
from Gragh import *
import pandas as pd
import os

def normal_int_sample(mean, std, min_val, max_val):
    """
    生成符合正态分布的整数，限制在[min_val, max_val]范围内

    参数:
        mean: 正态分布的均值（默认取4和50的中间值27）
        std: 标准差（控制分布分散程度，默认10）
        min_val: 最小值（默认4）
        max_val: 最大值（默认50）
    返回:
        符合条件的整数
    """
    # 生成正态分布的浮点数
    num = random.gauss(mean, std)
    # 四舍五入为整数
    int_num = round(num)
    # 检查是否在指定范围内，不在则重新生成
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
    """
    将二维列表All_reward中的每行数据绘制为单独的SVG图表

    参数:
    All_reward: 二维列表，每行代表一条曲线的数据
    output_prefix: 输出SVG文件的前缀名
    """
    # 设置中文字体，确保中文正常显示
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 获取数据的行数，即需要绘制的图表数量
    num_plots = len(All_reward)

    for i in range(num_plots):
        # 创建新的图表
        plt.figure(figsize=(10, 6))

        # 获取当前行的数据
        data = All_reward[i]

        # 生成x轴数据（假设是0,1,2,...n-1）
        x = np.arange(len(data))

        # 绘制曲线
        plt.plot(x, data, linestyle='-', linewidth=2)

        # 添加标题和标签
        plt.title(f'Reward curve')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存为SVG格式
        output_filename = f"{output_prefix}{i + 1}.svg"
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"saved picture: {output_filename}")

        # 关闭当前图表，防止内存占用过大
        plt.close()

episodes = 5000

envs = []
check_envs = []
test_envs = [[],[],[],[],[],[]]
random.seed(42)
np.random.seed(42)

for i in range(6):
    # target_num = 8
    # target_nodes = random.sample(range(1, 26), target_num)
    # edge = [random.choice([10, 20, 30, 40, 50]) for _ in range(74)]
    # # risk = [random.choice([0, 10, 30, 40, 60]) for _ in range(74)]
    # envs.append(Sample_Env(map=None,nodes_num=None,edges_num=None,edges=edge,target=target_nodes))
    for _ in range(5):
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
        edge = [random.choice([10, 20, 30, 40, 50]) for _ in range(edge_size)]
        times = [random.choice([10, 20, 30, 40, 50]) for _ in range(edge_size)]
        test_envs[i].append(Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes,times
        = times,k_num=k_v))

for i in range(10000):
    # target_num = 8
    # target_nodes = random.sample(range(0, 26), target_num)
    # edge = [random.choice([10, 20, 30, 40, 50]) for _ in range(74)]
    # # risk = [random.choice([0, 10, 30, 40, 60]) for _ in range(74)]
    # envs.append(Sample_Env(map=None,nodes_num=None,edges_num=None,edges=edge,target=target_nodes))
    n = random.randint(25, 100)
    G1 = generate_directed_graph(n)
    # visualize_directed_graph(G1)
    map, node_size, edge_size = graph_to_dict(G1)
    # target_num = 4
    # target_num = normal_int_sample(node_size / 5, 3, 1, node_size - 1)
    target_num = int(node_size / 3)
    target_nodes = random.sample(range(0, node_size), target_num)
    # edge = [random.choice([10, 20, 30, 40, 50]) for _ in range(edge_size)]
    edge = [random.randint(10, 50) for _ in range(edge_size)]
    times = [random.randint(10, 50) for _ in range(edge_size)]
    envs.append(Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes,times = times))

train_envs = envs[:]
sync_interval = 50
agent = DQNAgent(4)

train_start_times = np.random.uniform(low=0, high=24, size=1)
# train_time_num = len(train_start_times)
test_episodes = 1
seeds_num = 1
train_seeds = [0,1,2,3,4]

off_reward = [[[] for _ in range(5)] for _ in range(6)]
On_reward = [[[] for _ in range(5)] for _ in range(6)]

for hh in range(test_episodes):
    count = 0
    agent.reset_qnet()
    Astep = 0
    for i in range(int(episodes)):
        env = train_envs[i%10000]
        for j in train_start_times:
            for k in range(seeds_num):
                t_start_time = j
                env.reset_agent_time(t_start_time)
                env.reset_env()
                env.reset_agent_pos(random.choice([x for x in range(env.space_size) if x not in env.goal_states1]))
                done = False
                train_start_time = time.time()
                t1 = 0
                t2 = 0
                # train_seed = random.randint(0, 100000)
                train_seed = train_seeds[k]
                state, nodes_size = env.get_state()
                while not done:
                    action,weight = agent.get_action(state,nodes_size)
                    next_state, reward, done, _ = env.one_step(action)
                    a_loss = agent.update(state, action, reward / 10, next_state, done,nodes_size)
                    state = copy.deepcopy(next_state)
                    t1 += 1
                    t2 += 1
                    Astep += 1
                    if time.time()-train_start_time >= 1:
                        print(f"这一秒走了{t2}步")
                        t2 = 0
                        train_start_time = time.time()
                    if t1 >= 50 and agent.epsilon <= 0.1:
                        agent.epsilon += 0.9
                    if Astep >= 10:
                        agent.sync_qnet()
                        Astep = 0
                count+=1
                agent.current_step+=1
                print(f"第{count}完成")
                agent.epsilon = 0.1 + (1 - 0.1) * np.exp(-count / 2000)
                if count % 1000 == 0:
                    gc.collect()  # 手动触发垃圾回收
                if count % 10 == 0:
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
                                action, weight = agent.get_action(state, nodes_size, is_test=True)
                                actions.append((action, weight))
                                next_state, reward, done, violation = env.one_step(action)
                                total_reward += reward
                                step += 1
                                v[0] += violation[0]
                                v[1] += violation[1]
                                if step >= 50:
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
                                    action, weight = agent.get_action(state, nodes_size, is_test=True)
                                    actions.append((action, weight))
                                    next_state, reward, done, violation = env.one_step(action)
                                    total_reward += reward
                                    step += 1
                                    v[0] += violation[0]
                                    v[1] += violation[1]
                                    if step >= 50:
                                        break
                                if total_reward > max_reward:
                                    max_reward = total_reward
                            On_reward[g][e].append(max_reward)

torch.save(agent.qnet, 'model.pth')

save_dir = os.path.join(os.getcwd(), "output")

# 2. 如果文件夹不存在，自动创建它（关键！否则会报错）
os.makedirs(save_dir, exist_ok=True)
# =========================================================

# 导出文件
for file_index in range(6):
    data_dict = {}
    for col_index in range(5):
        data_dict[f"Column_{col_index + 1}"] = off_reward[file_index][col_index]

    df = pd.DataFrame(data_dict)

    # 3. 拼接完整的文件路径（文件夹路径 + 文件名）
    filename = os.path.join(save_dir, f"off_reward_{file_index + 1}.xlsx")

    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")

# 导出文件
for file_index in range(6):
    data_dict = {}
    for col_index in range(5):
        data_dict[f"Column_{col_index + 1}"] = On_reward[file_index][col_index]

    df = pd.DataFrame(data_dict)

    # 3. 拼接完整的文件路径（文件夹路径 + 文件名）
    filename = os.path.join(save_dir, f"On_reward_{file_index + 1}.xlsx")

    df.to_excel(filename, index=False)
    print(f"已成功生成文件: {filename}")