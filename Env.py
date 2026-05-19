import torch
import numpy as np
import copy
import random
from collections import Counter
import math

def normal_int_sample(mean, std, min_val, max_val):
    """
    生成符合正态分布的整数，限制在[min_val, max_val]范围内

    参数:
        mean: 正态分布的均值
        std: 标准差（控制分布分散程度）
        min_val: 最小值
        max_val: 最大值
    返回:
        符合条件的整数
    """
    # 确保均值和标准差为浮点数
    mean = float(mean)
    std = float(std)
    # 确保 min_val 和 max_val 为整数
    min_val = int(min_val)
    max_val = int(max_val)

    # 重新生成直到数值在范围内
    while True:
        num = random.gauss(mean, std)
        int_num = round(num)
        if min_val <= int_num <= max_val:
            return int_num

def min_max_normalize(edge_attr):
    # 转置以按特征维度处理
    features = list(zip(*edge_attr))
    normalized = []

    for i, feature in enumerate(features):
        # 跳过第5列到第8列（索引4到7）
        if 0 <= i <= 9:
            normalized_feature = list(feature)  # 保持原始值
        else:
            min_val = min(feature)
            max_val = max(feature)
            scale = max_val - min_val
            # 避免除零
            if scale == 0:
                normalized_feature = [0.0] * len(feature)
            else:
                normalized_feature = [(x - min_val) / scale for x in feature]
        normalized.append(normalized_feature)

    # 转回原始形状
    return list(zip(*normalized))


def time_to_multi_sincos(time_val, periods=None):
    """
    小时级转分钟级的多频率正余弦编码：先将小时转换为分钟，再进行多周期编码

    参数:
    time_val -- 时间值（小时级数值，如1.5表示1.5小时，23.9表示23.9小时）
    periods -- 周期列表（默认包含四种分钟级周期：30分钟、2小时、8小时、1天）

    返回:
    拼接的多频率sin和cos特征数组（长度为8，每个周期对应2个特征）
    """
    # 核心步骤：将小时级时间转换为分钟级（1小时 = 60分钟）
    minute_val = time_val * 60  # 例如1.5小时 → 90分钟

    if periods is None:
        # 四种分钟级周期设计（单位：分钟）
        periods = [30, 120, 480, 1440]  # 30分钟、2小时、8小时、1天

    features = []
    for T in periods:
        # 基于分钟级时间和分钟级周期计算弧度
        radians = 2 * np.pi * minute_val / T
        features.append(np.sin(radians))
        features.append(np.cos(radians))

    return np.array(features)

# def time_to_multi_sincos(time_val, periods=None):
#     """
#     多频率正余弦编码：为多个周期计算sin和cos，增强时间特征表达
#
#     参数:
#     time_val -- 时间值（已转换为数值，如"12:30"→12.5）
#     periods -- 多个周期尺度（如[24, 12, 6, 3]表示日、半天、6小时、3小时周期）
#
#     返回:
#     拼接的多频率sin和cos特征列表
#     """
#     if periods is None:
#         periods = [24, 12, 6, 3]
#     features = []
#     for T in periods:
#         radians = 2 * np.pi * time_val / T
#         features.append(np.sin(radians))
#         features.append(np.cos(radians))
#     return features


def is_within_time_period(time_limit_states, agent_time, agent_pos):
    for (i, lower_bound, upper_bound) in time_limit_states:
        # 检查位置是否匹配
        if i != agent_pos:
            continue

        # 处理普通时间段（lower_bound < upper_bound）
        if lower_bound < upper_bound:
            if lower_bound < agent_time < upper_bound:
                return True
        # 处理跨午夜的时间段（lower_bound > upper_bound）
        else:
            # 时间在lower_bound之后，或者在upper_bound之前
            if agent_time > lower_bound or agent_time < upper_bound:
                return True

    return False


def is_edge_active(edges, time_point, a, b):
    """
    判断在指定时间点，从节点a到节点b的边是否存在于边列表中且在其时间限制内

    参数:
    edges -- 边列表，元素格式为 (起点, 终点, 起始时间, 结束时间)
    a -- 起点节点
    b -- 终点节点
    time_point -- 需要判断的时间点

    返回:
    bool -- 如果存在符合条件的边返回True，否则返回False
    """
    for (start_node, end_node, lower_bound, upper_bound) in edges:
        if a != start_node or b != end_node:
            continue
        if lower_bound < upper_bound:
            if lower_bound < time_point < upper_bound:
                return True
        # 处理跨午夜的时间段（lower_bound > upper_bound）
        else:
            # 时间在lower_bound之后，或者在upper_bound之前
            if time_point > lower_bound or time_point < upper_bound:
                return True
    return False

def check_number_in_tuples(num, tuple_list):
    for t in tuple_list:
        if num == t[0]:
            return True,t
    return False,None

def check_edge_in_tuples(m,n, tuple_list):
    for t in tuple_list:
        if m == t[0] and n == t[1]:
            return True,t
    return False,None

def is_overlap(s1, e1, s2, e2):
    # 将时间窗转换为线性区间列表
    # 如果 s < e，区间为 [(s, e)]
    # 如果 s > e (跨午夜)，区间为 [(s, 24), (0, e)]
    def get_intervals(s, e):
        return [(s, e)] if s < e else [(s, 24.0), (0.0, e)]

    intervals1 = get_intervals(s1, e1)
    intervals2 = get_intervals(s2, e2)

    # 两两比较区间是否有交集
    for a_start, a_end in intervals1:
        for b_start, b_end in intervals2:
            # 两个线性区间重叠的条件：max(起点) < min(终点)
            if max(a_start, b_start) < min(a_end, b_end):
                return True
    return False

def find_index_by_first_element(lst, a):
    for index, sublist in enumerate(lst):
        if len(sublist) >= 1 and sublist[0] == a:
            return index
    return -1

class Sample_Env:
    def __init__(self,map = None,nodes_num=None,edges_num=None,edges=None,target=None,edge_limits = None,random_limits=True,times = None,same = False,one_same = False,k_num = None):
        if map is None:
            map=[
                [[-1,0,40], [-1,0,40], [5,14,80], [1,12,40]],#0
                [[-1,0,40], [0,12,40], [6,12,40], [2,18,40]],#1
                [[-1,0,40], [1,18,40], [7,6,80], [3,18,40]],#2
                [[-1,0,40], [2,18,40], [12,16,80], [4,56,80]],#3
                [[-1,0,40], [3,56,80], [14,16,80], [-1,0,40]],#4
                [[0,14,80], [-1,0,40], [8,12,80], [6,18,40]],#5
                [[1,12,40], [5,18,40], [10,16,40], [7,16,40]],#6
                [[2,6,80],  [11,18,40],[6,16,40], [-1,0,40]],#7
                [[5,12,80], [-1,0,40], [15,16,40], [9,12,40]],#8
                [[-1,0,40], [8,12,40], [16,16,40], [10,12,40]],#9
                [[6,16,40], [17,14,40], [9,12,40], [11,16,40]],#10
                [[7,18,40], [10,16,40], [18,14,40], [12,18,40]],#11
                [[3,16,80], [11,18,40], [19,16,40], [-1,0,40]],#12
                [[-1,0,40], [-1,0,40], [-1,0,40], [14,14,40]],#13
                [[4,16,80], [13,14,40], [21,16,40], [-1,0,40]],#14
                [[8,16,40], [-1,0,40], [-1,0,40], [16,14,40]],#15
                [ [15,14,40], [-1,0,40],[9,16,40], [17,18,80]],#16
                [[10,14,40], [16,18,80], [-1,0,40], [18,16,80]],#17
                [[11,14,40], [17,16,80], [22,16,40], [-1,0,40]],#18
                [ [-1,0,40], [23,14,40], [20,32,40],[12,16,40]],#19
                [[-1,0,40],  [24,14,40], [19,32,40],[21,24,40]],#20
                [[14,16,40], [20,24,40], [25,14,80], [-1,0,40]],#21
                [ [-1,0,40], [-1,0,40], [23,18,40],[18,16,40]],#22
                [[19,14,40],  [-1,0,40], [22,18,40],[24,32,80]],#23
                [[20,14,40], [23,32,80], [-1,0,40], [25,24,40]],#24
                [[21,14,80],  [-1,0,40], [24,24,40],[-1,0,40]]#25
                ]
        if target is None:
            target = [20, 12, 18, 14]
        if nodes_num is None:
            nodes_num = len(map)
        if edges_num is None:
            edges_num = 74
        if edge_limits is None and random_limits is False:
            edge_limits = [(2, 7, 7.00, 9.00), (5, 8, 6.00, 8.00), (6, 1, 7.00, 9.00), (16, 9 , 6.00, 8.00), (4, 3 , 8.00, 10.00),(7, 2, 7.00, 9.00), (8, 5, 6.00, 8.00), (1, 6, 7.00, 9.00), (9, 16 , 6.00, 8.00), (3, 4 , 8.00, 10.00)]
        self.map = map
        self.space_size = nodes_num
        self.edge_size = edges_num
        self.action_size = 4
        self.goal_states1 = target[:]
        self.goal_states = None
        self.reset_edge(edges,times, same)
        self.agent_pos = None
        self.edge_limits = self.reset_edge_limits(edge_limits, one_same,k_num)
        self.edges_index, self.edges_attr = None,None
        self.agent_time = None
        self.start_pos = None


    def reset_goal_states(self,target):
        self.goal_states1 = target[:]


    def reset_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos
        self.start_pos = agent_pos

    def reset_env(self):
        self.goal_states = self.goal_states1.copy()

    def reset_agent_time(self, agent_time):
        self.agent_time = agent_time

    def reset_edge(self, edges, times, same):
        if edges is not None:
            if same:
                i = 0
                for j, m in enumerate(self.map):
                    for n in m:
                        if n[0] != -1:
                            idex = find_index_by_first_element(self.map[n[0]], j)
                            n[1] = edges[i]
                            self.map[n[0]][idex][1] = edges[i]
                            n[2] = times[i]
                            self.map[n[0]][idex][2] = times[i]
                            i += 1
            else:
                i = 0
                for j, m in enumerate(self.map):
                    for n in m:
                        if n[0] != -1:
                            n[1] = edges[i]
                            n[2] = times[i]
                            i += 1
        self.map = [
            [tuple(sublist) for sublist in middle_list]
            for middle_list in self.map
        ]

    def reset_edge_limits(self, edge_limits, one_same,k_num =  None):
        """
        one_same的话必须无放回采样
        """
        # 如果已经有数据，直接返回副本，避免深层缩进
        if edge_limits is not None:
            return edge_limits[:]

        edge_limits = []

        # 1. 生成随机数
        if one_same:
            random_numbers = random.sample(range(self.edge_size), 12)
        else:
            # 注意：如果你希望下面的 'num' 大于 1，你必须使用 random.choices。
            # random.sample 保证了元素的唯一性，这意味着 num 永远只会是 0 或 1。
            if k_num == None:
                k_val = normal_int_sample(self.edge_size / 4, self.edge_size / 8, 16, self.edge_size / 3)
            else:
                k_val = k_num
            random_numbers = random.sample(range(self.edge_size), k_val)

        # 预先计算计数，以便后续实现 O(1) 的查找速度
        number_counts = Counter(random_numbers)

        j = 0
        for m in range(self.space_size):
            for n in range(self.action_size):
                target_node = self.map[m][n][0]

                # 跳过无效的边
                if target_node == -1:
                    j += 1
                    continue

                num = number_counts.get(j, 0)

                if num > 0:
                    if one_same:
                        start_limit = random.randint(0, 23)
                        end_limit = (start_limit + 1 + random.uniform(0, 12)) % 24

                        # 如果存在正向/反向边，先将它们移除（比通过索引修改更干净高效）
                        edge_limits = [item for item in edge_limits if not
                        ((item[0] == m and item[1] == target_node) or
                         (item[0] == target_node and item[1] == m))]

                        # 添加正向和反向的时间窗限制
                        edge_limits.append((m, target_node, start_limit, end_limit))
                        edge_limits.append((target_node, m, start_limit, end_limit))

                    else:
                        added_count = 0
                        max_retries = 20
                        current_retries = 0

                        while added_count < num:
                            if current_retries > max_retries:
                                break

                            start_limit = random.randint(0, 23)
                            end_limit = (start_limit + 1 + random.uniform(0, 12)) % 24

                            # 只针对同一条边检查时间窗是否重叠，使用 any() 提升效率
                            collision = any(
                                is_overlap(start_limit, end_limit, item[2], item[3])
                                for item in edge_limits
                                if item[0] == m and item[1] == target_node
                            )

                            # 如果没有冲突，则添加
                            if not collision:
                                edge_limits.append((m, target_node, start_limit, end_limit))
                                added_count += 1
                                current_retries = 0
                            else:
                                current_retries += 1
                j += 1

        return edge_limits[:]

    def one_step(self, action, Q=True):
        reward = 0.0
        done = False
        violation = [0, 0]
        # waste = self.map[self.agent_pos][action][1] / 60
        waste_time = self.map[self.agent_pos][action][2] / 60

        reward -= self.map[self.agent_pos][action][1] / 20

        if self.map[self.agent_pos][action][0] != -1:
            if is_edge_active(self.edge_limits, self.agent_time, self.agent_pos, self.map[self.agent_pos][action][0]):
                reward -= 10
                violation[1] += 1
            self.agent_pos = self.map[self.agent_pos][action][0]
        else:
            reward -= 10
            violation[0] += 1

        self.agent_time = (self.agent_time + waste_time) % 24

        if self.agent_pos in self.goal_states:
            reward += 5  # 到达一个新的目标地点之后给予奖励
            self.goal_states.remove(self.agent_pos)
        if len(self.goal_states) == 0 and self.agent_pos == self.start_pos:
            reward += 10  # 到达全部目标地点时给予大奖励
            done = True

        if Q:
            return self.get_state()[0], reward, done, violation
        else:
            return None, reward, done, violation

    def N_one_step(self, action, dry_run=False):
        """
        执行单步动作，返回距离、风险、是否完成及约束违反情况

        参数:
            action: 要执行的动作
            dry_run: 若为True，仅检查是否到达终点，不实际更新环境状态（用于路径生成时的提前判断）
        """
        distance = 0
        done = False
        violation = [0, 0]
        current_pos = self.agent_pos  # 记录当前位置（用于dry_run模式）
        current_time = self.agent_time  # 记录当前时间（用于dry_run模式）

        if self.map[current_pos][action][0] != -1:
            # 检查边是否激活（若激活则增加风险和约束违反）
            if is_edge_active(self.edge_limits, current_time, current_pos, self.map[current_pos][action][0]):
                distance += 1000
                violation[1] += 1
            # 累加距离和风险
            distance += self.map[current_pos][action][1]
            # 计算时间消耗（转换为小时）
            waste = self.map[current_pos][action][2] / 60
            new_time = (current_time + waste) % 24
            # 计算新位置
            new_pos = self.map[current_pos][action][0]
        else:
            # 无效动作：增加距离惩罚和约束违反
            distance += 1000
            violation[0] += 1
            new_pos = current_pos  # 位置不变
            new_time = current_time  # 时间不变

        # 检查是否到达目标（完成所有目标则done=True）
        temp_goals = self.goal_states.copy()  # 临时目标列表（用于dry_run判断）
        if new_pos in temp_goals:
            temp_goals.remove(new_pos)
            if len(temp_goals) == 0:
                done = True

        # 若不是dry_run模式，才实际更新环境状态
        if not dry_run:
            self.agent_pos = new_pos
            self.agent_time = new_time
            # 更新目标状态（移除已到达的目标）
            if new_pos in self.goal_states:
                self.goal_states.remove(new_pos)

        return distance, done, violation

    def edges(self):
        edges = []
        edges_attr = []
        for x in range(self.space_size):
            for y in range(4):
                if self.map[x][y][0] != -1:
                    indices = [i for i, item in enumerate(self.edge_limits)
                               if item[0] == x and item[1] == self.map[x][y][0]]
                    indices.append(-1)
                    for i in indices:
                        a = [0, 0]
                        done = False
                        case = None
                        if len(indices) > 1:
                            if i == -1:
                                break
                            case = self.edge_limits[i]
                            done = True
                        edges.append([self.map[x][y][0], x])
                        if done:
                            a[0] = (case[2] - self.agent_time) % 24
                            a[1] = (case[3] - self.agent_time) % 24
                            a1 = time_to_multi_sincos(a[0])
                            a2 = time_to_multi_sincos(a[1])
                        else:
                            a1 = [-2 for _ in range(8)]
                            a2 = [-2 for _ in range(8)]
                        if y == 0:
                            edges_attr.append([1, 0, 0, 0, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a2[0], a2[1],a2[2], a2[3], a2[4], a2[5], a2[6], a2[7], self.map[x][y][1] / 50,self.map[x][y][2] / 50])
                        elif y == 1:
                            edges_attr.append([0, 1, 0, 0, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a2[0], a2[1],a2[2], a2[3], a2[4], a2[5], a2[6], a2[7], self.map[x][y][1] / 50,self.map[x][y][2] / 50])
                        elif y == 2:
                            edges_attr.append([0, 0, 1, 0, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a2[0], a2[1],a2[2], a2[3], a2[4], a2[5], a2[6], a2[7], self.map[x][y][1] / 50,self.map[x][y][2] / 50])
                        else:
                            edges_attr.append([0, 0, 0, 1, a1[0], a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a2[0], a2[1],a2[2], a2[3], a2[4], a2[5], a2[6], a2[7], self.map[x][y][1] / 50,self.map[x][y][2] / 50])
        # edges_attr = min_max_normalize(edges_attr)
        return edges, edges_attr

    def get_state(self):
        nodes = []
        # 假设我们用 4 个维度来表示位置编码
        # 你可以自己定义任意周期，想多少个就多少个
        periods = [300]  # 显式周期列表

        for x in range(self.space_size):
            a = [1.0 if x == self.start_pos else 0.0]
            b = [1.0 if x in self.goal_states else 0.0]

            p = []

            for T in periods:
                radians = 2 * math.pi * x / T
                p.append(math.sin(radians))
                p.append(math.cos(radians))

            nodes.append(a + b + p)

        # 构建边：相邻节点连接
        # edges = []
        self.edges_index, self.edges_attr = self.edges()
        edges = copy.deepcopy(self.edges_index)
        edges_attr = copy.deepcopy(self.edges_attr)

        data_dict = [torch.tensor(nodes, dtype=torch.float32), torch.tensor(edges, dtype=torch.int64).t(),
                     torch.tensor(edges_attr, dtype=torch.float32), torch.tensor([self.agent_pos], dtype=torch.int)]

        return data_dict,self.space_size

    def shuffle_space(self):
        for row in self.map:
            random.shuffle(row)
