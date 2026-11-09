import torch
import numpy as np
import copy
import random
from collections import Counter
import math

def normal_int_sample(mean, std, min_val, max_val):
    """
    Draw an integer from a normal distribution, clipped to [min_val, max_val].

    Args:
        mean: mean of the normal distribution
        std: standard deviation (spread of the distribution)
        min_val: lower bound
        max_val: upper bound
    Returns:
        an integer within the bounds
    """
    # Cast mean and std to float
    mean = float(mean)
    std = float(std)
    # Cast the bounds to int
    min_val = int(min_val)
    max_val = int(max_val)

    # Resample until the rounded value falls in range
    while True:
        num = random.gauss(mean, std)
        int_num = round(num)
        if min_val <= int_num <= max_val:
            return int_num

def min_max_normalize(edge_attr):
    # Transpose so that each feature is processed along its own column
    features = list(zip(*edge_attr))
    normalized = []

    for i, feature in enumerate(features):
        # Keep the first ten columns unscaled
        if 0 <= i <= 9:
            normalized_feature = list(feature)  # keep the raw values
        else:
            min_val = min(feature)
            max_val = max(feature)
            scale = max_val - min_val
            # Guard against division by zero
            if scale == 0:
                normalized_feature = [0.0] * len(feature)
            else:
                normalized_feature = [(x - min_val) / scale for x in feature]
        normalized.append(normalized_feature)

    # Restore the original shape
    return list(zip(*normalized))


def time_to_multi_sincos(time_val, periods=None):
    """
    Multi-frequency sine/cosine encoding on a minute scale: hours are converted to
    minutes first, then encoded against several periods.

    Args:
    time_val -- time in hours (e.g. 1.5 means 1.5 h, 23.9 means 23.9 h)
    periods -- list of periods (default: four minute-scale periods, 30 min, 2 h, 8 h, 1 day)

    Returns:
    concatenated sin/cos features (length 8, two features per period)
    """
    # Convert hours to minutes (1 h = 60 min)
    minute_val = time_val * 60  # e.g. 1.5 h -> 90 min

    if periods is None:
        # Minute-scale periods (unit: minutes)
        periods = [30, 120, 480, 1440]  # 30 min, 2 h, 8 h, 1 day

    features = []
    for T in periods:
        # Angle in radians from the minute-scale time and period
        radians = 2 * np.pi * minute_val / T
        features.append(np.sin(radians))
        features.append(np.cos(radians))

    return np.array(features)


def is_within_time_period(time_limit_states, agent_time, agent_pos):
    for (i, lower_bound, upper_bound) in time_limit_states:
        # Position must match
        if i != agent_pos:
            continue

        # Ordinary window (lower_bound < upper_bound), closed interval [lower, upper]
        if lower_bound < upper_bound:
            if lower_bound <= agent_time <= upper_bound:
                return True
        # Window crossing midnight (lower_bound > upper_bound), also closed
        else:
            # Time at or after lower_bound, or at or before upper_bound
            if agent_time >= lower_bound or agent_time <= upper_bound:
                return True

    return False


def is_edge_active(edges, time_point, a, b):
    """
    Decide whether the arc a -> b exists in the edge list and is usable at time_point.

    Args:
    edges -- edge list, entries are (start_node, end_node, start_time, end_time)
    a -- start node
    b -- end node
    time_point -- time to test

    Returns:
    bool -- True if such an arc exists and admits time_point, else False
    """
    for (start_node, end_node, lower_bound, upper_bound) in edges:
        if a != start_node or b != end_node:
            continue
        if lower_bound < upper_bound:
            if lower_bound <= time_point <= upper_bound:
                return True
        # Window crossing midnight (lower_bound > upper_bound), also closed
        else:
            # Time at or after lower_bound, or at or before upper_bound
            if time_point >= lower_bound or time_point <= upper_bound:
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
    # Convert time windows to linear interval lists
    # s < e: the window is [(s, e)]
    # s > e (crossing midnight): the window is [(s, 24), (0, e)]
    def get_intervals(s, e):
        return [(s, e)] if s < e else [(s, 24.0), (0.0, e)]

    intervals1 = get_intervals(s1, e1)
    intervals2 = get_intervals(s2, e2)

    # Compare every pair of intervals for an intersection
    for a_start, a_end in intervals1:
        for b_start, b_end in intervals2:
            # Two closed intervals overlap iff max(start) <= min(end); touching counts
            if max(a_start, b_start) <= min(a_end, b_end):
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

        # Return a copy when limits are already provided
        if edge_limits is not None:
            return edge_limits[:]

        edge_limits = []

        # Draw the indices of the restricted arcs
        if one_same:
            random_numbers = random.sample(range(self.edge_size), 12)
        else:
            # random.sample guarantees unique indices, so every count is 0 or 1;
            # use random.choices if multiplicities above 1 are needed.
            if k_num == None:
                k_val = normal_int_sample(self.edge_size / 4, self.edge_size / 8, 16, self.edge_size / 3)
            else:
                k_val = k_num
            random_numbers = random.sample(range(self.edge_size), k_val)

        # Precompute counts for O(1) lookups
        number_counts = Counter(random_numbers)

        j = 0
        for m in range(self.space_size):
            for n in range(self.action_size):
                target_node = self.map[m][n][0]

                # Skip invalid arcs
                if target_node == -1:
                    continue

                num = number_counts.get(j, 0)

                if num > 0:
                    if one_same:
                        start_limit = random.randint(0, 23)
                        end_limit = (start_limit + 1 + random.uniform(0, 12)) % 24

                        # Drop any existing forward/backward window for this pair
                        edge_limits = [item for item in edge_limits if not
                        ((item[0] == m and item[1] == target_node) or
                         (item[0] == target_node and item[1] == m))]

                        # Apply the same window in both directions
                        edge_limits.append((m, target_node, start_limit, end_limit))
                        edge_limits.append((target_node, m, start_limit, end_limit))

                    else:
                        start_limit = random.randint(0, 23)
                        end_limit = (start_limit + 1 + random.uniform(0, 12)) % 24
                        edge_limits.append((m, target_node, start_limit, end_limit))
                j += 1

        return edge_limits[:]

    def get_action_mask(self):
        """
        Return the action mask (boolean list of length action_size):
        - True  means the action is available
        - False means the action is forbidden

        The forbidden conditions match the penalties of one_step / N_one_step:
        1. no road in that direction (map[agent_pos][a][0] == -1)
        2. the arc is currently inside its time window (is_edge_active is True,
           i.e. using it would violate the time-window restriction)
        """
        mask = []
        for a in range(self.action_size):
            next_node = self.map[self.agent_pos][a][0]
            if next_node == -1:
                mask.append(False)
            elif is_edge_active(self.edge_limits, self.agent_time, self.agent_pos, next_node):
                mask.append(False)
            else:
                mask.append(True)
        return mask

    def one_step(self, action, Q=True):
        reward = 0.0
        done = False
        violation = [0, 0]
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
            reward += 5  # reward for reaching a new target
            self.goal_states.remove(self.agent_pos)
        if len(self.goal_states) == 0 and self.agent_pos == self.start_pos:
            reward += 10  # large bonus once all targets are visited and the agent is home
            done = True

        if Q:
            return self.get_state()[0], reward, done, violation
        else:
            return None, reward, done, violation

    def N_one_step(self, action, dry_run=False):
        """
        Execute one action; returns distance, completion flag and constraint violations.

        Args:
            action: action to execute
            dry_run: if True, only check whether the target set would be completed
                     without updating the environment (used for path generation lookahead)
        """
        distance = 0
        done = False
        violation = [0, 0]
        current_pos = self.agent_pos  # current position (used by dry_run)
        current_time = self.agent_time  # current time (used by dry_run)

        if self.map[current_pos][action][0] != -1:
            # Active window: add the large penalty and count the violation
            if is_edge_active(self.edge_limits, current_time, current_pos, self.map[current_pos][action][0]):
                distance += 1000
                violation[1] += 1
            # Accumulate the arc length
            distance += self.map[current_pos][action][1]
            # Time consumed by the arc, in hours
            waste = self.map[current_pos][action][2] / 60
            new_time = (current_time + waste) % 24
            # New position
            new_pos = self.map[current_pos][action][0]
        else:
            # Invalid action: distance penalty and violation
            distance += 1000
            violation[0] += 1
            new_pos = current_pos  # position unchanged
            new_time = current_time  # time unchanged

        # Reaching the last target finishes the episode
        temp_goals = self.goal_states.copy()  # temporary copy of the target list
        if new_pos in temp_goals:
            temp_goals.remove(new_pos)
            if len(temp_goals) == 0:
                done = True

        # Update the environment only outside dry_run
        if not dry_run:
            self.agent_pos = new_pos
            self.agent_time = new_time
            # Remove the reached target
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
        return edges, edges_attr

    def get_state(self):
        nodes = []
        periods = [300]

        for x in range(self.space_size):
            a = [1.0 if x == self.start_pos else 0.0]
            b = [1.0 if x in self.goal_states else 0.0]

            p = []

            for T in periods:
                radians = 2 * math.pi * x / T
                p.append(math.sin(radians))
                p.append(math.cos(radians))

            nodes.append(a + b)

        # Edge indices and edge features of the current graph
        self.edges_index, self.edges_attr = self.edges()
        edges = copy.deepcopy(self.edges_index)
        edges_attr = copy.deepcopy(self.edges_attr)

        data_dict = [torch.tensor(nodes, dtype=torch.float32), torch.tensor(edges, dtype=torch.int64).t(),
                     torch.tensor(edges_attr, dtype=torch.float32), torch.tensor([self.agent_pos], dtype=torch.int)]

        return data_dict,self.space_size

    def shuffle_space(self):
        for row in self.map:
            random.shuffle(row)
