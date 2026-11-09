from ENV import *
random.seed(42)
np.random.seed(42)
from Gragh import *
import time as T
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from typing import List, Tuple, Optional, Dict, Sequence
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




class PathSolution:
    def __init__(self, actions, env: 'Sample_Env',start_path=None,start_pos=0,start_time=0.0):
        """
        Initialize a path solution.

        Args:
            actions: action sequence (e.g. [0, 2, 1, ...])
            env: path planning environment
            start_time: departure time
        """
        if actions is None:
            self.actions = path_to_actions(start_path, env.map)
        else:
            self.actions = actions
        self.env = env
        self.violation = [0,0]
        self.start_pos = start_pos
        self.start_time = start_time
        self.path = self._build_path()
        self.fitness = None  # fitness (distance, risk)
        self.reward = None
        self.Pass = None

    def _build_path(self) -> List[Tuple[int, float]]:
        """Build the path from the action sequence; returns (position, arrival time) tuples."""
        # Reset the environment to its initial state
        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)

        path = [(self.env.agent_pos, self.env.agent_time)]

        for action in self.actions:
            _, _, violation = self.env.N_one_step(action)
            path.append((self.env.agent_pos, self.env.agent_time))

        return path

    def evaluate(self):
        """Evaluate path fitness (distance and risk)."""
        if not self.path:
            self.fitness = float('inf')
            return self.fitness

        # Accumulate the distance along the action sequence
        done = False
        distance = 0
        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)
        for i in range(len(self.actions)):
            a, c, d = self.env.N_one_step(self.actions[i])
            distance += a
            done = c
            if done:
                if i < len(self.actions) - 1:
                    self.actions = self.actions[:i + 1]
                break
            self.violation[0] += d[0]
            self.violation[1] += d[1]

        if not done:
            distance += 10000
            self.Pass = False
        self.fitness = distance

        if self.violation[0] == 0 and self.violation[1] == 0:
            self.Pass = True
        else:
            self.Pass = False

        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)
        reward = 0
        for i in range(len(self.actions)):
            _, r, _, _ = self.env.one_step(self.actions[i])
            reward += r
        self.reward = reward
        return self.fitness

def map_to_graph(map_data, special_nodes=None, special_color='yellow',
                 path_sequence=None, path_color='red', time_window_edges=None):
    # Initialization and graph construction
    if special_nodes is None: special_nodes = []
    special_nodes = list(set(special_nodes))
    all_node_ids = set(range(len(map_data)))
    special_nodes = [n for n in special_nodes if n in all_node_ids]
    if path_sequence is None: path_sequence = []
    if time_window_edges is None: time_window_edges = []

    edges_with_weight = []
    for from_node, connections in enumerate(map_data):
        for conn in connections:
            to_node, weight, _ = conn
            if to_node != -1:
                edges_with_weight.append((from_node, to_node, weight))

    valid_tw_edges = []
    original_edges = set((u, v) for u, v, w in edges_with_weight)
    for tw_edge in time_window_edges:
        if len(tw_edge) != 4: continue
        u_tw, v_tw, t_start, t_end = tw_edge
        if (u_tw in all_node_ids and v_tw in all_node_ids and (u_tw, v_tw) in original_edges):
            valid_tw_edges.append(tw_edge)

    G = nx.DiGraph()
    G.add_nodes_from(all_node_ids)
    G.add_weighted_edges_from(edges_with_weight)

    # Compute the layout with a reduced visual weight gap
    G_layout = G.copy()
    all_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    if not all_weights: all_weights = [1.0]
    min_w, max_w = min(all_weights), max(all_weights)
    VISUAL_GAP = 0.5
    for u, v, data in G_layout.edges(data=True):
        real_weight = data['weight']
        if max_w == min_w:
            visual_weight = 1.0
        else:
            normalized = (real_weight - min_w) / (max_w - min_w)
            visual_weight = 1.0 + (normalized * VISUAL_GAP)
        data['weight'] = visual_weight

    pos = nx.kamada_kawai_layout(G_layout, weight='weight', scale=20)

    # Plot the graph
    plt.figure(figsize=(20, 16), dpi=100)
    NODE_SIZE = 5000

    # Draw nodes
    special_node_set = set(special_nodes)
    path_start = path_sequence[0] if path_sequence else None
    normal_nodes = [n for n in all_node_ids if n not in special_node_set and n != path_start]
    special_only_nodes = [n for n in special_nodes if n != path_start]

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=NODE_SIZE,
                           node_color='skyblue', edgecolors='navy', linewidths=2)
    if special_only_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=special_only_nodes, node_size=NODE_SIZE,
                               node_color=special_color, edgecolors='black', linewidths=3, alpha=0.9)
    if path_start is not None and path_start in all_node_ids:
        nx.draw_networkx_nodes(G, pos, nodelist=[path_start], node_size=NODE_SIZE,
                               node_color='red', edgecolors='darkred', linewidths=3, alpha=0.95)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(u, v) for u, v, w in edges_with_weight],
                           width=3,
                           edge_color=mcolors.to_rgba('darkgray', 0.5),
                           arrowstyle='-|>',  # filled arrowhead for clarity
                           arrowsize=20,
                           node_size=NODE_SIZE)  # align arrow size with node size

    # Draw highlighted path edges
    if len(path_sequence) >= 2:
        # Locate the index at which all special nodes have been visited
        target_specials = set(special_nodes)  # targets that must be visited
        visited_specials = set()  # targets visited so far
        split_index = len(path_sequence)  # default: the whole path is phase 1

        # Scan the path until every target has been visited
        for i, node in enumerate(path_sequence):
            if node in target_specials:
                visited_specials.add(node)

            # The transition point is where the visited set first covers all targets
            if target_specials and target_specials.issubset(visited_specials):
                split_index = i
                break

        # Split the path into a task segment and a return segment
        phase1_edges = []  # edges before all targets are reached (straight, path_color)
        phase2_edges = []  # edges after all targets are reached (curved, return_color)

        # Color of the return segment
        return_color = 'green'  # any matplotlib color name

        for i in range(len(path_sequence) - 1):
            u = path_sequence[i]
            v = path_sequence[i + 1]

            # Draw only edges that exist in the map
            if (u, v) in original_edges:
                # Edges before split_index belong to the task segment
                if i < split_index:
                    phase1_edges.append((u, v))
                else:
                    phase2_edges.append((u, v))

        # Task segment: straight, aligned with the base road network
        if phase1_edges:
            nx.draw_networkx_edges(G, pos,
                                   edgelist=phase1_edges,
                                   width=6,
                                   edge_color=path_color,  # default red
                                   arrowstyle='-|>',
                                   arrowsize=30,
                                   alpha=0.8,
                                   node_size=NODE_SIZE)

        # Return segment: curved so that it does not overlap the task segment
        if phase2_edges:
            nx.draw_networkx_edges(G, pos,
                                   edgelist=phase2_edges,
                                   width=5,  # slightly thinner to distinguish
                                   edge_color=return_color,
                                   arrowstyle='-|>',
                                   arrowsize=30,
                                   alpha=0.9,
                                   node_size=NODE_SIZE,
                                   # rad sets the curvature; its sign sets the bending direction
                                   connectionstyle='arc3, rad=0.2',
                                   style='dashed')  # dashed style distinguishes the return segment

    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold', font_color='navy')

    # Draw weight and time-window labels without overlap

    # Point at a given ratio along the segment
    def get_pos_at_ratio(pos, u, v, ratio=0.5):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        return (x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio)

    # Group directed edges by unordered node pair so both directions share one label
    # Structure: edge_pair_data[(min_id, max_id)] = {'weights': {}, 'tws': []}
    edge_pair_data = {}

    # Collect weights
    for u, v, w in edges_with_weight:
        pair_key = tuple(sorted((u, v)))
        if pair_key not in edge_pair_data:
            edge_pair_data[pair_key] = {'weights': {}, 'tws': []}
        edge_pair_data[pair_key]['weights'][(u, v)] = w

    # Collect time windows
    for u, v, t_start, t_end in valid_tw_edges:
        pair_key = tuple(sorted((u, v)))
        # Guard against a time window whose edge has no weight entry
        if pair_key in edge_pair_data:
            edge_pair_data[pair_key]['tws'].append((u, v, t_start, t_end))

    # Draw the merged labels
    for (u, v), data in edge_pair_data.items():
        # u, v are sorted IDs; the ratio position does not depend on direction for
        # a straight segment

        # Weight label at 35% of the segment
        weights = data['weights']
        label_text = ""

        # One direction only
        if len(weights) == 1:
            val = list(weights.values())[0]
            label_text = f"{val}"
        # Both directions
        else:
            w_uv = weights.get((u, v), "?")  # lookup by the actual direction
            w_vu = weights.get((v, u), "?")

            # identical weights: print a single value
            if w_uv == w_vu:
                label_text = f"{w_uv}"
            # otherwise print "forward / backward"
            else:
                label_text = f"{w_uv} / {w_vu}"

        w_x, w_y = get_pos_at_ratio(pos, u, v, ratio=0.35)  # towards the source end
        plt.text(w_x, w_y, label_text, fontsize=18, color='darkred',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
                 ha='center', va='center', zorder=10)

        # Time-window label at 65% of the segment
        tws = data['tws']
        if tws:
            tw_texts = []
            seen_tw_values = set()  # already emitted (start, end) pairs

            for (curr_u, curr_v, t_s, t_e) in tws:
                # key used to deduplicate identical windows
                tw_val_key = (t_s, t_e)

                # emit each distinct window once
                if tw_val_key not in seen_tw_values:
                    def convert_time(t):
                        hour = int(t)
                        minute = int((t - hour) * 60)
                        return f"{hour}:{minute:02d}"  # zero-padded minute: 6.5 -> 6:30

                    tw_texts.append(f"[{convert_time(t_s)}, {convert_time(t_e)}]")
                    seen_tw_values.add(tw_val_key)

            # one line per distinct time window
            tw_final_label = "\n".join(tw_texts)

            tw_x, tw_y = get_pos_at_ratio(pos, u, v, ratio=0.65)  # towards the target end
            plt.text(tw_x, tw_y, tw_final_label, fontsize=16, color='blue',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5, edgecolor='none'),
                     ha='center', va='center', zorder=10)

    # Save and display
    plt.rcParams['font.family'] = 'Times New Roman'  # figure font
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # fallback for CJK labels
    plt.rcParams['axes.unicode_minus'] = False  # render the minus sign correctly

    title = 'start_time: 0.00'
    plt.title(title, fontsize=35, fontweight='bold', pad=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('MY.svg', format='svg', bbox_inches='tight')
    plt.show()

def get_env_adj(env):
    """
    Build an adjacency list from the environment without mutating it.

    adj[i] = [(j, t_ij, d)]: node j with travel time t_ij, reached from node i
        in direction d.
    dir_map[i][j] = d: direction d that moves from node i to node j, so that
        one_step can be called later.
    """
    adj = [[] for _ in range(env.space_size)]
    dir_map = [dict() for _ in range(env.space_size)]
    for i in range(env.space_size):
        for d in range(env.action_size):
            j = env.map[i][d][0]
            t_ij = env.map[i][d][1]
            if j != -1:  # keep feasible edges only (target node != -1)
                adj[i].append((j, t_ij, d))
                dir_map[i][j] = d
    return adj, dir_map

class ACO_ShortestTime:
    def __init__(self, env, ant_num=50, max_iter=100, beta=2.0, gamma=3.0, rho1=0.1, rho2=0.2, alpha=0.05,start_pos=0,start_time=0.0):
        """
        Initialize the ant colony algorithm.

        Args:
            env: Sample_Env instance, accessed only through its public methods
            ant_num: number of ants per iteration
            max_iter: maximum number of iterations
            beta: pheromone weight (higher values rely more on history)
            gamma: heuristic weight (higher values prefer faster edges)
            rho1: local pheromone evaporation rate (0 < rho1 < 1)
            rho2: global pheromone evaporation rate (0 < rho2 < 1)
            alpha: penalty factor for time-constrained edges (smaller values
                avoid them more strongly)
            start_pos: ant start node
            start_time: ant start time
        """
        self.env = env  # environment instance (read-only)
        self.ant_num = ant_num
        self.max_iter = max_iter
        self.beta = beta
        self.gamma = gamma
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.start_pos = start_pos  # start node (defaults to env.origin)
        self.start_time = start_time
        self.goal_num = len(env.goal_states1)  # number of targets
        self.nodes_num = env.space_size  # number of nodes

        # Adjacency list is built once, outside the environment
        self.adj, self.dir_map = get_env_adj(env)

        # Pheromone initialization: tau0 is derived from the mean travel time
        avg_t = 0.0
        edge_count = 0
        for i in range(self.nodes_num):
            for (j, t_ij, d) in self.adj[i]:
                avg_t += t_ij
                edge_count += 1
        avg_t /= edge_count if edge_count > 0 else 1
        self.tau0 = 1.0 / (self.ant_num * avg_t)  # initial pheromone level
        self.tau = np.ones((self.nodes_num, self.nodes_num)) * 1e-6  # small positive floor
        for i in range(self.nodes_num):
            for (j, t_ij, d) in self.adj[i]:
                self.tau[i][j] = self.tau0  # feasible edges start at tau0

        # Best solution found so far
        self.best_reward = -np.inf  # highest reward (shortest travel time)
        self.best_weight = np.inf     # shortest travel time
        self.best_path = []         # best node sequence


    def _reset_ant_env(self, ant_start_pos=0, ant_start_time=0.0):
        """
        Reset the environment before a single ant explores.

        Args:
            ant_start_pos: ant start node
            ant_start_time: ant start time (24-hour clock)
        """
        self.env.reset_env()          # restore the full target list (goal_states1)
        self.env.reset_agent_pos(ant_start_pos)  # reset ant position
        self.env.reset_agent_time(ant_start_time)# reset ant clock

    def _select_next_action(self, current_pos, current_time):
        """
        Select the next action, guiding the ant home once all targets are visited.
        """
        feasible_actions = []  # feasible (direction, target node, travel time)
        for (j, t_ij, d) in self.adj[current_pos]:
            feasible_actions.append((d, j, t_ij))
        if not feasible_actions:
            return None

        # All targets visited: the only remaining task is returning to the start
        returning_home_mode = (len(self.env.goal_states) == 0)

        prob_list = []
        for (d, j, t_ij) in feasible_actions:
            # heuristic: shorter edges are preferred
            eta = 1.0 / (t_ij + 1e-6)

            # Strong homing guidance: prefer an edge straight back to the start
            if returning_home_mode and j == self.start_pos:
                eta *= 100.0  # large boost so this edge is almost always chosen

            # time-window penalty
            if is_edge_active(self.env.edge_limits, current_time, current_pos, j):
                eta *= self.alpha

            # pheromone term
            tau_term = (self.tau[current_pos][j]) ** self.beta

            # combined selection weight
            prob_list.append(tau_term * (eta ** self.gamma))

        # roulette-wheel selection
        sum_prob = sum(prob_list)
        if sum_prob < 1e-6:
            selected_idx = random.randint(0, len(feasible_actions) - 1)
        else:
            prob_list = [p / sum_prob for p in prob_list]
            selected_idx = np.random.choice(len(feasible_actions), p=prob_list)

        return feasible_actions[selected_idx][0]

    def _ant_explore(self):
        """
        Explore with a step limit that prevents infinite loops.
        """
        path = [self.env.agent_pos]
        total_time = 0.0
        total_reward = 0.0
        done = False

        # Safety valve against loops: three times the node count is enough to
        # visit every target and return; otherwise the run counts as failed
        max_steps = self.nodes_num * 3
        steps = 0

        while not done and steps < max_steps:
            current_pos = self.env.agent_pos
            current_time = self.env.agent_time

            next_action = self._select_next_action(current_pos, current_time)

            if next_action is None:
                break  # dead end

            # Sample_Env step semantics:
            # 1. reaching a target removes it and adds 5 to the reward
            # 2. with no targets left, returning to start_pos adds 10 and sets done
            _, reward, done, _ = self.env.one_step(next_action, Q=False)

            total_reward += reward
            t_ij = self.env.map[current_pos][next_action][1]
            total_time += t_ij
            path.append(self.env.agent_pos)

            # local pheromone update
            next_pos = self.env.agent_pos
            self.tau[current_pos][next_pos] = (1 - self.rho1) * self.tau[current_pos][next_pos] + self.rho1 * self.tau0

            steps += 1

        # A step-limited run is incomplete and must not update the global pheromone
        if not done:
            return path, total_time, -np.inf, False  # minimal reward penalty

        return path, total_time, total_reward, done


    def _global_tau_update(self, optimal_path):
        """
        Reinforce pheromone on a path (once per iteration).

        Args:
            optimal_path: node sequence to reinforce
        """
        for k in range(len(optimal_path) - 1):
            i = optimal_path[k]
            j = optimal_path[k + 1]
            # tau = (1 - rho2) * tau + rho2 * best_reward
            self.tau[i][j] = (1 - self.rho2) * self.tau[i][j] + self.rho2 * self.best_reward


    def run(self):
        """
        Run the ant colony algorithm.

        Returns:
            best path, shortest travel time, highest reward
        """
        # Seeding is left to the caller so that each instance is reproducible
        for iter in range(self.max_iter):
            valid_ants = []  # ants that completed all targets

            # Each ant explores independently from a freshly reset environment
            for _ in range(self.ant_num):
                # every ant starts at start_pos/start_time with the full target list
                self._reset_ant_env(ant_start_pos=self.start_pos, ant_start_time=self.start_time)
                path, total_time, total_reward, done = self._ant_explore()
                if done:  # keep completed runs only
                    valid_ants.append((path, total_time, total_reward))

            # Best ant of this iteration; higher reward means shorter travel time
            if valid_ants:
                valid_ants.sort(key=lambda x: x[2], reverse=True)
                current_best_path, current_best_weight, current_best_reward = valid_ants[0]

                # update the global best
                if current_best_reward > self.best_reward:
                    self.best_reward = current_best_reward
                    self.best_weight = current_best_weight
                    self.best_path = current_best_path

                # reinforce pheromone on this iteration's best path
                self._global_tau_update(current_best_path)

            # progress
            if (iter + 1) % 10 == 0:
                print(f"迭代{iter+1:3d} | 全局最优时间：{self.best_weight:.2f} | 全局最优Reward：{self.best_reward:.2f}")

        # final report
        print("\n" + "="*60)
        print("蚁群算法求解完成（完全基于原Sample_Env）")
        print(f"1. 最优路径（节点序列）：{self.best_path}")
        print(f"2. 最短时间消耗：{self.best_weight:.2f}（单位与原环境一致）")
        print(f"3. 最大累计Reward：{self.best_reward:.2f}")
        # count time-constrained edges on the best path
        limit_edge_count = 0
        for k in range(len(self.best_path)-1):
            i = self.best_path[k]
            j = self.best_path[k+1]
            # elapsed time before this step
            step_time = 0.0
            for m in range(k):
                mi = self.best_path[m]
                mj = self.best_path[m+1]
                md = self.dir_map[mi][mj]
                step_time += self.env.map[mi][md][1] / 60.0  # minutes to hours
            current_time = (0.0 + step_time) % 24.0  # start at 0.0 h and wrap at 24 h
            if is_edge_active(self.env.edge_limits, current_time, i, j):
                limit_edge_count += 1
        print(f"4. 最优路径中时间约束边数量：{limit_edge_count}（共{len(self.best_path)-1}条边）")
        print("="*60)
        return self.best_path, self.best_weight, self.best_reward

if __name__ == "__main__":
    env_seed = 9
    random.seed(env_seed)
    np.random.seed(env_seed)

    n = 300
    G1 = generate_directed_graph(n)
    map, node_size, edge_size = graph_to_dict(G1)
    target_num = int((node_size*8) / 25)
    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(10, 50) for _ in range(edge_size)]
    times = [random.randint(10, 50) for _ in range(edge_size)]
    env = Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes, times=times,
                     same=True, one_same=False)


    print("初始化原Sample_Env完成：")
    print(f"- 节点数：{env.space_size}")
    print(f"- 边数：{env.edge_size}")
    print(f"- 起点：{env.origin}")
    print(f"- 目标点：{env.goal_states1}")
    print(f"- 时间约束边数量：{len(env.edge_limits)}")
    print()

    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)
    # initialize the ant colony algorithm
    aco = ACO_ShortestTime(
        env=env,
        ant_num=50,  # ants per iteration
        max_iter=100,  # iterations
        beta=2.0,  # pheromone weight
        gamma=3.0,  # heuristic (travel time) weight
        rho1=0.1,  # local evaporation rate
        rho2=0.2,  # global evaporation rate
        alpha=0.05,  # penalty for time-constrained edges
        start_pos=0,  # ant start node
        start_time=0.0,  # ant start time
    )
    start_time = T.time()
    best_path, best_weight, best_reward = aco.run()
    print(f"运行时间：{T.time()-start_time:.2f}s")
    solution = PathSolution(None, env, best_path)
    solution.evaluate()
