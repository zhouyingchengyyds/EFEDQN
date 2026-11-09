from ENV import *
from typing import List, Tuple, Optional, Dict, Sequence
from Gragh import *
import time as T
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from ASO import PathSolution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def map_to_graph(map_data, special_nodes=None, special_color='yellow',
                 path_sequence=None, path_color='red', time_window_edges=None):
    """
    Draw a directed graph whose visual edge lengths reflect the actual weights,
    and save it as SVG.

    Features:
    1. Apply a common special color to a given list of nodes.
    2. Draw a specified route from a node sequence.
    3. Accept time-constrained edges and label their time windows
       (format: [(from_node, to_node, start, end), ...]).

    Args:
        map_data: list[list[list]]
            Node connectivity, structured as
            [from_node[action[to_node, travel time, unused], ...], ...].
            A to_node of -1 means the edge is not traversable.

        special_nodes: list, optional
            Nodes drawn with the special color, e.g. [0, 5, 10].

        special_color: str, optional
            Color of the special nodes, default 'yellow'.

        path_sequence: list, optional
            Node sequence (route), e.g. [0, 3, 5, 8].

        path_color: str, optional
            Color of the route, default 'red'.

        time_window_edges: list of tuples, optional
            Time-constrained edges, format
            [(from_node, to_node, start, end), ...],
            e.g. [(2, 7, 7.00, 9.00), (5, 8, 6.00, 8.00)].
    """
    # Default arguments
    # Deduplicate special nodes and drop invalid IDs
    if special_nodes is None:
        special_nodes = []
    special_nodes = list(set(special_nodes))
    all_node_ids = set(range(len(map_data)))  # all valid node IDs
    special_nodes = [node for node in special_nodes if node in all_node_ids]

    # Default path sequence
    if path_sequence is None:
        path_sequence = []

    # Default time-window edges
    if time_window_edges is None:
        time_window_edges = []

    # Extract feasible weighted edges; defined before use below
    edges_with_weight = []  # (from_node, to_node, travel time)
    for from_node, connections in enumerate(map_data):
        for conn in connections:
            to_node, weight, _ = conn
            if to_node != -1:  # keep traversable edges only (to_node != -1)
                edges_with_weight.append((from_node, to_node, weight))

    # Filter time-window edges; requires edges_with_weight above
    valid_tw_edges = []  # valid time-window edges
    original_edges = [(u, v) for u, v, w in edges_with_weight]  # feasible edges without weights
    for tw_edge in time_window_edges:
        # a time-window edge must contain four elements
        if len(tw_edge) != 4:
            print(f"警告：时间窗边 {tw_edge} 格式错误（需传入4个元素：起点,终点,开始时间,结束时间），已跳过")
            continue
        u_tw, v_tw, t_start, t_end = tw_edge
        # the edge must exist in the map
        if (u_tw in all_node_ids and  # source exists
            v_tw in all_node_ids and  # target exists
            (u_tw, v_tw) in original_edges):  # edge exists
            valid_tw_edges.append(tw_edge)
        else:
            print(f"警告：时间窗边 ({u_tw},{v_tw}) 无效（节点不存在或边不可达），已跳过")

    # Build the graph
    G = nx.DiGraph()
    G.add_nodes_from(all_node_ids)
    G.add_weighted_edges_from(edges_with_weight)

    # Layout: edge weights act as layout distances
    pos = nx.kamada_kawai_layout(
        G,
        weight='weight',  # weight used as the layout distance
        scale=20          # overall drawing scale
    )

    # Draw in layers so later layers stay visible
    plt.figure(figsize=(20, 16), dpi=100)

    # Nodes: normal nodes first, special nodes on top
    special_node_set = set(special_nodes)
    normal_nodes = [node for node in all_node_ids if node not in special_node_set]

    # Draw normal nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=normal_nodes,
        node_size=1200,
        node_color='skyblue',
        edgecolors='navy',
        linewidths=2
    )

    # Special nodes: larger, thicker border
    if special_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=special_nodes,
            node_size=1500,  # larger than normal nodes
            node_color=special_color,
            edgecolors='black',
            linewidths=3,    # thicker border
            alpha=0.9        # slightly more opaque
        )

    # Edges: base edges, then path edges on top
    # Base edges (grey)
    if edges_with_weight:  # guard against a graph without edges
        all_weights = [w for u, v, w in edges_with_weight]
        max_w, min_w = max(all_weights), min(all_weights)
    else:
        max_w, min_w = 1, 1  # fallback when there are no edges

    for u, v, weight in edges_with_weight:
        # fixed width and opacity
        edge_width = 3
        edge_alpha = 0.5

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=edge_width,
            edge_color = mcolors.to_rgba('darkgray', edge_alpha),
            arrowstyle='->',
            arrowsize=20
        )

    # Path edges on top
    if len(path_sequence) >= 2:
        # keep path edges that exist in the map
        path_edges = [(path_sequence[i], path_sequence[i+1])
                      for i in range(len(path_sequence)-1)
                      if (path_sequence[i], path_sequence[i+1]) in original_edges]
        if path_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                width=6,          # wide enough to stand out without hiding the base edges
                edge_color=path_color,
                arrowstyle='->',
                arrowsize=30,
                alpha=0.8
            )

    # Labels: nodes, then edge weights, then time windows
    # Node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_color='navy'
    )

    # Edge weight labels
    edge_labels = {(u, v): f'{weight}' for u, v, weight in edges_with_weight}
    if edge_labels:
        # Place the label perpendicular to the edge
        def get_label_pos(pos, u, v, offset=0.1):
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            dx = x2 - x1
            dy = y2 - y1

            # unit normal, obtained by rotating the direction by 90 degrees
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length == 0:  # zero-length edge
                return (x1, y1)
            perp_dx = -dy / length
            perp_dy = dx / length

            # midpoint plus a perpendicular offset
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label_x = mid_x + perp_dx * offset
            label_y = mid_y + perp_dy * offset

            return (label_x, label_y)

        tw_edge_pairs = [(u, v) for u, v, t_start, t_end in valid_tw_edges]
        all_edge_pairs = [(u, v) for (u, v), label in edge_labels.items()]
        # draw each label manually to control the offset
        for (u, v), label in edge_labels.items():
            if (u, v) in all_edge_pairs:
                if (u, v) in tw_edge_pairs:
                    label_x, label_y = get_label_pos(pos, u, v, offset=0.7)
                    plt.text(
                        label_x, label_y,
                        label,
                        fontsize=18,
                        color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='none', alpha=0.7,edgecolor='none'),
                        ha='center',
                        va='center'
                    )
                else:
                    current_edge_label = {(u, v): label}
                    nx.draw_networkx_edge_labels(
                        G, pos,
                        edge_labels=current_edge_label,
                        font_size=18,
                        font_color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='none', alpha=0.7,edgecolor='none')
                    )
                all_edge_pairs.remove((u, v))
                all_edge_pairs.remove((v, u))

    # Time-window labels
    if valid_tw_edges:
        tw_labels = {(u, v): f'T:[{t_start:.2f}, {t_end:.2f}]'
                     for u, v, t_start, t_end in valid_tw_edges}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=tw_labels,
            font_size=11,
            font_color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.4)  # yellow background highlights the windows
        )

    # Title, save and display
    if valid_tw_edges:
        title = f'GA'
    else:
        title = 'GA'
    plt.title(title, fontsize=35, fontweight='bold', pad=30)

    plt.axis('off')  # hide axes
    plt.tight_layout()  # avoid clipped labels
    plt.savefig('GA.svg', format='svg', bbox_inches='tight')
    plt.show()


class GAPathSolution:
    def __init__(self, actions: List[int], env: 'Sample_Env',
                 start_pos=0, start_time=0.0):
        self.actions = actions
        self.env = env
        self.start_pos = start_pos
        self.start_time = start_time
        self.fitness = None  # distance / cost
        self.reward = None  # reward
        self.Pass = False  # all constraints satisfied
        # The path is built lazily in evaluate() to keep object creation cheap

    def evaluate(self):
        """
        Compute distance (fitness) and reward in a single pass.
        """
        # return the cached value
        if self.fitness is not None:
            return self.fitness

        # reset the environment once, at start_pos/start_time
        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)

        total_distance = 0.0
        total_reward = 0.0
        violation_count = [0, 0]  # [invalid-node count, time-window violation count]
        done = False

        # single pass over the action sequence
        valid_action_count = 0
        for i, action in enumerate(self.actions):
            current_pos = self.env.agent_pos

            # N_one_step returns the distance and violations but not the reward
            dist, _, viol = self.env.N_one_step(action, dry_run=False)
            # dry_run=False advances agent_pos and agent_time

            total_distance += dist
            violation_count[0] += viol[0]
            violation_count[1] += viol[1]

            # Reward is not used for ranking here; only the fitness is computed

            valid_action_count = i + 1
            # Completion matches ENV.one_step: all targets visited and back at start_pos.
            # (N_one_step reports done on reaching the last target, which would
            #  truncate the return leg and leave the solution away from the depot)
            if len(self.env.goal_states) == 0 and \
                    self.env.agent_pos == self.start_pos:
                done = True
                break

        # drop trailing actions after completion
        if done and valid_action_count < len(self.actions):
            self.actions = self.actions[:valid_action_count]

        # penalise unfinished paths
        if not done:
            total_distance += 10000
            self.Pass = False
        else:
            # Pass requires completion and zero violations
            if violation_count[0] == 0 and violation_count[1] == 0:
                self.Pass = True
            else:
                self.Pass = False

        self.fitness = total_distance

        # Reward is a placeholder here; the fitness pass does not track it
        self.reward = 0

        return self.fitness


class GA:
    def __init__(self, env: 'Sample_Env', pop_size: int = 200, max_iter=100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 tour_size=2, start_pos: int = 0, start_time=0.0):
        self.env = env
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pareto_front = None
        self.max_iter = max_iter
        self.tour_size = tour_size
        self.start_pos = start_pos
        self.start_time = start_time
        self.best_path = None

    def initialize_population(self) -> List[GAPathSolution]:
        population = []
        for i in range(self.pop_size):
            goal_order = self.env.goal_states1.copy()
            random.shuffle(goal_order)
            goal_order.append(self.env.start_pos)

            actions = []
            current_pos = self.start_pos

            # build a random route through the shuffled targets
            from GA import random_weighted  # helper from GA
            for goal in goal_order:
                res = random_weighted(self.env.map, current_pos, goal)
                sub_path = res['actions']
                actions.extend(sub_path)
                # advance to the last node of the sub-path
                if res['path']:
                    current_pos = res['path'][-1][2]
                else:
                    current_pos = goal  # fallback

            solution = GAPathSolution(actions, self.env)
            solution.evaluate()
            population.append(solution)
        return population

    def fast_sort(self, population: List[GAPathSolution]) -> List[GAPathSolution]:
        return sorted(population, key=lambda x: x.fitness)

    def tournament_selection(self, population: List[GAPathSolution]) -> GAPathSolution:
        """Select the fittest of a random sample of candidates."""
        candidates = random.sample(population, self.tour_size)
        # lowest fitness wins
        best = candidates[0]
        for cand in candidates[1:]:
            if cand.fitness < best.fitness:
                best = cand
        return best

    def crossover(self, parent1: GAPathSolution, parent2: GAPathSolution) -> Tuple[GAPathSolution, GAPathSolution]:
        if random.random() > self.crossover_rate:
            # new objects: mutation must not alter the parents
            return (GAPathSolution(parent1.actions[:], self.env,
                                   start_pos=self.start_pos,
                                   start_time=self.start_time),
                    GAPathSolution(parent2.actions[:], self.env,
                                   start_pos=self.start_pos,
                                   start_time=self.start_time))

        # length of the common action prefix
        min_length = min(len(parent1.actions), len(parent2.actions))
        common_prefix_length = 0
        for i in range(min_length):
            if parent1.actions[i] == parent2.actions[i]:
                common_prefix_length = i + 1
            else:
                break

        # swap the suffixes after the common prefix
        child1_actions = parent1.actions[:common_prefix_length] + parent2.actions[common_prefix_length:]
        child2_actions = parent2.actions[:common_prefix_length] + parent1.actions[common_prefix_length:]

        return (GAPathSolution(child1_actions, self.env,
                               start_pos=self.start_pos,
                               start_time=self.start_time),
                GAPathSolution(child2_actions, self.env,
                               start_pos=self.start_pos,
                               start_time=self.start_time))

    def mutate(self, solution: GAPathSolution) -> GAPathSolution:
        if random.random() > self.mutation_rate:
            return solution

        actions = solution.actions  # the action list is already a private copy

        if not actions: return solution

        mutation_type = random.choice(['替换', '插入', '删除'])

        if mutation_type == '替换':
            idx = random.randint(0, len(actions) - 1)
            actions[idx] = random.randint(0, self.env.action_size - 1)
        elif mutation_type == '插入':
            idx = random.randint(0, len(actions))  # may append at the end
            actions.insert(idx, random.randint(0, self.env.action_size - 1))
        elif mutation_type == '删除' and len(actions) > 1:
            idx = random.randint(0, len(actions) - 1)
            actions.pop(idx)

        # invalidate the cached fitness
        solution.fitness = None
        return solution

    def evolve(self, population: List[GAPathSolution]) -> List[GAPathSolution]:
        # Elitism: keep the best 10%; sort defensively in case the input is unsorted
        sorted_pop = self.fast_sort(population)
        elite_count = int(self.pop_size * 0.1)
        new_population = sorted_pop[:elite_count]  # elites keep their evaluated fitness

        # Offspring
        offspring = []
        target_count = self.pop_size - elite_count

        # two offspring per crossover
        while len(offspring) < target_count:
            p1 = self.tournament_selection(sorted_pop)
            p2 = self.tournament_selection(sorted_pop)

            c1, c2 = self.crossover(p1, p2)

            c1 = self.mutate(c1)
            c2 = self.mutate(c2)

            # evaluate only the offspring that were mutated
            if c1.fitness is None: c1.evaluate()
            if c2.fitness is None: c2.evaluate()

            offspring.append(c1)
            offspring.append(c2)

        # drop the extra offspring
        if len(offspring) > target_count:
            offspring = offspring[:target_count]

        return new_population + offspring

    def optimize(self):
        population = self.initialize_population()
        # initial ranking
        population = self.fast_sort(population)
        self.pareto_front = population[0]

        for generation in range(self.max_iter):
            population = self.evolve(population)

            # after evolve only the elite prefix is sorted; scan for the minimum
            current_best = min(population, key=lambda x: x.fitness)

            if current_best.fitness < self.pareto_front.fitness:
                self.pareto_front = current_best

            if (generation + 1) % 10 == 0:
                print(f"GA 代 {generation + 1}/{self.max_iter}: 最小权重={self.pareto_front.fitness:.2f}")

        self.best_path = self.pareto_front.actions
        return self.pareto_front

    def visualize_best_path(self):
        print(f"最佳路径动作: {self.pareto_front.actions}")
        print(f"最佳Fitness: {self.pareto_front.fitness}")


def random_weighted(map_data, start, end):
    predecessors = {node: None for node in range(len(map_data))}
    actions = {node: None for node in range(len(map_data))}
    visited = set()
    stack = [(start, None)]
    found = False
    while stack:
        current_node, _ = stack.pop()
        if current_node in visited: continue
        visited.add(current_node)
        if current_node == end:
            found = True
            break
        neighbors = []
        for action in range(len(map_data[current_node])):
            next_node, _, _ = map_data[current_node][action]
            if next_node != -1:
                neighbors.append((action, next_node))
        random.shuffle(neighbors)
        for action, next_node in neighbors:
            if next_node not in visited:
                predecessors[next_node] = current_node
                actions[next_node] = action
                stack.append((next_node, action))
    if not found or predecessors[end] is None:
        return {'path': [], 'actions': []}
    path = []
    action_list = []
    current = end
    while current != start:
        prev_node = predecessors[current]
        act = actions[current]
        path.append((prev_node, act, current))
        action_list.append(act)
        current = prev_node
    action_list.reverse()
    path.reverse()  # rebuild below as (start, action, next) triples
    full_path_tuples = []
    curr = start
    for a in action_list:
        nxt = map_data[curr][a][0]
        full_path_tuples.append((curr, a, nxt))
        curr = nxt
    return {'path': full_path_tuples, 'actions': action_list}

if __name__ == "__main__":
    env_seed = 9
    random.seed(env_seed)
    np.random.seed(env_seed)

    n = 300
    G1 = generate_directed_graph(n)
    map, node_size, edge_size = graph_to_dict(G1)
    target_num = int((node_size * 8) / 25)
    target_nodes = random.sample(range(1, node_size), target_num)
    edge = [random.randint(10, 50) for _ in range(edge_size)]
    times = [random.randint(10, 50) for _ in range(edge_size)]
    env = Sample_Env(map=map, nodes_num=node_size, edges_num=edge_size, edges=edge, target=target_nodes, times=times,
                     same=True, one_same=False)

    env.reset_agent_time(0.0)
    env.reset_env()
    env.reset_agent_pos(0)

    # initialize the genetic algorithm
    ga = GA(
        env=env,
        pop_size=200,  # population size
        max_iter=100,  # generations
        crossover_rate=0.8,  # crossover rate
        mutation_rate=0.1,  # mutation rate
        tour_size=2,  # tournament size
        start_pos=0,  # start node
        start_time=0  # start time
    )
    # run
    start_time = T.time()
    ga.optimize()
    ga.visualize_best_path()
    print("运行时间: {:.2f}s".format(T.time() - start_time))
    solution = PathSolution(ga.pareto_front.actions, env)
    solution.evaluate()