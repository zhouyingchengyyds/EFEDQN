from ENV import *
from typing import List, Tuple, Optional, Dict, Sequence
from Gragh import *
import time as T
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ASO import PathSolution


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
        title = f'PSO'
    else:
        title = 'PSO'
    plt.title(title, fontsize=35, fontweight='bold', pad=30)

    plt.axis('off')  # hide axes
    plt.tight_layout()  # avoid clipped labels
    plt.savefig('PSO.svg', format='svg', bbox_inches='tight')
    plt.show()

class PathParticle:
    def __init__(self, actions: List[int], env: 'Sample_Env', start_pos=0, start_time=0.0):
        """
        Initialize a particle path.

        Args:
            actions: action sequence
            env: path planning environment
            start_pos: start position
            start_time: departure time
        """
        self.actions = actions
        self.env = env
        self.violation = [0, 0]
        self.start_pos = start_pos
        self.start_time = start_time
        self.path = self._build_path()
        self.fitness = None
        self.velocity = []  # particle velocity (action offsets)
        self.pbest_actions = actions.copy()  # personal-best action sequence
        self.pbest_fitness = None  # personal-best fitness
        self.reward = None

    def _build_path(self) -> List[Tuple[int, float]]:
        """Build the path from the action sequence; returns (position, arrival time) tuples."""
        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)

        path = [(self.env.agent_pos, self.env.agent_time)]

        for action in self.actions:
            _, _, violation = self.env.N_one_step(action)
            path.append((self.env.agent_pos, self.env.agent_time))

        return path

    def evaluate(self) -> float:
        """Evaluate particle fitness."""
        if not self.path:
            self.fitness = float('inf')
            return self.fitness

        done = False
        distance = 0
        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)

        for i in range(len(self.actions)):
            a, _, d = self.env.N_one_step(self.actions[i])
            distance += a
            self.violation[0] += d[0]
            self.violation[1] += d[1]
            # Completion matches ENV.one_step: all targets visited and back at start_pos.
            # (N_one_step reports done on reaching the last target, which would
            #  truncate the return leg and leave the solution away from the depot)
            if len(self.env.goal_states) == 0 and \
                    self.env.agent_pos == self.start_pos:
                done = True
                if i < len(self.actions) - 1:
                    self.actions = self.actions[:i + 1]
                break

        if not done:
            distance += 10000

        self.fitness = distance
        if self.pbest_fitness is None or self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_actions = self.actions.copy()

        self.env.reset_env()
        self.env.reset_agent_pos(self.start_pos)
        self.env.reset_agent_time(self.start_time)
        reward = 0
        for i in range(len(self.actions)):
            _,r,_,_ = self.env.one_step(self.actions[i],Q=False)
            reward += r
        self.reward = reward
        return self.fitness

    def __len__(self):
        """Return the number of actions in the path."""
        return len(self.actions)


class PSO:
    def __init__(self, env: 'Sample_Env', pop_size: int = 200, max_iter=100,
                 w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
                 start_pos: int = 0, start_time=0.0):
        """
        Initialize particle swarm optimization.

        Args:
            env: path planning environment
            pop_size: swarm size
            max_iter: maximum number of iterations
            w: inertia weight
            c1: cognitive coefficient
            c2: social coefficient
            start_pos: start position
            start_time: start time
        """
        self.env = env
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.start_pos = start_pos
        self.start_time = start_time
        self.gbest_particle = None  # global-best particle
        self.particles = []
        self.best_path = None

    def initialize_swarm(self) -> List[PathParticle]:
        """Initialize the swarm from random target orders."""
        particles = []

        for i in range(self.pop_size):
            # random target order, followed by the start node
            goal_order = self.env.goal_states1.copy()
            random.shuffle(goal_order)
            goal_order.append(self.env.start_pos)

            # concatenate sub-paths between consecutive targets
            actions = []
            current_pos = self.start_pos  # current position

            for goal in goal_order:
                # sub-path from the current position to the target
                sub_path = self._generate_sub_path(current_pos, goal)
                actions.extend(sub_path)
                current_pos = goal  # advance

            particle = PathParticle(actions, self.env, self.start_pos, self.start_time)
            particle.evaluate()
            particle.pbest_fitness = particle.fitness  # personal-best fitness
            particle.pbest_actions = particle.actions.copy()  # personal-best actions
            particle.velocity = [random.choice([-1, 0, 1]) for _ in range(len(actions))]  # initial velocity

            particles.append(particle)

        return particles

    def _generate_sub_path(self, start_pos, goal_pos) -> List[int]:
        """Random sub-path from start_pos to goal_pos."""
        from GA import random_weighted
        sub_path = random_weighted(self.env.map, start_pos, goal_pos)['actions']
        return sub_path

    def update_velocity_and_position(self, particle: PathParticle, gbest_actions: List[int]):
        """Update the velocity and action sequence of a particle."""
        new_velocity = []
        new_actions = particle.actions.copy()

        max_length = max(len(particle.actions), len(particle.pbest_actions), len(gbest_actions))

        # pad all action sequences to the same length with -1
        while len(particle.actions) < max_length:
            particle.actions.append(-1)
        while len(particle.pbest_actions) < max_length:
            particle.pbest_actions.append(-1)
        while len(gbest_actions) < max_length:
            gbest_actions.append(-1)

        for i in range(max_length):
            # standard PSO velocity update
            r1, r2 = random.random(), random.random()
            cognitive = self.c1 * r1 * (
                particle.pbest_actions[i] - particle.actions[i] if particle.pbest_actions[i] != -1 and particle.actions[
                    i] != -1 else 0)
            social = self.c2 * r2 * (
                gbest_actions[i] - particle.actions[i] if gbest_actions[i] != -1 and particle.actions[i] != -1 else 0)

            # velocity as an integer action offset
            vel_change = int(self.w * particle.velocity[i] + cognitive + social) if i < len(particle.velocity) else int(
                cognitive + social)
            new_velocity.append(vel_change)

            # move the action by the velocity
            if i < len(new_actions) and new_actions[i] != -1:
                new_action = new_actions[i] + vel_change
                # clamp to the valid action range
                if 0 <= new_action < self.env.action_size:
                    new_actions[i] = new_action
                elif new_action >= self.env.action_size:
                    new_actions[i] = self.env.action_size - 1
                else:
                    new_actions[i] = 0

        # drop the -1 padding
        particle.velocity = [v for v in new_velocity if v != -1]
        particle.actions = [a for a in new_actions if a != -1]

        # occasional mutation keeps the swarm diverse
        if random.random() < 0.1:  # 10% mutation probability
            if len(particle.actions) > 0 and random.random() < 0.5:
                # replace a random action
                idx = random.randint(0, len(particle.actions) - 1)
                particle.actions[idx] = random.randint(0, self.env.action_size - 1)
            else:
                # insert a random action
                if len(particle.actions) > 0:
                    idx = random.randint(0, len(particle.actions))
                    particle.actions.insert(idx, random.randint(0, self.env.action_size - 1))

    def optimize(self):
        """Run particle swarm optimization."""
        self.particles = self.initialize_swarm()

        # global best initialised from the fittest particle
        self.gbest_particle = PathParticle(
                        min(self.particles, key=lambda p: p.fitness).actions.copy(),
                        self.env,
                        self.start_pos,
                        self.start_time
                    )
        self.gbest_particle.evaluate()
        gbest_actions = self.gbest_particle.actions.copy()
        gbest_fitness = self.gbest_particle.pbest_fitness
        gbest_reward = self.gbest_particle.reward

        # main loop
        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.evaluate()

                # update the global best
                if particle.fitness < gbest_fitness:
                    gbest_fitness = particle.fitness
                    gbest_reward = particle.reward
                    gbest_actions = particle.actions.copy()
                    # deep copy: the global best must not alias a particle
                    self.gbest_particle = PathParticle(
                        particle.actions.copy(),
                        self.env,
                        self.start_pos,
                        self.start_time
                    )
                    self.gbest_particle.evaluate()  # evaluate the copy

            # update velocity and position for every particle
            for particle in self.particles:
                self.update_velocity_and_position(particle, gbest_actions.copy())

            # progress
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iter}: 最小权重={gbest_fitness:.2f}，奖励值={gbest_reward:.2f}")
        self.best_path = [pos for pos, _ in self.gbest_particle.path]

        return self.gbest_particle

    def visualize_best_path(self):
        """Print the best path found."""
        if self.gbest_particle is None:
            print("未找到最佳路径")
            return

        solution = self.gbest_particle

        # path summary
        print(f"路径动作序列: {solution.actions}")
        print(f"路径节点序列: {[pos for pos, _ in solution.path]}")
        print(f"路径时间序列: {[time for _, time in solution.path]}")
        print(f"路径权重: {solution.fitness}")


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

    # initialize particle swarm optimization
    pso = PSO(
        env=env,
        pop_size=200,  # swarm size
        max_iter=100,  # iterations
        w=0.5,  # inertia weight
        c1=1.5,  # cognitive coefficient
        c2=1.5,  # social coefficient
        start_pos=0,  # start node
        start_time=0  # start time
    )

    # run
    start_time = T.time()
    pso.optimize()
    pso.visualize_best_path()
    print("运行时间: {:.2f}s".format(T.time() - start_time))

    solution = PathSolution(pso.gbest_particle.actions, env)
    solution.evaluate()