import copy

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


def get_direction_number(u, v):
    """
    Return the direction of point v relative to point u.

    Args:
        u: tuple or list with the coordinates of u (u_x, u_y)
        v: tuple or list with the coordinates of v (v_x, v_y)

    Returns:
        int: direction code
             0 - upper left
             1 - upper right
             2 - lower left
             3 - lower right
    """
    # Unpack the coordinates
    u_x, u_y = u
    v_x, v_y = v

    # Left or right
    if v_x < u_x:
        is_left = True
    elif v_x > u_x:
        is_left = False
    else:
        raise ValueError("v点与u点在同一竖直线上，无法判断左右方向")

    # Above or below
    if v_y < u_y:
        is_up = True
    elif v_y > u_y:
        is_up = False
    else:
        raise ValueError("v点与u点在同一水平线上，无法判断上下方向")

    # Direction code from the sign combination
    if is_left and is_up:
        return 0  # upper left
    elif not is_left and is_up:
        return 1  # upper right
    elif is_left and not is_up:
        return 2  # lower left
    else:  # not is_left and not is_up
        return 3  # lower right


def generate_directed_graph(num_nodes):
    """Generate a weakly connected directed graph whose nodes have degree at most 4"""


    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Random coordinates for every node
    coordinates = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in G.nodes}

    # Euclidean distance
    def distance(u, v):
        x1, y1 = coordinates[u]
        x2, y2 = coordinates[v]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def has_element_with_value(candidate_edges,u,value):
        """
        Check whether candidate_edges holds an element with item[2] == value and item[0] == u

        Args:
            candidate_edges: list whose elements are iterables of at least 3 items

        Returns:
            bool: True if such an element exists, else False
        """
        # Check every element
        for item in candidate_edges:
            # require item[2] == value and item[0] == u
            if item[2] == value and item[0] == u:
                return True
        return False

    candidate_edges = []
    for u in G.nodes:
        distances = [(v, distance(u, v),get_direction_number(coordinates[u],coordinates[v])) for v in G.nodes if v != u and not G.has_edge(u, v)]

        # For each of the four directions, take the nearest node
        for _ in range(4):
            if np.random.rand() < 0.25:
                ori = 0
            elif np.random.rand() < 0.5:
                ori = 1
            elif np.random.rand() < 0.75:
                ori = 2
            else:
                ori = 3
            tt=0
            while has_element_with_value(candidate_edges,u,ori) and tt < 4:
                ori = (ori+1)%4
                tt = tt+1
            filtered_and_sorted = sorted([x for x in distances if x[2] == ori], key=lambda x: x[1])
            for v, _ , direction in filtered_and_sorted[:1]:
                candidate_edges.append((u, v, direction))

    # Add the candidate edges, keeping out-degree and in-degree at most 4
    random.shuffle(candidate_edges)
    for u, v,_ in candidate_edges:
        if G.out_degree(u) >= 4 or G.in_degree(v) >= 4:
            continue

        common_length = random.randint(10, 50)
        speed = random.choice([40, 60, 80, 100])

        G.add_edge(u, v, direction=f"{u}->{v}", length=common_length, speed=speed)
        G.add_edge(v, u, direction=f"{v}->{u}", length=common_length, speed=speed)

    if nx.is_weakly_connected(G):
        return G
    else:
        return generate_directed_graph(num_nodes)




# Visualisation helpers
def visualize_directed_graph(G):
    """Draw the directed graph together with its edge attributes"""
    plt.figure(figsize=(12, 10))

    # Spring layout
    pos = nx.spring_layout(G, seed=42)

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.7)

    # Directed edges (arrows show the direction)
    edges = G.edges()
    edge_colors = ['gray' for _ in edges]
    edge_widths = [1.5 for _ in edges]

    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_colors,
        width=edge_widths, arrows=True, arrowsize=20,
        arrowstyle='->', connectionstyle='arc3,rad=0.1'
    )

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='SimHei')

    # Edge labels (length and speed)
    edge_labels = {(u, v): f"{G[u][v]['direction']}\nlong:{G[u][v]['length']}\nspeed:{G[u][v]['speed']}"
                   for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"nodes_num:{len(G.nodes)}", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def graph_to_dict(G):
    """Convert the graph to (map, node count, edge count) accepted by Sample_Env"""

    map = [[[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]] for _ in range(G.number_of_nodes())]

    # Record each edge once; the slot is drawn at random and probed linearly
    for u, v in G.edges:
        i = random.choice([0, 1, 2, 3])
        t=0
        while map[u][i][0] != -1 and t<4:
            i = (i+1)%4
            t += 1
        if map[u][i][0] == -1:
            map[u][i][0] = v
            map[u][i][1] = G[u][v]['length']
            map[u][i][2] = G[u][v]['speed']

    return copy.deepcopy(map),G.number_of_nodes(),G.number_of_edges()