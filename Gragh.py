import copy

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


# def generate_directed_graph(num_nodes):
#     """
#     生成一个固定节点数的有向图，每个节点最多与四个其他节点相连（四个方向），
#     且双向边的长度相同，但方向和速度可以不同
#
#     参数:
#     - num_nodes: 节点数量
#
#     返回:
#     - G: 生成的有向图
#     """
#
#     # 创建有向图
#     G = nx.DiGraph()
#
#     # 添加节点
#     G.add_nodes_from(range(num_nodes))
#
#     # 为每个节点分配坐标
#     coordinates = {}
#     for node in G.nodes:
#         x = random.uniform(0, 100)
#         y = random.uniform(0, 100)
#         coordinates[node] = (x, y)
#
#     # 计算节点间的欧几里得距离
#     def distance(u, v):
#         x1, y1 = coordinates[u]
#         x2, y2 = coordinates[v]
#         return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#     # 为每个节点找到最近的候选节点
#     candidate_edges = []
#     for u in G.nodes:
#         distances = [(v, distance(u, v)) for v in G.nodes if v != u]
#         distances.sort(key=lambda x: x[1])
#
#         # 选择最近的k个节点
#         k = min(4, len(distances))
#         for v, _ in distances[:k]:
#             candidate_edges.append((u, v))
#
#     # 随机打乱候选边的顺序
#     random.shuffle(candidate_edges)
#
#     # 添加边，同时确保每个节点的出度和入度不超过4
#     for u, v in candidate_edges:
#         if G.out_degree(u) >= 4 or G.in_degree(v) >= 4:
#             continue
#
#         common_length = random.randint(10, 50)
#         speed_a = random.choice([40, 60, 80, 100])
#
#         G.add_edge(u, v, direction=f"{u}->{v}", length=common_length, speed=speed_a)
#         G.add_edge(v, u, direction=f"{v}->{u}", length=common_length, speed=speed_a)
#
#     # 确保图是弱连通的
#     max_attempts = 100  # 防止无限循环
#     attempt = 0
#
#     while not nx.is_weakly_connected(G) and attempt < max_attempts:
#         components = list(nx.weakly_connected_components(G))
#         comp_count = len(components)
#
#         if comp_count < 2:
#             break
#
#         # 尝试连接所有不相连的分量对
#         for i in range(comp_count):
#             for j in range(i + 1, comp_count):
#                 # 从两个分量中各选一个节点
#                 u = random.choice(list(components[i]))
#                 v = random.choice(list(components[j]))
#
#                 # 尝试添加双向边 (u->v 和 v->u)
#                 if G.out_degree(u) < 4 and G.in_degree(v) < 4:
#                     common_length = random.randint(10, 50)
#                     speed_uv = random.choice([40, 60, 80, 100])
#
#                     G.add_edge(u, v, direction=f"{u}->{v}", length=common_length, speed=speed_uv)
#
#                     # 检查反向边是否可以添加
#                     if G.out_degree(v) < 4 and G.in_degree(u) < 4:
#                         G.add_edge(v, u, direction=f"{v}->{u}", length=common_length, speed=speed_uv)
#
#         attempt += 1
#
#     # 最后检查，如果仍不连通则强制添加边
#     if not nx.is_weakly_connected(G):
#         components = list(nx.weakly_connected_components(G))
#
#         for i in range(len(components) - 1):
#             u = random.choice(list(components[i]))
#             v = random.choice(list(components[i + 1]))
#
#             # 找到u的一个出度空位
#             while G.out_degree(u) >= 4:
#                 u = random.choice(list(components[i]))
#
#             # 找到v的一个入度空位
#             while G.in_degree(v) >= 4:
#                 v = random.choice(list(components[i + 1]))
#
#             common_length = random.randint(10, 50)
#             speed_uv = random.choice([40, 60, 80, 100])
#
#             G.add_edge(u, v, direction=f"{u}->{v}", length=common_length, speed=speed_uv)
#
#             # 尝试添加反向边
#             if G.out_degree(v) < 4 and G.in_degree(u) < 4:
#                 G.add_edge(v, u, direction=f"{v}->{u}", length=common_length, speed=speed_uv)
#
#     return G

def get_direction_number(u, v):
    """
    判断点v相对于点u的方向并返回对应数字

    参数:
        u: 元组或列表，包含u点的坐标 (u_x, u_y)
        v: 元组或列表，包含v点的坐标 (v_x, v_y)

    返回:
        int: 方向对应的数字
             0 - 左上
             1 - 右上
             2 - 左下
             3 - 右下
    """
    # 提取坐标
    u_x, u_y = u
    v_x, v_y = v

    # 判断左右方向
    if v_x < u_x:
        is_left = True
    elif v_x > u_x:
        is_left = False
    else:
        raise ValueError("v点与u点在同一竖直线上，无法判断左右方向")

    # 判断上下方向
    if v_y < u_y:
        is_up = True
    elif v_y > u_y:
        is_up = False
    else:
        raise ValueError("v点与u点在同一水平线上，无法判断上下方向")

    # 根据方向组合返回对应数字
    if is_left and is_up:
        return 0  # 左上
    elif not is_left and is_up:
        return 1  # 右上
    elif is_left and not is_up:
        return 2  # 左下
    else:  # not is_left and not is_up
        return 3  # 右下


def generate_directed_graph(num_nodes):
    """生成弱连通的有向图，每个节点度数不超过4"""


    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # 为每个节点分配坐标
    coordinates = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in G.nodes}

    # 计算欧几里得距离
    def distance(u, v):
        x1, y1 = coordinates[u]
        x2, y2 = coordinates[v]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def has_element_with_value(candidate_edges,u,value):
        """
        检查candidate_edges中是否存在元素的第三个值为1

        参数:
            candidate_edges: 包含元素的列表，每个元素应是至少有3个元素的可迭代对象

        返回:
            bool: 如果存在这样的元素返回True，否则返回False
        """
        # 遍历列表中的每个元素
        for item in candidate_edges:
            # 检查元素是否有至少3个值，并且第三个值为1
            if item[2] == value and item[0] == u:
                return True
        # 遍历完所有元素都没有找到符合条件的，返回False
        return False

    candidate_edges = []
    for u in G.nodes:
        distances = [(v, distance(u, v),get_direction_number(coordinates[u],coordinates[v])) for v in G.nodes if v != u and not G.has_edge(u, v)]

        # 选择最近的k个节点
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

    # 随机添加候选边，同时确保度数不超过4
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




# 可视化函数（更新以适应有向图）
def visualize_directed_graph(G):
    """可视化有向图，并显示边的属性"""
    plt.figure(figsize=(12, 10))

    # 使用spring布局算法
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.7)

    # 绘制有向边（使用箭头表示方向）
    edges = G.edges()
    edge_colors = ['gray' for _ in edges]
    edge_widths = [1.5 for _ in edges]

    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_colors,
        width=edge_widths, arrows=True, arrowsize=20,
        arrowstyle='->', connectionstyle='arc3,rad=0.1'
    )

    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='SimHei')

    # 添加边标签（显示长度和速度）
    edge_labels = {(u, v): f"{G[u][v]['direction']}\nlong:{G[u][v]['length']}\nspeed:{G[u][v]['speed']}"
                   for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"nodes_num:{len(G.nodes)}", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 示例：生成一个包含10个节点的有向图
# for i in range(100):
#     G = generate_directed_graph(num_nodes=25)
#
#     # 打印图的基本信息
#     print(f"nodes_num: {G.number_of_nodes()}")
#     print(f"edges_num: {G.number_of_edges()}")
#
#     # 可视化图
#     visualize_directed_graph(G)
#
#     # 验证双向边的长度是否一致
#     print("\n验证双向边的长度一致性:")
#     for u, v in list(G.edges)[:5]:
#         if G.has_edge(v, u):
#             print(f"边 {u}->{v}: 长度={G[u][v]['length']}, 速度={G[u][v]['speed']}")
#             print(f"边 {v}->{u}: 长度={G[v][u]['length']}, 速度={G[v][u]['speed']}")
#             print(f"长度是否一致: {G[u][v]['length'] == G[v][u]['length']}")
#             print("-" * 30)


def graph_to_dict(G):
    """将图转换为包含节点和边信息的字典"""
    # 提取节点信息
    # for node_id in G.nodes:
    #     node_data = {
    #         "id": node_id,
    #         "attributes": dict(G.nodes[node_id])  # 获取节点所有属性
    #     }
    #     graph_data["nodes"].append(node_data)

    map = [[[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]] for _ in range(G.number_of_nodes())]

    # 提取边信息（无向图每条边只记录一次）
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