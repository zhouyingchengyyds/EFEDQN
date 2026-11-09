"""
Time-window count verification (diagnostic only; the generation logic is untouched)

Instance construction follows train.py / evaluate.py exactly:
    generate_directed_graph(n) -> graph_to_dict -> Sample_Env(..., k_num=K)

For every instance this checks whether len(env.edge_limits) equals the requested k_num
and attributes the deviation:
  A. the sampled index j lands on an empty map slot (-1, no such arc) -- reset_edge_limits
     skips it with continue, so that window is lost
  B. sampling covers range(edge_size) (number of directed arcs in G), while j indexes
     4*nodes slots, so the two number spaces do not coincide
  C. graph_to_dict keeps only 4 slots per node, so arcs beyond out-degree 4 are silently
     dropped (edge_size > number of real slots)
"""
import random

import numpy as np

from Gragh import generate_directed_graph, graph_to_dict
from Env import Sample_Env

SEED = 1
NUM_INSTANCES = 100
N_NODES = 100        # matches the 100n_64tw condition (100 nodes)
K_NUM = 64           # requested number of time windows
TARGET_NUM = 32


def analyze(env):
    """Scan all slots using the j numbering of reset_edge_limits and collect structural statistics"""
    edge_size = env.edge_size                    # G.number_of_edges(), upper bound of the sampling range
    total_slots = env.space_size * 4             # total number of j values
    valid_slots = 0                              # slots that hold a real arc in map
    valid_below = 0                              # real slots with index < edge_size (can be sampled)
    invalid_below = 0                            # empty slots with index < edge_size (wasted if sampled)
    j = 0
    for m in range(env.space_size):
        for n in range(4):
            if env.map[m][n][0] != -1:
                valid_slots += 1
                if j < edge_size:
                    valid_below += 1
            elif j < edge_size:
                invalid_below += 1
            j += 1
    wasted_range = max(0, edge_size - total_slots)   # sampled indices >= 4n have no slot at all
    dropped_by_map = edge_size - valid_slots         # arcs dropped because graph_to_dict had no free slot
    actual = len(env.edge_limits)
    return actual, dict(edge_size=edge_size, total_slots=total_slots,
                        valid_slots=valid_slots, valid_below=valid_below,
                        invalid_below=invalid_below, wasted_range=wasted_range,
                        dropped_by_map=dropped_by_map)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"验证 {NUM_INSTANCES} 个实例: n={N_NODES}, 设定时间窗 k_num={K_NUM}")
    print("-" * 105)
    print(f"{'inst':>4} | {'G边数':>6} | {'槽位总数4n':>9} | {'真实边槽':>8} | "
          f"{'空槽(可采到)':>11} | {'浪费索引':>8} | {'实际窗数':>8} | {'设定':>4} | {'差值':>5}")
    print("-" * 105)

    exact = 0
    actuals = []
    stats_acc = []
    for inst in range(NUM_INSTANCES):
        G = generate_directed_graph(N_NODES)
        map_, node_size, edge_size = graph_to_dict(G)
        target_nodes = random.sample(range(1, node_size), TARGET_NUM)
        edge = [random.choice([10, 20, 30, 40, 50]) for _ in range(edge_size)]
        times = [random.choice([10, 20, 30, 40, 50]) for _ in range(edge_size)]
        env = Sample_Env(map=map_, nodes_num=node_size, edges_num=edge_size,
                         edges=edge, target=target_nodes, times=times, k_num=K_NUM)

        actual, st = analyze(env)
        actuals.append(actual)
        stats_acc.append(st)
        if actual == K_NUM:
            exact += 1
        print(f"{inst:>4} | {st['edge_size']:>6} | {st['total_slots']:>9} | {st['valid_slots']:>8} | "
              f"{st['invalid_below']:>11} | {st['wasted_range']:>8} | {actual:>8} | {K_NUM:>4} | {actual - K_NUM:>5}")

    actuals = np.array(actuals)
    valid_ratio = np.mean([st['valid_below'] / st['edge_size'] for st in stats_acc])
    print("-" * 105)
    print(f"恰好等于设定值 {K_NUM} 的实例: {exact}/{NUM_INSTANCES}")
    print(f"实际时间窗数: mean={actuals.mean():.2f}, min={actuals.min()}, max={actuals.max()}")
    print(f"平均缺口: {K_NUM - actuals.mean():.2f} 个（{(K_NUM - actuals.mean()) / K_NUM * 100:.1f}%）")
    print(f"所有实例实际值均 <= 设定值: {bool((actuals <= K_NUM).all())}")
    print()
    print("偏差来源统计（100 实例平均）:")
    print(f"  G 有向边数 edge_size:          {np.mean([st['edge_size'] for st in stats_acc]):.1f}")
    print(f"  槽位总数 4n:                   {np.mean([st['total_slots'] for st in stats_acc]):.1f}")
    print(f"  map 中真实边槽位:              {np.mean([st['valid_slots'] for st in stats_acc]):.1f}")
    print(f"  graph_to_dict 丢弃的边:        {np.mean([st['dropped_by_map'] for st in stats_acc]):.1f}")
    print(f"  采样范围内空槽(-1)数量:        {np.mean([st['invalid_below'] for st in stats_acc]):.1f}")
    print(f"  无槽位对应的浪费索引数:        {np.mean([st['wasted_range'] for st in stats_acc]):.1f}")
    print(f"  采样范围内真实边占比:          {valid_ratio:.4f}")
    print(f"  理论预测实际窗数 K*占比:       {K_NUM * valid_ratio:.2f}（对比观测均值 {actuals.mean():.2f}）")


if __name__ == "__main__":
    main()
