# EFormer-STSP 适配说明

将开源 EFormer（IJCAI 2025, *An Effective Edge-based Transformer for Vehicle Routing
Problems*）适配到论文问题场景 **S-TSP-TVRA**：带时变道路通行限制的 Steiner 旅行商问题
（稀疏有向路网、边级受限时间窗、禁止等待、访问全部目标后返回仓库）。

本目录为完全适配论文场景的精简版本；原版 TSP/CVRP 代码未包含在本目录中。

## 1. 问题场景与映射关系

| 维度 | 原版 EFormer (TSP) | 论文场景 S-TSP-TVRA | 适配方式 |
|---|---|---|---|
| 图结构 | 完全图 + 欧氏坐标 | 稀疏有向图（出度≤4），边有 length/speed | 节点级稠密张量 + 邻接掩码 |
| 输入 | 节点坐标 (B,N,2) | 边长、行驶速度、边级时间窗 | 边特征 (B,N,N,8) |
| 任务 | 访问全部节点 | 访问目标子集（Steiner），可途经中转节点，返回仓库 | 节点标志 + 终止条件改造 |
| 时间 | 无 | 当前时刻演化 t=(t+speed/60)%24；闭区间受限窗（可跨午夜） | 时间特征 + 动态动作掩码 |
| 动作 | 不受到边时间窗限制 | 只能选当前节点的出邻居（稀疏局部动作） | 动态掩码（非邻居 -inf） |
| 训练 | POMO 多起点 REINFORCE | 同左，但起点固定 | pomo 维度 = 并行采样 rollout，组内均值 baseline |
| 评估 | 最优 tour 长度 gap | pass rate / 条件成本 / gap vs Gurobi / 时间 | 对接论文 benchmark 实例池 |

## 2. 模型结构（保持原版三段式与规模）

- **Precoder**（MatNet 式混合得分注意力）：输入代价矩阵（归一化边长，无边=1.5 域外
  哨兵，对角=0），将边信息转化为临时节点嵌入。结构不变。
- **GraphEncoder**（ResidualGatedGCN，6 层）：边值嵌入由 `Linear(1)` 扩展为
  `Linear(8)`（8 通道边特征，见下）；tag 嵌入 {0=无边, 1=有向边, 2=对角} 不变。
- **NodeEncoder**（6 层 Transformer）：输入 = Precoder 特征 + 节点标志嵌入
  （`Linear(2)`：起点/目标标志），保持边中心原则，标志仅作辅助信号。
- **Decoder**（双流上下文注意力）：原版 q_first（起点嵌入）+ q_last（当前节点嵌入）
  之外，新增两个适配上下文：
  - `q_target`：剩余未访问目标节点的掩码均值池化（Steiner 终止感知）；
  - `q_time`：当前时刻的多周期 sin/cos 编码（周期 30min/2h/8h/24h，对齐论文
    `time_to_multi_sincos`）投影（时间感知）。

模型规模与原版完全一致（embedding 256 / heads 16 / 编码器 6 层 / GCN 6 层 /
qkv 16 / FF 512），满足审稿意见 C4 对基线架构规模一致的要求。

### 边特征 8 通道（相对出发时刻 t0）

| 通道 | 含义 |
|---|---|
| 0 | length/50（归一化边长） |
| 1 | speed/50（归一化速度） |
| 2 | 边是否存在 |
| 3 | 是否有受限时间窗 |
| 4/5 | sin/cos((s-t0)%24 / 24)（窗口起点，周期 24h） |
| 6/7 | sin/cos((e-t0)%24 / 24)（窗口终点） |

### 编码方式与时间感知的说明

采用**一次性编码（one-shot）+ 解码期动态动作掩码**，与论文对基线的描述
（"equipped with a dynamic action mask"）一致；这也构成审稿意见 C11 中
one-time encoding vs step-wise re-encoding 消融的对照配置。当前时刻信息通过
两路进入模型：(a) 解码器 `q_time` 上下文；(b) 动态掩码屏蔽当前时刻受限的边。

### 动态动作掩码（稀疏局部动作的显式处理）

每步可选动作 = 当前节点的出邻居 ∧ 当前时刻不受限的边：
- 非邻居节点：-inf；
- 受限窗内（闭区间 [s,e]，可跨午夜）：-inf；
- 死局（全部出边受限）：回合按失败终止；
- 已结束 rollout：停在当前节点（one-hot），保证批内张量规整；
- 不允许等待（与论文一致）。

## 3. 数据与训练协议（对齐 EFEDQN，保证公平比较）

- **实例生成**：直接调用论文代码 `long_reward/Gragh.py::generate_directed_graph`
  与 `long_reward/Env.py::Sample_Env`（单一来源，分布完全一致）：边长/速度
  i.i.d. U{1..50}，时间窗数量 ~ N(|E|/4, |E|/8) 截断 [16, |E|/3]，
  窗口 [s, s+1+U(0,12)] mod 24（闭区间，可跨午夜）。
- **训练分布**（对齐 `long_reward/train.py`）：n=100、目标数 = n/3（可含仓库）、
  随机非目标起点、出发时间 U(0,24)。实例池预生成后循环使用。
- **优化器**：Adam(lr=1e-4, weight_decay=1e-6)，与原版 EFormer 一致。
- **奖励**：-(总边长 + 1000 × 失败)，失败 = 死局或超过 100 步未完成。
- **训练目标**：POMO 式 REINFORCE，每实例 pomo_size 个并行采样 rollout，
  advantage = reward − 组内均值（共享 baseline）。

## 4. 评估协议（对齐 long_reward/evaluate.py）

- **池模式**（默认）：加载认证实例池 `benchmark/n{10,20,30,50}/*.json`
  （Gurobi 认证），逐实例贪心 rollout（起点=节点0、出发时间=0、实际移动步数≤100），
  输出与 EFEDQN 相同口径的指标：`cost / feasible / gap(vs OPTIMAL 或 bound) /
  time_ms`，以及汇总 `pass_rate / mean_cost_feasible / mean_gap_pct / fitness /
  time`——同一实例池支持与 EFEDQN 的配对比较。
- **随机模式**：按论文表 6/7 的 5 组条件（25n_16tw / 50n_32tw / 100n_64tw /
  100n_32tw / 100n_128tw）现场生成实例评估。
- **best-of-K 协议**（`--aug-k K`）：额外输出 K 次 rollout 取最优的结果，
  对应 EFEDQN-E 的对称搜索预算比较（审稿意见 C4）。多样性来源由
  `--aug-mode` 决定：`sample`（默认）为逐步 multinomial 采样解码（pomo 维
  复用为 K 个并行采样，共享一次编码）；`seed` 为**原版 EFormer 实例增强**
  ——同一实例复制 K 份沿 batch 维，一次批量 pre_forward 中每份独立重抽
  Precoder 的 MatNet 式随机 one-hot 列嵌入种子，解码全程贪心 argmax
  （不切 softmax），即原论文"同一实例多次输入、每次生成唯一嵌入"的增强协议。

## 5. 使用方法

```bash
# 训练（默认 n=100，对齐 EFEDQN 训练分布；每 batch 现场随机生成实例，一次性消耗）
# 每 10 个 epoch 在固定验证集（初始化时一次性生成，独立种子）上贪心评估；
# 连续 5 次测试无改善则早停，最优模型存为 checkpoint-best.pt
python train.py --epochs 500
python train.py --epochs 500 --test-interval 25 --test-size 64 --patience 8

# 冒烟测试（快速验证流程，n=25）
python train.py --smoke

# 评估：认证实例池（与 EFEDQN 相同实例，配对 gap）
python test.py --model-path result/xxx/checkpoint-500.pt \
               --instances-dir ../../long_reward/benchmark

# 评估：实例池 + best-of-10 实例增强协议（默认 sample：逐步采样 + 共享编码）
python test.py --model-path ... --aug-k 10

# 评估：随机条件模式（论文表 6/7 协议）
python test.py --model-path ... --mode random --num-instances 100
```

依赖：`torch>=2.0`、`numpy`、`networkx`（实例生成复用论文代码）、`matplotlib`
（论文 `Gragh.py` 顶层 import，仅 import 不出图）。

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `STSPProblemDef.py` | 实例生成（复用论文生成器）、特征张量构建、benchmark JSON 加载 |
| `STSPEnv.py` | 向量化环境：动态动作掩码、时间演化、Steiner 终止条件、奖励 |
| `STSPModel.py` | 适配后的 EFormer 模型（Precoder + 双编码器 + 时间/目标感知解码器） |
| `gcn_layers.py` | ResidualGatedGCN 层（原版保留） |
| `STSPTrainer.py` | POMO 式 REINFORCE 训练器 |
| `STSPTester.py` | 评估器（池模式 / 随机模式 / best-of-K，对齐论文指标） |
| `train.py` / `test.py` | 训练 / 评估入口 |
| `utils.py` | 精简后的日志与计时工具 |

## 7. 相对原版 EFormer 移除的内容

- CVRP 全部代码（与本问题无关）；
- TSP 的 n20/n50/n100 训练/测试脚本与数据加载（.txt/.pkl、最优解标签）；
- KNN 图构建与 `KNN_edge / Gknn_Nab / Gab_Nknn` 等方法分支（本问题图结构给定）；
- 欧氏坐标 8-fold 增广（场景无坐标）；
- pytz 时区、matplotlib 日志绘图、`copy_all_src` 等非必要依赖。

## 8. 已知设计决策（供论文写作与复现说明引用）

1. **一次性编码 vs 逐步重编码**：EFormer 基线采用一次性编码 + 动态掩码，
   EFEDQN 采用逐步重编码；两者对比即审稿意见 C11 要求的消融之一。
2. **掩码为硬约束**：受限边不可通行（无 -10 软惩罚），与论文对基线的描述一致；
   EFEDQN 允许带惩罚违规，二者在此语义上不同，比较时已注明。
3. **失败惩罚 1000**：训练奖励中的失败惩罚（对齐论文成本口径中每次非法动作
   +1000 的量级）；评估口径中失败仅记 feasible=0（fitness 惩罚 1e4），
   不向 cost 加惩罚，保持与 evaluate.py 的条件成本统计一致。
4. **无边哨兵 1.5**：Precoder 代价矩阵中无边条目取 1.5（归一化边长域 [0.02,1.0]
   之外），配合 tag 嵌入区分"无边"与"长边"。
