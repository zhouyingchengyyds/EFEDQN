# 实验框架权重/数据放置约定

本目录下各子目录存放实验脚本（exp0–exp7）所需的模型权重与数据文件。
文件补齐后无需改动任何代码，直接运行对应实验脚本即可。

## 需要补充的文件清单

| 路径 | 内容 | 用于实验 |
|---|---|---|
| `No_timwin/model_seed1_no_tw.pth` | 无边时间窗特征的静态编码版 EFEDQN（架构与 GAT_DQN 相同、整模型保存，训练时窗口特征以常数 -2 填充），种子 1 | exp2（C4 公平对比表）、exp3（C5 诊断变体 a/c） |
| `Freeze_code/model_seed1_no_dy.pth`（+ 模型定义 `GAT_Frozen.py`） | 冻结编码版 EFEDQN（一次编码、不再逐步重编码；解码器输入 4 维流逝时间特征、推理与完整模型同样施加动态动作掩码；整模型保存，torch.load 整模型时按该目录解析类），由 `Freeze_code/train.py` 训练产出，种子 1 | exp3（C5 诊断变体 b）、exp4（C8 扰动重规划） |
| `EFEDQN/model_seed2.pth`、`EFEDQN/model_seed3.pth` | 完整模型种子 2/3（与 model_seed1.pth 同目录、同保存方式） | exp5（消融表 Full model 行的 mean±std） |
| `Ablation/Gate_control/model_seed{1,2,3}.pth`（+ 模型定义 .py） | 门控机制消融模型；文件夹内需含该模型类的定义文件（torch.load 整模型时按该目录解析类） | exp5 |
| `Ablation/Decoder_arch/model_seed{1,2,3}.pth`（+ 模型定义 .py） | 解码器架构消融（普通 MLP vs 残差 MLP） | exp5 |
| `Ablation/Replay_strategy/model_seed{1,2,3}.pth`（+ 模型定义 .py） | 经验回放消融（均匀采样 vs PER） | exp5 |
| `EFEDQN/output_seed1/off_reward_{1..6}.xlsx`、`On_reward_{1..6}.xlsx` | 训练期周期评估数据（3000 回合、每 100 回合评估一次 = 每文件 30 行；列 = 15 个测试环境） | exp0（图 4/图 5 训练曲线） |

## 已就位

- `EFEDQN/model_seed1.pth`（EFEDQN / EFEDQN-E 共用）
- `EFormer/checkpoint-best.pt`（EFormer / EFormer-K{k} 共用）
- `benchmark/`（Gurobi 认证实例池，exp2/exp3 使用）
- `real_scene/luoyang_stsp_20node.json`（exp7 使用）
- `Re_planning/rolling_horizon.py`（滚动时域求解器，exp2/exp4 使用）

## 运行环境

需要 PyTorch（含 torch_geometric，加载 GAT_DQN）、pandas、openpyxl、matplotlib、networkx、numpy。
所有实验默认种子 1，逐实例逐方法派生，输出至 `results/<实验名>/`。

## 每张表/图的复现命令（供审稿人回复引用）

| 论文表/图 | 命令（在 test/ 目录下） |
|---|---|
| 图 4/图 5（训练评估曲线） | `python exp0_training_curves.py` |
| 表 6/表 7（主对比） | `python exp1_main_comparison.py` |
| 表 8（认证池解质量+公平预算合并表） | `python exp2_quality_fair.py` |
| 表 9（时间诊断） | `python exp3_temporal_diag.py` |
| 表 10（在线扰动重规划） | `python exp4_disruption.py` |
| 表 11（消融+置换等变对照） | `python exp5_ablation.py` |
| 表 12/表 13（规模外推/分布偏移） | `python exp6_generalization.py` |
| 图 3 + 表 14（真实路网） | `python exp7_real_scene.py` |
