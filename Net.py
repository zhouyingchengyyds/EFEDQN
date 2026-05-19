import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from typing import Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class TransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: int,
        beta: bool,
        concat: bool = True,  # <--- 加上这一行，把 concat 拦截下来
        dropout: float = 0.0,
        **kwargs
    ):
        # 设定消息传递的聚合方式为 'add'
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim

        # 因为你的参数 concat=True，输出维度始终是 heads * out_channels
        hidden_dim = heads * out_channels

        # 1. 节点的 Query, Key, Value 线性变换矩阵
        self.lin_key = Linear(in_channels, hidden_dim)
        self.lin_query = Linear(in_channels, hidden_dim)
        self.lin_value = Linear(in_channels, hidden_dim)

        # 2. 边特征的线性变换矩阵 (因为你的 edge_dim 不为 None)
        self.lin_edge = Linear(edge_dim, hidden_dim, bias=False)

        # 3. 残差连接 (Skip connection) 矩阵
        self.lin_skip = Linear(in_channels, hidden_dim, bias=True)

        # 4. 门控机制 Beta 矩阵 (动态判断当前层是否启用 beta)
        if self.beta:
            self.lin_beta = Linear(3 * hidden_dim, 1, bias=False)
        else:
            self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        H, C = self.heads, self.out_channels

        # 对输入节点特征进行线性映射，并改变形状为 (N, heads, out_channels)
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # 开始消息传递机制，底层会自动调用 self.message() 函数
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)

        # 聚合完成后，因为你用了 concat=True，所以直接把多头展平拼接
        out = out.view(-1, self.heads * self.out_channels)

        # 残差连接分支
        x_r = self.lin_skip(x)

        # 如果启用了 beta，通过门控网络融合信息；否则直接相加
        if self.lin_beta is not None:
            # 拼接 [新特征, 原始映射特征, 两者之差] 送入门控网络
            beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
            beta = beta.sigmoid()
            out = beta * x_r + (1 - beta) * out
        else:
            out = out + x_r

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        # 将边特征映射到与 Key/Value 相同的维度空间
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)

        # 将边特征注入到邻居节点的 Key 中
        key_j = key_j + edge_attr

        # 计算多头注意力系数 (Dot Product) 并除以缩放因子 sqrt(d)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        # 在目标节点 (i) 的邻居范围内做 Softmax 归一化
        alpha = softmax(alpha, index, ptr, size_i)

        # Dropout (根据你的参数 dropout=0，这一步在训练时相当于无操作，但保留逻辑)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 将边特征也注入到 Value 中
        out = value_j + edge_attr

        # 用计算出来的注意力权重乘上 Value
        out = out * alpha.view(-1, self.heads, 1)

        return out

class CrossAttention(nn.Module):
    def __init__(self, temporal_dim,spacial_dim):
        super().__init__()
        self.get_weight = nn.Sequential(
            nn.Linear(temporal_dim + spacial_dim, 32),
            nn.Tanh(),
            # nn.Dropout(p=0.3),
            nn.Linear(32, 16),
            nn.Tanh(),
            # nn.Dropout(p=0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.value_temporal = nn.Linear(temporal_dim,temporal_dim)

    def forward(self,temporal_features,spatial_features):
        # 计算空间特征对时间特征的注意力
        input = torch.cat([temporal_features,spatial_features], dim=1)
        weight = self.get_weight(input)
        temporal_attended = self.value_temporal(temporal_features)
        attended_temporal = temporal_attended * weight

        return attended_temporal,weight

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块，用于计算图嵌入与节点特征之间的注意力权重
    适配输入：图嵌入为(batch, 2*embed_dim)，节点特征为(total_nodes, embed_dim)（按图分组）
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 节点嵌入维度
        self.num_heads = num_heads  # 注意力头数量
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度

        # 验证：嵌入维度必须能被头数整除，确保多头拆分无冗余
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"

        # 定义线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # 图嵌入查询投影（输入是2*embed_dim）
        self.k_proj = nn.Linear(embed_dim, embed_dim)      # 节点特征键投影
        self.v_proj = nn.Linear(embed_dim, embed_dim)      # 节点特征值投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)    # 多头结果拼接后的输出投影
        self.residual_proj = nn.Linear(embed_dim, embed_dim)  # 残差连接投影（匹配维度）

    def forward(self, graph_hidden, reshaped, nodes_sizes):
        """
        参数:
            graph_hidden: 图级别特征嵌入，形状为 (batch_size, 2*embed_dim)
            reshaped: 所有图的节点特征拼接，形状为 (total_nodes, embed_dim)
            nodes_sizes: 列表，每个元素表示对应图的节点数量，如[3,5,2]表示3个图分别有3/5/2个节点

        返回:
            new_graph_hidden: 融合节点信息后的新图嵌入，形状为 (batch_size, embed_dim)
        """
        # 获取基础维度信息
        batch_size = graph_hidden.size(0)  # 图的数量
        total_nodes, embed_dim = reshaped.size()  # 总节点数和节点嵌入维度

        # --------------------------
        # 1. 构建图节点归属掩码（核心步骤）
        # 目的：区分不同图的节点，避免跨图注意力干扰
        # --------------------------
        # 生成每个节点所属的图索引，形状为(total_nodes,)
        # 例如：nodes_sizes=[2,3] → 索引为[0,0,1,1,1]
        graph_indices = torch.cat([
            torch.full((size,), i, device=reshaped.device)  # 为第i个图的所有节点标记索引i
            for i, size in enumerate(nodes_sizes)
        ], dim=0)

        # 转换为one-hot编码：(total_nodes, batch_size)，1表示节点属于对应图
        graph_mask = F.one_hot(graph_indices, num_classes=batch_size).float()
        # 调整维度为(batch_size, 1, total_nodes)，便于后续广播到多头维度
        # 第1维设为1：为了与多头注意力的head维度兼容（后续通过广播匹配num_heads）
        mask = graph_mask.t().unsqueeze(1)  # 转置后增加维度

        # --------------------------
        # 2. 处理查询向量（图嵌入）
        # --------------------------
        # 图嵌入投影并拆分为多头：
        # (batch_size, 2*embed_dim) → 线性投影 → (batch_size, embed_dim)
        # → 拆分维度 → (batch_size, 1, num_heads, head_dim)
        # → 转置后 → (batch_size, num_heads, 1, head_dim)
        # 注：中间增加的维度1表示"每个图只有1个查询向量"
        q = self.q_proj(graph_hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # --------------------------
        # 3. 处理键和值向量（节点特征）
        # --------------------------
        # 节点特征投影并拆分为多头：
        # (total_nodes, embed_dim) → 线性投影 → (total_nodes, embed_dim)
        # → 拆分维度 → (total_nodes, num_heads, head_dim)
        # → 转置后 → (num_heads, total_nodes, head_dim)
        k = self.k_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        # 扩展为批量维度：(num_heads, total_nodes, head_dim)
        # → 增加batch维度 → (1, num_heads, total_nodes, head_dim)
        # → 扩展到batch_size → (batch_size, num_heads, total_nodes, head_dim)
        # 注：使用expand而非repeat，避免实际复制数据，节省内存
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # -1表示保持该维度不变
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # --------------------------
        # 4. 计算注意力分数
        # --------------------------
        # 矩阵乘法计算相似度：
        # q形状(batch_size, num_heads, 1, head_dim)
        # k转置后形状(batch_size, num_heads, head_dim, total_nodes)
        # 结果形状 → (batch_size, num_heads, 1, total_nodes)
        # 缩放因子：避免维度过高导致分数过大，softmax后梯度消失
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # --------------------------
        # 5. 应用掩码过滤无效节点
        # --------------------------
        # 掩码形状调整为(batch_size, 1, 1, total_nodes)，与attn_scores维度完全匹配
        # 将无效节点（掩码为0）的分数设为-1e9，确保softmax后权重为0
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        # 计算注意力权重：在节点维度做softmax，确保每个图的节点权重和为1
        attn_weights = F.softmax(attn_scores, dim=-1)

        # --------------------------
        # 6. 应用注意力权重到值向量
        # --------------------------
        # 加权求和：(batch_size, num_heads, 1, total_nodes) × (batch_size, num_heads, total_nodes, head_dim)
        # 结果形状 → (batch_size, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # --------------------------
        # 7. 拼接多头结果并投影
        # --------------------------
        # 转置后 → (batch_size, 1, num_heads, head_dim)
        # 拼接多头 → (batch_size, 1, embed_dim)（因为num_heads×head_dim=embed_dim）
        # 输出投影并压缩维度 → (batch_size, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        graph_output = self.out_proj(attn_output).squeeze(1)  # 压缩中间的维度1

        # --------------------------
        # 8. 残差连接（稳定训练）
        # --------------------------
        # 将原始图嵌入从2*embed_dim投影到embed_dim，与注意力输出维度匹配后相加
        residual = self.residual_proj(graph_hidden)
        new_graph_hidden = graph_output + residual  # 残差连接

        return new_graph_hidden

class Net_DQN(nn.Module):
    def __init__(self, time_dim, input_dim, hidden_dim, pos_hidden_dim, output_dim, pos_output_dim, action_dim,
                 edge_dim, heads, feed_forward_hidden=512, debug=False):
        """
        hidden_dim == output_dim
        debug: 是否开启调试模式（输出中间层形状）
        """
        super().__init__()
        self.debug = debug
        self.heads = heads
        self.output_dim = output_dim

        # 边缘特征嵌入层
        self.init_embed = nn.Linear(edge_dim, hidden_dim * heads)

        # GAT+FFN模块配置（使用列表存储参数，便于循环构建）
        self.gat_blocks = nn.ModuleList()
        gat_dims = [
            (input_dim, hidden_dim),  # 第一层：输入维度 -> hidden_dim
            (hidden_dim * heads, hidden_dim),  # 第二层：上一层输出 -> hidden_dim
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, output_dim)  # 第三层：上一层输出 -> output_dim
        ]

        for i, (in_dim, out_dim) in enumerate(gat_dims):
            self.gat_blocks.append(nn.ModuleDict({
                'gat': TransformerConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    concat=True,
                    dropout=0,
                    edge_dim=hidden_dim * heads,
                    beta=(i >= 0)
                ),
                'ff': nn.Linear(out_dim * heads, out_dim * heads),
                'bn': nn.LayerNorm(out_dim * heads),
                'ffn': nn.Sequential(
                    nn.Linear(out_dim * heads, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, out_dim * heads),
                ),
                'bn_ffn': nn.LayerNorm(out_dim * heads)
            }))

        # 解码器和输出层
        self.decoder = MultiHeadAttention(embed_dim=output_dim * heads, num_heads=heads)

        # 主全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(output_dim * heads * 2),
            nn.Linear(output_dim * heads * 2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 残差子模块
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # 残差连接层
        self.residual = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_dim * heads * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )


    def reset_parameters(self):
        """统一初始化所有可学习参数"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, node_attr, edge_indices, edge_attrs, agent_pos, num_nodes):
        # 调试信息：输入形状
        if self.debug:
            print(f"Input node_attr shape: {node_attr.shape}")
            print(f"Input edge_attrs shape: {edge_attrs.shape}")

        # 边缘特征嵌入
        edge_emb = self.init_embed(edge_attrs)
        x = node_attr  # 初始化节点特征

        # 循环执行GAT块（替代手动逐层调用）
        for i, block in enumerate(self.gat_blocks):
            # GAT层 + 前馈
            if i == 0:
                gat_out = block['gat'](x, edge_indices, edge_emb)
                ff_out = block['ff'](gat_out)
                x = block['bn'](ff_out)

                # FFN
                ffn_out = block['ffn'](x)
                x = block['bn_ffn'](ffn_out)
            else:
                residual = x
                gat_out = block['gat'](x, edge_indices, edge_emb)
                ff_out = block['ff'](gat_out)
                x = block['bn'](ff_out + residual)

                # FFN + 残差
                residual = x
                ffn_out = block['ffn'](x)
                x = block['bn_ffn'](ffn_out + residual)
            if self.debug:
                print(f"GAT Block {i + 1} output shape: {x.shape}")

        # 提取代理特征
        agent_features = x[agent_pos]
        if self.debug:
            print(f"Agent features shape: {agent_features.shape}")

        # 解码器处理
        new_graph_hidden = self.decoder(agent_features, x, num_nodes)
        if self.debug:
            print(f"Decoder output shape: {new_graph_hidden.shape}")

        # 拼接特征并输出
        concatenated = torch.cat([agent_features, new_graph_hidden], dim=1)
        main_out = self.fc(concatenated)
        res_out = self.residual(concatenated)
        output = main_out + res_out

        return output, 0