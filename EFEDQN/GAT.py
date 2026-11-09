import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
from torch_scatter import scatter_mean


class CrossAttention(nn.Module):
    def __init__(self, temporal_dim,spacial_dim):
        super().__init__()
        self.get_weight = nn.Sequential(
            nn.Linear(temporal_dim + spacial_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.value_temporal = nn.Linear(temporal_dim,temporal_dim)

    def forward(self,temporal_features,spatial_features):
        # Spatial features gate the temporal features
        input = torch.cat([temporal_features,spatial_features], dim=1)
        weight = self.get_weight(input)
        temporal_attended = self.value_temporal(temporal_features)
        attended_temporal = temporal_attended * weight

        return attended_temporal,weight


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention between a graph embedding and its node features.
    Inputs: graph embedding (batch, embed_dim), node features (total_nodes, embed_dim)
    grouped by graph.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # node embedding dimension
        self.num_heads = num_heads  # number of attention heads
        self.head_dim = embed_dim // num_heads  # dimension per head

        # The embedding dimension must be divisible by the number of heads
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"

        # Linear projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # query projection of the graph embedding
        self.k_proj = nn.Linear(embed_dim, embed_dim)      # key projection of the node features
        self.v_proj = nn.Linear(embed_dim, embed_dim)      # value projection of the node features
        self.out_proj = nn.Linear(embed_dim, embed_dim)    # output projection after concatenating the heads
        self.residual_proj = nn.Linear(embed_dim, embed_dim)  # residual projection (dimension matching)

    def forward(self, graph_hidden, reshaped, nodes_sizes):
        """
        Args:
            graph_hidden: graph-level embedding, shape (batch_size, embed_dim)
            reshaped: node features of all graphs concatenated, shape (total_nodes, embed_dim)
            nodes_sizes: node count of each graph, e.g. [3,5,2] for three graphs with 3/5/2 nodes

        Returns:
            new_graph_hidden: updated graph embedding, shape (batch_size, embed_dim)
        """
        # Basic dimensions
        batch_size = graph_hidden.size(0)  # number of graphs
        total_nodes, embed_dim = reshaped.size()  # total nodes and node embedding dimension

        # 1. Build the node-to-graph assignment mask so that attention cannot cross graphs
        # Graph index of each node, shape (total_nodes,)
        # e.g. nodes_sizes=[2,3] -> indices [0,0,1,1,1]
        graph_indices = torch.cat([
            torch.full((size,), i, device=reshaped.device)  # tag every node of graph i with i
            for i, size in enumerate(nodes_sizes)
        ], dim=0)

        # One-hot encoding (total_nodes, batch_size): 1 marks membership
        graph_mask = F.one_hot(graph_indices, num_classes=batch_size).float()
        # Reshape to (batch_size, 1, total_nodes) for broadcasting over the head axis,
        # which is kept at 1 and later matched against num_heads
        mask = graph_mask.t().unsqueeze(1)  # transpose, then add the axis

        # 2. Query: the graph embedding
        # (batch_size, embed_dim) -> (batch_size, 1, num_heads, head_dim)
        # -> transposed -> (batch_size, num_heads, 1, head_dim)
        # The axis of size 1 holds the single query vector of each graph
        q = self.q_proj(graph_hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Keys and values: the node features
        # (total_nodes, embed_dim) -> (total_nodes, num_heads, head_dim)
        # -> transposed -> (num_heads, total_nodes, head_dim)
        k = self.k_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        # Add the batch axis: (num_heads, total_nodes, head_dim)
        # -> (1, num_heads, total_nodes, head_dim) -> (batch_size, num_heads, total_nodes, head_dim)
        # expand (not repeat) avoids copying the data
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # -1 keeps this axis unchanged
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 4. Attention scores
        # q: (batch_size, num_heads, 1, head_dim)
        # k transposed: (batch_size, num_heads, head_dim, total_nodes)
        # scores: (batch_size, num_heads, 1, total_nodes)
        # Scaling keeps the scores small enough for a stable softmax
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 5. Mask out nodes of other graphs
        # mask: (batch_size, 1, 1, total_nodes), aligned with attn_scores
        # Foreign nodes get score -1e9 so their softmax weight is 0
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        # Softmax over nodes: the weights of each graph sum to 1
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 6. Weighted sum of the value vectors
        # (batch_size, num_heads, 1, total_nodes) x (batch_size, num_heads, total_nodes, head_dim)
        # -> (batch_size, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 7. Concatenate the heads and project
        # transpose -> (batch_size, 1, num_heads, head_dim)
        # concat -> (batch_size, 1, embed_dim) since num_heads * head_dim = embed_dim
        # output projection -> (batch_size, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        graph_output = self.out_proj(attn_output).squeeze(1)  # drop the axis of size 1

        # 8. Residual connection
        residual = self.residual_proj(graph_hidden)
        new_graph_hidden = graph_output + residual

        return new_graph_hidden

class GAT_DQN(nn.Module):
    def __init__(self, time_dim, input_dim, hidden_dim, pos_hidden_dim, output_dim, pos_output_dim, action_dim,
                 edge_dim, heads, feed_forward_hidden=512, debug=False):
        """
        hidden_dim == output_dim
        debug: print intermediate shapes when True
        """
        super().__init__()
        self.debug = debug
        self.heads = heads
        self.output_dim = output_dim

        # Edge feature embedding layer
        self.init_embed = nn.Linear(edge_dim, hidden_dim * heads)

        # GAT + FFN blocks (parameters kept in a list for the construction loop)
        self.gat_blocks = nn.ModuleList()
        gat_dims = [
            (input_dim, hidden_dim),  # layer 1: input_dim -> hidden_dim
            (hidden_dim * heads, hidden_dim),  # subsequent: hidden_dim*heads -> hidden_dim
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, output_dim)  # last layer: -> output_dim
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

        # Decoder and output layer
        self.decoder = MultiHeadAttention(embed_dim=output_dim * heads, num_heads=heads)

        # Main fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(output_dim * heads * 2),
            nn.Linear(output_dim * heads * 2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            # Residual sub-block
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

        # Residual shortcut
        self.residual = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_dim * heads * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )


    def reset_parameters(self):
        """Reset all learnable parameters"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, node_attr, edge_indices, edge_attrs, agent_pos, num_nodes):
        # Debug: input shapes
        if self.debug:
            print(f"Input node_attr shape: {node_attr.shape}")
            print(f"Input edge_attrs shape: {edge_attrs.shape}")

        # Edge feature embedding
        edge_emb = self.init_embed(edge_attrs)
        x = node_attr  # node features

        # GAT + FFN blocks
        for i, block in enumerate(self.gat_blocks):
            # GAT layer + feed-forward
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

                # FFN + residual
                residual = x
                ffn_out = block['ffn'](x)
                x = block['bn_ffn'](ffn_out + residual)
            if self.debug:
                print(f"GAT Block {i + 1} output shape: {x.shape}")

        # Agent features
        agent_features = x[agent_pos]
        if self.debug:
            print(f"Agent features shape: {agent_features.shape}")

        # Decoder
        new_graph_hidden = self.decoder(agent_features, x, num_nodes)
        if self.debug:
            print(f"Decoder output shape: {new_graph_hidden.shape}")

        # Concatenate and apply both output heads
        concatenated = torch.cat([agent_features, new_graph_hidden], dim=1)
        main_out = self.fc(concatenated)
        res_out = self.residual(concatenated)
        output = main_out + res_out

        return output, 0