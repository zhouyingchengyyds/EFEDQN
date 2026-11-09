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
        # Attention of the spatial features over the temporal features
        input = torch.cat([temporal_features,spatial_features], dim=1)
        weight = self.get_weight(input)
        temporal_attended = self.value_temporal(temporal_features)
        attended_temporal = temporal_attended * weight

        return attended_temporal,weight


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention between graph-level embeddings and node features.
    Input: graph embeddings (batch, 2*embed_dim), node features (total_nodes, embed_dim)
    grouped by graph.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # node embedding size
        self.num_heads = num_heads  # number of attention heads
        self.head_dim = embed_dim // num_heads  # size of one head

        # The embedding size must split evenly into heads
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # query projection of the graph embeddings
        self.k_proj = nn.Linear(embed_dim, embed_dim)      # key projection of the node features
        self.v_proj = nn.Linear(embed_dim, embed_dim)      # value projection of the node features
        self.out_proj = nn.Linear(embed_dim, embed_dim)    # output projection of the concatenated heads
        self.residual_proj = nn.Linear(embed_dim, embed_dim)  # residual projection

    def forward(self, graph_hidden, reshaped, nodes_sizes):
        """
        Args:
            graph_hidden: graph-level embeddings, shape (batch_size, 2*embed_dim)
            reshaped: node features of all graphs concatenated, shape (total_nodes, embed_dim)
            nodes_sizes: number of nodes per graph, e.g. [3,5,2]

        Returns:
            new_graph_hidden: graph embedding updated with node information,
            shape (batch_size, embed_dim)
        """
        # Base dimensions
        batch_size = graph_hidden.size(0)  # number of graphs
        total_nodes, embed_dim = reshaped.size()  # total nodes and node embedding size

        # 1. Node-to-graph mask, which keeps attention inside each graph
        # Graph index of every node, shape (total_nodes,)
        # e.g. nodes_sizes=[2,3] -> [0,0,1,1,1]
        graph_indices = torch.cat([
            torch.full((size,), i, device=reshaped.device)  # tag the nodes of graph i with index i
            for i, size in enumerate(nodes_sizes)
        ], dim=0)

        # One-hot encoding: (total_nodes, batch_size), 1 marks membership of the graph
        graph_mask = F.one_hot(graph_indices, num_classes=batch_size).float()
        # Reshape to (batch_size, 1, total_nodes) so that it broadcasts over the head dimension
        mask = graph_mask.t().unsqueeze(1)  # transpose and add a dimension

        # 2. Queries from the graph embeddings
        # (batch_size, 2*embed_dim) -> (batch_size, embed_dim)
        # -> (batch_size, 1, num_heads, head_dim) -> (batch_size, num_heads, 1, head_dim)
        # The dimension of size 1 means one query per graph
        q = self.q_proj(graph_hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Keys and values from the node features
        # (total_nodes, embed_dim) -> (total_nodes, num_heads, head_dim)
        # -> (num_heads, total_nodes, head_dim)
        k = self.k_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(reshaped).view(total_nodes, self.num_heads, self.head_dim).transpose(0, 1)

        # Broadcast to the batch: (num_heads, total_nodes, head_dim)
        # -> (batch_size, num_heads, total_nodes, head_dim)
        # expand avoids copying the data
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # -1 keeps that dimension unchanged
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 4. Attention scores
        # q (batch_size, num_heads, 1, head_dim) x k^T (batch_size, num_heads, head_dim, total_nodes)
        # -> (batch_size, num_heads, 1, total_nodes)
        # Scaling keeps the scores small enough for a stable softmax
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 5. Mask out the nodes of the other graphs
        # mask (batch_size, 1, 1, total_nodes) matches the score shape after unsqueeze
        # Forbidden nodes get -1e9, so their softmax weight is 0
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        # Softmax over the node dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 6. Weighted sum of the values
        # (batch_size, num_heads, 1, total_nodes) x (batch_size, num_heads, total_nodes, head_dim)
        # -> (batch_size, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 7. Concatenate the heads and project
        # -> (batch_size, 1, num_heads, head_dim) -> (batch_size, 1, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        graph_output = self.out_proj(attn_output).squeeze(1)  # drop the temporary dimension

        # 8. Residual connection
        residual = self.residual_proj(graph_hidden)
        new_graph_hidden = graph_output + residual  # residual connection

        return new_graph_hidden


class GAT_DQN(nn.Module):
    def __init__(self, time_dim, input_dim, hidden_dim, pos_hidden_dim, output_dim, pos_output_dim, action_dim,
                 edge_dim, heads, feed_forward_hidden=512, debug=False):
        """
        hidden_dim == output_dim
        debug: print the shapes of the intermediate tensors
        """
        super().__init__()
        self.debug = debug
        self.heads = heads
        self.output_dim = output_dim

        # Edge feature embedding
        self.init_embed = nn.Linear(edge_dim, hidden_dim * heads)

        # Layer sizes of the GAT + FFN blocks
        self.gat_blocks = nn.ModuleList()
        gat_dims = [
            (input_dim, hidden_dim),  # first block: input_dim -> hidden_dim
            (hidden_dim * heads, hidden_dim),  # next blocks: previous output -> hidden_dim
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, hidden_dim),
            (hidden_dim * heads, output_dim)  # last block: previous output -> output_dim
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

        # Decoder and output heads
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
            # extra MLP block
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

        # Residual head
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
        # Input shapes for debugging
        if self.debug:
            print(f"Input node_attr shape: {node_attr.shape}")
            print(f"Input edge_attrs shape: {edge_attrs.shape}")

        # Edge feature embedding
        edge_emb = self.init_embed(edge_attrs)
        x = node_attr  # node features

        # Apply the GAT blocks in sequence
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

        # Concatenate and combine the two heads
        concatenated = torch.cat([agent_features, new_graph_hidden], dim=1)
        main_out = self.fc(concatenated)
        res_out = self.residual(concatenated)
        output = main_out + res_out

        return output, 0