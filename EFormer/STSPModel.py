"""EFormer model for S-TSP-TVRA.

Three-stage architecture:
    Precoder (MatNet-style mixed-score attention, edges -> temporary node embeddings)
      -> parallel encoders: GraphEncoder (ResidualGatedGCN, edge features)
         and NodeEncoder (Transformer)
      -> Decoder (two-stream context attention)

Inputs are sparse directed graphs described by edge feature tensors of shape
(B,N,N,8) rather than complete-graph Euclidean distances, so
edges_values_embedding is a Linear(8) layer. The Precoder cost matrix is the
normalized edge-length matrix (1.0 sentinel for absent edges, 0 on the diagonal).
The NodeEncoder input is the Precoder node feature plus a node-flag embedding
(start, target), which makes the model target-aware for the Steiner objective
while keeping edges as the main signal. On top of the first/last node queries,
the Decoder adds q_target, a masked mean pool over the remaining target nodes,
and q_time, a projection of the multi-period sin/cos encoding of the current time.
The first action is fixed to the start node (depot).

Encoding is one-shot, with a dynamic action mask applied during decoding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn_layers import ResidualGatedGCNLayer, MLP


class STSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.node_encoder = STSP_NodeEncoder(**model_params)
        self.graph_encoder = STSP_GraphEncoder(**model_params)
        self.node_encodingblock = Node_EncodingBlock(**model_params)   # Precoder
        self.decoder = STSP_Decoder(**model_params)

        self.encoded_nodes = None
        self.encoded_graph = None
        self.node_feature = None
        self.start = None

    def pre_forward(self, reset_state):
        # Precoder: edges (cost matrix) -> temporary node embeddings
        self.node_feature = self.node_encodingblock(reset_state.cost_mat)
        # shape: (batch, node, embedding)

        # Parallel encoders: graph encoder (edge features + GCN) and node encoder
        # (Precoder features + node flags)
        self.encoded_graph = self.graph_encoder(reset_state.edge_feats,
                                                reset_state.x_edges,
                                                self.node_feature)
        self.encoded_nodes = self.node_encoder(self.node_feature,
                                               reset_state.node_flags)
        self.start = reset_state.start  # (B,)
        self.decoder.set_kv(self.encoded_graph, self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            # First step: every rollout starts at the depot
            selected = self.start[:, None].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=selected.device)
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            encoded_first_graph = _get_encoding(self.encoded_graph, selected)
            self.decoder.set_q1(encoded_first_node, encoded_first_graph)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            encoded_last_graph = _get_encoding(self.encoded_graph, state.current_node)
            probs = self.decoder(encoded_last_node, encoded_last_graph,
                                 ninf_mask=state.ninf_mask,
                                 remaining=state.remaining,
                                 cur_time=state.time)
            # shape: (batch, pomo, problem)
            if self.training or self.model_params["eval_type"] == "softmax":
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1) \
                        .multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


# Precoder (cost-matrix input)
class Node_EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params["embedding_dim"]
        self.row_encoding_block = EncodingBlock(**model_params)

    def forward(self, cost_mat):
        # cost_mat.shape: (batch, node, node), normalized edge lengths
        batch_size = cost_mat.size(0)
        problem_size = cost_mat.size(1)

        # [A] row_emb = 0
        row_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim),
                              device=cost_mat.device)
        # [B] col_emb = random one-hot seeds (MatNet style)
        col_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim),
                              device=cost_mat.device)

        if problem_size <= self.embedding_dim:
            seed_cnt = problem_size
            rand = torch.rand(batch_size, seed_cnt, device=cost_mat.device)
            batch_rand_perm = rand.argsort(dim=1)
        else:
            seed_cnt = self.embedding_dim
            rand = torch.rand(batch_size, seed_cnt, device=cost_mat.device)
            batch_rand_perm = rand.argsort(dim=1)
            reps = problem_size // self.embedding_dim + 1
            batch_rand_perm = batch_rand_perm.repeat(1, reps)[:, :problem_size]

        rand_idx = batch_rand_perm[:, :problem_size]
        b_idx = torch.arange(batch_size, device=cost_mat.device)[:, None] \
            .expand(batch_size, problem_size)
        n_idx = torch.arange(problem_size, device=cost_mat.device)[None, :] \
            .expand(batch_size, problem_size)
        col_emb[b_idx, n_idx, rand_idx] = 1

        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        return row_emb_out  # (batch, node, embedding)


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

        self.add_n_normalization_1 = Add_And_Batch_Normalization(**model_params)
        self.add_n_normalization_2 = Add_And_Batch_Normalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3


# DOUBLE ENCODER
class STSP_GraphEncoder(nn.Module):
    """ResidualGatedGCN graph encoder over 8-channel edge feature inputs."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params["embedding_dim"]
        self.hidden_dim = self.model_params["hidden_dim"]
        self.num_layers = self.model_params["GCN_dim"]
        self.mlp_layers = self.model_params["mlp_layers"]
        self.aggregation = self.model_params["aggregation"]

        self.edge_feat_dim = self.model_params.get("edge_feat_dim", 8)
        self.edges_values_embedding = nn.Linear(self.edge_feat_dim, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(3, self.hidden_dim // 2)  # 0 = no edge, 1 = directed edge, 2 = diagonal

        gcn_layers = []
        for _ in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.mlp_nodes = MLP(self.hidden_dim, self.embedding_dim, self.mlp_layers)

    def forward(self, edge_feats, x_edges, node_feature):
        # edge_feats: (B,N,N,edge_feat_dim), x_edges: (B,N,N) long, node_feature: (B,N,emb)
        x = node_feature
        e_vals = self.edges_values_embedding(edge_feats)   # (B,N,N,H/2)
        e_tags = self.edges_embedding(x_edges)             # (B,N,N,H/2)
        e = torch.cat((e_vals, e_tags), dim=3)             # (B,N,N,H)

        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)

        y_pred_nodes = self.mlp_nodes(x)                   # (B,N,emb)
        return y_pred_nodes


class STSP_NodeEncoder(nn.Module):
    """Transformer node encoder over Precoder features plus node-flag embeddings."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        self.node_flag_embedding = nn.Linear(2, embedding_dim)  # [start, target]
        self.layers = nn.ModuleList([EncoderLayer(**model_params)
                                     for _ in range(self.model_params["encoder_layer_num"])])

    def forward(self, node_feature, node_flags):
        # node_feature: (B,N,emb), node_flags: (B,N,2)
        embedded_input = node_feature + self.node_flag_embedding(node_flags)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

        self.addAndNormalization1 = Add_And_Batch_Normalization(**model_params)
        self.addAndNormalization2 = Add_And_Batch_Normalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3


# DECODER
class STSP_Decoder(nn.Module):
    """Two-stream context decoder with pooled target and time contexts."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        # [Stream 1: node encoding]
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

        # [Stream 2: graph encoding]
        self.Wq_first_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k_e = None
        self.v_e = None
        self.single_head_key_e = None
        self.q_first_e = None

        # [Pooled remaining-target context (one per stream)]
        self.Wq_tgt = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_tgt_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # [Projection of the multi-period sin/cos time encoding (4 periods, 8 dims)]
        self.Wq_time = nn.Linear(8, head_num * qkv_dim, bias=False)

    def set_kv(self, encoded_graph, encoded_nodes):
        head_num = self.model_params["head_num"]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

        self.k_e = reshape_by_heads(self.Wk_e(encoded_graph), head_num=head_num)
        self.v_e = reshape_by_heads(self.Wv_e(encoded_graph), head_num=head_num)
        self.single_head_key_e = encoded_graph.transpose(1, 2)

    def set_q1(self, encoded_q1, encoded_first_graph):
        head_num = self.model_params["head_num"]
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        self.q_first_e = reshape_by_heads(self.Wq_first_e(encoded_first_graph), head_num=head_num)

    def forward(self, encoded_last_node, encoded_last_graph, ninf_mask,
                remaining=None, cur_time=None):
        # encoded_last_node/graph: (batch, pomo, embedding)
        # ninf_mask: (batch, pomo, problem); remaining: (batch, pomo, N) bool
        # cur_time: (batch, pomo) in hours
        head_num = self.model_params["head_num"]
        batch_size, pomo_size = encoded_last_node.size(0), encoded_last_node.size(1)

        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q_last_e = reshape_by_heads(self.Wq_last_e(encoded_last_graph), head_num=head_num)

        # Masked mean pool over the remaining targets (Steiner termination awareness)
        # Pooling uses the encoded node/graph embeddings (transposed single_head_key),
        # giving one context per stream
        if remaining is not None:
            rem = remaining.float().unsqueeze(-1)                     # (B,P,N,1)
            cnt = rem.sum(dim=2).clamp(min=1.0)                       # (B,P,1)
            enc_nodes = self.single_head_key.transpose(1, 2)          # (B,N,emb)
            enc_graph = self.single_head_key_e.transpose(1, 2)        # (B,N,emb)
            tgt_ctx_node = (enc_nodes.unsqueeze(1) * rem).sum(dim=2) / cnt   # (B,P,emb)
            tgt_ctx_graph = (enc_graph.unsqueeze(1) * rem).sum(dim=2) / cnt  # (B,P,emb)
            q_tgt = reshape_by_heads(self.Wq_tgt(tgt_ctx_node), head_num=head_num)
            q_tgt_e = reshape_by_heads(self.Wq_tgt_e(tgt_ctx_graph), head_num=head_num)
        else:
            q_tgt = q_tgt_e = 0.0

        # Multi-period sin/cos context of the current time (time awareness)
        if cur_time is not None:
            t = cur_time.unsqueeze(-1)                                # (B,P,1)
            periods = (0.5, 2.0, 8.0, 24.0)
            sincos = torch.cat([torch.cat([torch.sin(2 * math.pi * t / T),
                                           torch.cos(2 * math.pi * t / T)], dim=-1)
                                for T in periods], dim=-1)            # (B,P,8)
            q_time = reshape_by_heads(self.Wq_time(sincos.to(q_last.dtype)), head_num=head_num)
        else:
            q_time = 0.0

        q_x = self.q_first + q_last + q_tgt
        q_e = self.q_first_e + q_last_e + q_tgt_e
        q = q_x + q_e + q_time

        out_concat1 = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out_concat2 = multi_head_attention(q, self.k_e, self.v_e, rank3_ninf_mask=ninf_mask)

        mh_atten_out1 = self.multi_head_combine(out_concat1)
        mh_atten_out2 = self.multi_head_combine2(out_concat2)
        mh_atten_out = mh_atten_out1 + mh_atten_out2
        single_head_key = self.single_head_key + self.single_head_key_e

        score = torch.matmul(mh_atten_out, single_head_key)
        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]
        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs


# NN SUB CLASS / FUNCTIONS
def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=q.dtype, device=q.device))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


class Add_And_Batch_Normalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input1, input2):
        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)
        added = input1 + input2
        normalized = self.norm(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)
        return back_trans


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        head_num = model_params["head_num"]
        ms_hidden_dim = model_params["ms_hidden_dim"]
        mix1_init = model_params["ms_layer1_init"]
        mix2_init = model_params["ms_layer2_init"]

        mix1_weight = torch.distributions.Uniform(low=-mix1_init, high=mix1_init) \
            .sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-mix1_init, high=mix1_init) \
            .sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)

        mix2_weight = torch.distributions.Uniform(low=-mix2_init, high=mix2_init) \
            .sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-mix2_init, high=mix2_init) \
            .sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)

    def forward(self, q, k, v, cost_mat):
        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        sqrt_qkv_dim = self.model_params["sqrt_qkv_dim"]

        dot_product = torch.matmul(q, k.transpose(2, 3))
        dot_product_score = dot_product / sqrt_qkv_dim
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        two_scores_transposed = two_scores.transpose(1, 2)
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        ms1_activated = F.relu(ms1)
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        mixed_scores = ms2.transpose(1, 2)
        mixed_scores = mixed_scores.squeeze(4)
        weights = nn.Softmax(dim=3)(mixed_scores)
        out = torch.matmul(weights, v)
        out_transposed = out.transpose(1, 2)
        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        return out_concat
