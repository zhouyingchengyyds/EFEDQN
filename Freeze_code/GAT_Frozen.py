# -*- coding: utf-8 -*-
"""GAT_Frozen.py — graph-attention DQN with frozen graph encoding (Freeze_code).

The model implements three behaviors:

1. Frozen encoding: during inference (eval mode without grad) each episode
   encodes the graph once on its first step (init_embed + GAT blocks) and
   caches the node embeddings under a static key; later steps reuse the cache.
   The static key covers the start flag (node feature column 0), the edge index
   and the static edge features (slot one-hot columns 0-3, distance column 20,
   duration column 21), and excludes the dynamic part (remaining-target flag
   column 1, window encoding relative to the current time, columns 4-19). The
   encoder input is therefore pinned to the snapshot taken at the start of the
   episode (or of a resumed run), which matches the training semantics where
   the start snapshot is fed directly. In training mode every call runs a full
   forward pass and nothing is cached, so gradients stay safe.

2. Decoder time channel: forward also accepts elapsed (hours, modulo 24), the
   time since the encoding timestamp, encoded as four sin/cos features
   (periods 1440 and 480 minutes; the 24 h period pins the position on the
   ring) and concatenated into the inputs of both the fc and the residual
   head. elapsed=None is treated as 0, so 5-argument calls still work. The
   network can then anticipate future window openings; the dynamic action mask
   is applied at the agent / rollout layer, and forward itself is mask-agnostic.

3. Goal-aware decoder: since the encoder input is pinned to the start snapshot,
   the remaining-target flag in node feature column 1 keeps the initial full
   set for the whole episode, so the model cannot tell which targets were
   already visited. The decoder is therefore GoalAwareMultiHeadAttention, whose
   forward also accepts goal_mask ([total_nodes] 0/1 mask, 1 = still a
   remaining target) that is concatenated into Q (the agent node flag) and into
   K/V (each node's flag) and shrinks with env.goal_states every step;
   goal_mask=None falls back to node_attr column 1.

Saving the whole model with torch.save pickles the module name GAT_Frozen, so
this file must stay in the Freeze_code directory; loaders inject that directory
through extra_sys_path before unpickling.
"""

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT import GAT_DQN, MultiHeadAttention


class GoalAwareMultiHeadAttention(MultiHeadAttention):
    """Goal-aware decoder: Q/K/V/residual projections take embed_dim + 1 (node embedding + remaining-target flag).

    The attention forward follows MultiHeadAttention with two changes:
    1. the four projection layers accept embed_dim+1 inputs and still output
       embed_dim;
    2. forward is overridden because the final view would otherwise use the
       embed_dim inferred from the input (here embed_dim+1), which breaks the
       shapes; self.embed_dim (the true output width) is used instead.
    """

    def __init__(self, embed_dim, num_heads, flag_dim=1):
        super().__init__(embed_dim, num_heads)
        in_dim = embed_dim + flag_dim
        self.q_proj = nn.Linear(in_dim, embed_dim)
        self.k_proj = nn.Linear(in_dim, embed_dim)
        self.v_proj = nn.Linear(in_dim, embed_dim)
        self.residual_proj = nn.Linear(in_dim, embed_dim)

    def forward(self, graph_hidden, reshaped, nodes_sizes):
        """Same computation as MultiHeadAttention.forward, except that the final
        view uses self.embed_dim (the output width) instead of the width
        inferred from the input, which carries the extra flag column.

        graph_hidden: [batch, embed_dim+1] (agent node embedding + remaining-target flag)
        reshaped:     [total_nodes, embed_dim+1] (node embeddings + per-node remaining-target flags)
        """
        batch_size = graph_hidden.size(0)
        total_nodes, _ = reshaped.size()

        # 1. Graph-membership mask: restrict each query to its own graph
        graph_indices = torch.cat([
            torch.full((int(size),), i, device=reshaped.device)
            for i, size in enumerate(nodes_sizes)
        ], dim=0)
        graph_mask = F.one_hot(graph_indices, num_classes=batch_size).float()
        mask = graph_mask.t().unsqueeze(1)

        # 2. Query: agent node embedding plus its remaining-target flag
        q = self.q_proj(graph_hidden).view(
            batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Keys/values: every node embedding plus its own remaining-target flag
        k = self.k_proj(reshaped).view(
            total_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(reshaped).view(
            total_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 4. Scaled attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 5. Membership masking and softmax
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 6. Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)

        # 7. Head merge and output projection (self.embed_dim is required here; the input width is embed_dim+1)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.embed_dim)
        graph_output = self.out_proj(attn_output).squeeze(1)

        # 8. Residual connection
        residual = self.residual_proj(graph_hidden)
        new_graph_hidden = graph_output + residual

        return new_graph_hidden


class GAT_DQN_Frozen(GAT_DQN):
    """GAT_DQN with frozen encoding: one encoding at the episode start, an
    elapsed time channel in the decoder and a per-step remaining-target mask."""

    # Read by the rollout side (exp_common._qnet_forward): forward takes elapsed
    needs_time = True
    # Read by the rollout side: forward takes the current remaining-target mask
    needs_goals = True

    # sin/cos periods for elapsed, in minutes: 24 h (1440) pins the global ring position, 8 h (480) adds a mid-scale
    TIME_PERIODS_MIN = (1440.0, 480.0)
    TIME_FEAT_DIM = 2 * len(TIME_PERIODS_MIN)  # = 4

    def __init__(self, time_dim, input_dim, hidden_dim, pos_hidden_dim,
                 output_dim, pos_output_dim, action_dim, edge_dim, heads,
                 feed_forward_hidden=512, debug=False):
        super().__init__(time_dim, input_dim, hidden_dim, pos_hidden_dim,
                         output_dim, pos_output_dim, action_dim, edge_dim,
                         heads, feed_forward_hidden, debug)
        # Only the two output heads are redefined: four elapsed-time features
        # extend the parent input widths, while the GAT encoder and decoder keep
        # the parent structure and parameter counts
        in_dim = output_dim * heads * 2 + self.TIME_FEAT_DIM
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
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
        self.residual = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # Goal-aware decoder: with the encoder input pinned to the start
        # snapshot, the set of unvisited targets can only enter the network as a
        # per-step dynamic input on the decoder side
        self.decoder = GoalAwareMultiHeadAttention(output_dim * heads, heads)
        # Inference encoding cache (single entry: latest static key -> node embeddings)
        self._cache_key = None
        self._cache_x = None

    # Cache management
    def clear_cache(self):
        self._cache_key = None
        self._cache_x = None

    def train(self, mode=True):
        self.clear_cache()  # drop the cache on mode switches so encodings do not leak across modes
        return super().train(mode)

    @staticmethod
    def _static_key(node_attr, edge_indices, edge_attrs):
        """Cache key over the static part of an episode; dynamic columns stay out of the key."""
        h = hashlib.md5()
        parts = (
            node_attr[:, 0:1],                    # start flag (static within an episode)
            edge_indices,                         # edge structure
            edge_attrs[:, [0, 1, 2, 3, 20, 21]],  # slot one-hot / distance / duration
        )
        for t in parts:
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
            h.update(b"|")
        return h.digest()

    # Encoding / time features
    def _encode(self, node_attr, edge_indices, edge_attrs):
        """Full graph encoding (init_embed + the GAT blocks), identical to the first half of the parent forward."""
        edge_emb = self.init_embed(edge_attrs)
        x = node_attr
        for i, block in enumerate(self.gat_blocks):
            if i == 0:
                gat_out = block['gat'](x, edge_indices, edge_emb)
                ff_out = block['ff'](gat_out)
                x = block['bn'](ff_out)
                ffn_out = block['ffn'](x)
                x = block['bn_ffn'](ffn_out)
            else:
                residual = x
                gat_out = block['gat'](x, edge_indices, edge_emb)
                ff_out = block['ff'](gat_out)
                x = block['bn'](ff_out + residual)
                residual = x
                ffn_out = block['ffn'](x)
                x = block['bn_ffn'](ffn_out + residual)
        return x

    def _time_features(self, elapsed, batch, device, dtype):
        """Map elapsed (hours, scalar / 0-d / 1-d tensor) to [batch, 4] sin/cos features; None means 0."""
        if elapsed is None:
            e = torch.zeros(batch, device=device, dtype=dtype)
        elif torch.is_tensor(elapsed):
            e = elapsed.to(device=device, dtype=dtype).reshape(-1)
            if e.numel() == 1 and batch > 1:
                e = e.expand(batch)
            elif e.numel() != batch:
                raise ValueError(
                    "elapsed 数量(%d)与 batch(%d)不符" % (e.numel(), batch))
        else:
            e = torch.full((batch,), float(elapsed), device=device, dtype=dtype)
        minute_val = e * 60.0  # hours -> minutes
        feats = []
        for T in self.TIME_PERIODS_MIN:
            rad = 2.0 * math.pi * minute_val / T
            feats.append(torch.sin(rad))
            feats.append(torch.cos(rad))
        return torch.stack(feats, dim=-1)

    # Forward pass
    def forward(self, node_attr, edge_indices, edge_attrs, agent_pos, num_nodes,
                elapsed=None, goal_mask=None):
        """Forward pass (signature compatible with the base encoder, with optional
        elapsed / goal_mask).

        Inference (eval + no_grad): a static-key hit reuses the cached encoding.
        Training mode or grad-enabled calls: full encoding every time, nothing
        is cached. Batched graphs (concatenated by batch_reshape) and single
        graphs (rollout) are both supported.
        goal_mask: [total_nodes] 0/1 mask (1 = remaining target), concatenated
                   into decoder Q/K/V; None falls back to node_attr column 1,
                   which under frozen encoding equals the initial full set.
        """
        use_cache = (not self.training) and (not torch.is_grad_enabled())
        x = None
        if use_cache:
            key = self._static_key(node_attr, edge_indices, edge_attrs)
            if (self._cache_x is not None and key == self._cache_key
                    and self._cache_x.shape[0] == node_attr.shape[0]
                    and self._cache_x.device == node_attr.device):
                x = self._cache_x
        if x is None:
            x = self._encode(node_attr, edge_indices, edge_attrs)
            if use_cache:
                self._cache_key = key
                self._cache_x = x

        if goal_mask is None:
            # Fallback: use the target-flag column of the input node features (the initial full set here)
            goal_mask = node_attr[:, 1].to(x.dtype)
        if goal_mask.dim() == 1:
            goal_mask = goal_mask.unsqueeze(-1)              # [total, 1]
        agent_features = x[agent_pos]                        # [batch, embed]
        q_in = torch.cat([agent_features, goal_mask[agent_pos]], dim=1)
        x_dec = torch.cat([x, goal_mask], dim=1)             # [total, embed+1]
        new_graph_hidden = self.decoder(q_in, x_dec, num_nodes)
        concatenated = torch.cat([agent_features, new_graph_hidden], dim=1)
        t_feat = self._time_features(elapsed, concatenated.shape[0],
                                     concatenated.device, concatenated.dtype)
        concatenated = torch.cat([concatenated, t_feat], dim=1)
        main_out = self.fc(concatenated)
        res_out = self.residual(concatenated)
        return main_out + res_out, 0
