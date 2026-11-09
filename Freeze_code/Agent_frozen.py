# -*- coding: utf-8 -*-
"""Agent_frozen.py — DQN agent with frozen graph encoding (Freeze_code).

Builds on DQNAgent (PER + Double DQN + soft update tau=0.01) with:
1. Q network and target network instantiated as GAT_DQN_Frozen (GAT_Frozen.py);
2. a six-element state: [s0_node, s0_edge_idx, s0_edge_attr, pos, elapsed,
   goal_mask], where the s0* tensors are the episode-start snapshot used as
   encoder input and consistent with the inference-time cached encoding, pos is
   the current position ([pos] tensor), elapsed is the time in hours since the
   encoding timestamp (0-d tensor), and goal_mask is the current 0/1 mask of
   remaining targets ([N] tensor, shrinking with env.goal_states every step and
   consumed by the goal-aware decoder);
3. action masking: exploration samples only among feasible actions; greedy
   selection and TD targets are masked as well.
"""

import os
import sys

import numpy as np
import torch
import torch.optim as optim

# Depend on Agent / Replaybuffer / GAT from the EFEDQN directory
_EFEDQN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EFEDQN")
if _EFEDQN_DIR not in sys.path:
    sys.path.insert(0, _EFEDQN_DIR)

from Agent import DQNAgent, batch_reshape  # noqa: E402
from Replaybuffer import ReplayBuffer      # noqa: E402
from GAT_Frozen import GAT_DQN_Frozen      # noqa: E402


class FrozenDQNAgent(DQNAgent):
    """DQN agent with frozen encoding: GAT_DQN_Frozen network, state carries elapsed and goal_mask."""

    def _new_qnet(self):
        return GAT_DQN_Frozen(0, 2, 32, 0, 32, 0, 4, 22, 8).to(self.device)

    def __init__(self, action_size):
        # Hyperparameters as in DQNAgent.__init__; only the networks and optimizer are built here
        self.gamma = 0.99
        self.lr = 0.00001
        self.epsilon = 1
        self.buffer_size = 1000000
        self.batch_size = 16
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = self._new_qnet()
        self.qnet_target = self._new_qnet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.lr)
        self.tau = 0.01
        self.current_step = 0

    def reset_qnet(self):
        self.qnet = self._new_qnet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.epsilon = 1
        self.current_step = 0
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.lr)

    # Single-state forward pass: pass elapsed / goal_mask to the model
    def _forward_single(self, state, nodes_size):
        t_state = [state[0].to(self.device), state[1].to(self.device),
                   state[2].to(self.device), state[3].to(self.device)]
        nodes_size = torch.tensor([nodes_size], dtype=torch.int32).to(self.device)
        elapsed = state[4] if len(state) > 4 else None
        goal_mask = state[5].to(self.device) if len(state) > 5 else None
        return self.qnet(t_state[0], t_state[1], t_state[2], t_state[3],
                         nodes_size, elapsed=elapsed, goal_mask=goal_mask)

    # Action selection. Masking follows DQNAgent: exploration samples only
    # feasible actions, greedy sets forbidden Q values to -inf before argmax;
    # _sanitize_mask / _masked_argmax are inherited.
    def get_action(self, state, nodes_size, is_test=False, action_mask=None):
        mask = self._sanitize_mask(action_mask)
        with torch.no_grad():
            if is_test:
                qs, weight = self._forward_single(state, nodes_size)
                return self._masked_argmax(qs, mask), weight
            if np.random.rand() < self.epsilon:
                # Exploration: sample uniformly among feasible actions
                valid_actions = [a for a, m in enumerate(mask) if m]
                return int(np.random.choice(valid_actions)), 0
            qs, weight = self._forward_single(state, nodes_size)
            return self._masked_argmax(qs, mask), weight

    def get_actions(self, states, nodes_sizes, action_masks=None):
        """Batched action selection used by the training loop: greedy slots are
        forwarded in one batch (single network version), exploring slots sample
        only feasible actions. Returns a list of int actions."""
        batch = len(states)
        masks = [None] * batch if action_masks is None else action_masks
        actions = [0] * batch
        explore = np.random.rand(batch) < self.epsilon
        for i in range(batch):
            mask = self._sanitize_mask(masks[i])
            if explore[i]:
                # Exploration: sample uniformly among feasible actions
                valid_actions = [a for a, m in enumerate(mask) if m]
                actions[i] = int(np.random.choice(valid_actions))
        greedy_idx = [i for i in range(batch) if not explore[i]]
        if not greedy_idx:
            return actions
        s0 = [states[i][0] for i in greedy_idx]
        s1 = [states[i][1] for i in greedy_idx]
        s2 = [states[i][2] for i in greedy_idx]
        s3 = [states[i][3] for i in greedy_idx]
        sizes = [nodes_sizes[i] for i in greedy_idx]
        elapsed = [float(states[i][4]) for i in greedy_idx]
        goals = [states[i][5] for i in greedy_idx]
        g_masks = [self._sanitize_mask(masks[i]) for i in greedy_idx]
        with torch.no_grad():
            x0, x1, x2, x3 = batch_reshape(s0, s1, s2, s3, sizes)
            t_sizes = torch.tensor(sizes, dtype=torch.int32).to(self.device)
            t_elapsed = torch.tensor(elapsed, dtype=torch.float32).to(self.device)
            # Remaining-target mask ordered like the node concatenation in batch_reshape, shape [total]
            t_goals = torch.cat(goals, dim=0).to(self.device)
            qs, _ = self.qnet(x0.to(self.device), x1.to(self.device),
                              x2.to(self.device), x3.to(self.device),
                              t_sizes, elapsed=t_elapsed, goal_mask=t_goals)
            # Apply the mask for greedy selection: set forbidden Q values to -inf, then argmax
            mask_t = self._mask_tensor(g_masks, len(greedy_idx))
            greedy_actions = qs.masked_fill(~mask_t, float('-inf')).argmax(dim=1).tolist()
        for k, i in enumerate(greedy_idx):
            actions[i] = int(greedy_actions[k])
        return actions

    # Training update (PER + Double DQN). States are six-element vectors that
    # carry elapsed / goal_mask; both the action selection and the evaluation in
    # the TD target apply next_action_mask.
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        experiences, buffer_indices, probs = self.replay_buffer.sample()
        (states1, actions, rewards, next_states1, dones, nodes_sizes,
         action_masks, next_action_masks) = zip(*experiences)

        states = [[], [], [], [], [], []]
        for s in states1:
            for k in range(6):
                states[k].append(s[k])
        next_states = [[], [], [], [], [], []]
        for s in next_states1:
            for k in range(6):
                next_states[k].append(s[k])

        actions = torch.tensor(list(actions), dtype=torch.int).to(self.device)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(dones), dtype=torch.float32).to(self.device)
        t_nodes_sizes = torch.tensor(list(nodes_sizes), dtype=torch.int32).to(self.device)
        t_elapsed = torch.tensor([float(e) for e in states[4]],
                                 dtype=torch.float32).to(self.device)
        t_next_elapsed = torch.tensor([float(e) for e in next_states[4]],
                                      dtype=torch.float32).to(self.device)
        # Remaining-target mask ordered like the node concatenation in batch_reshape, shape [total]
        t_goals = torch.cat(states[5], dim=0).to(self.device)
        t_next_goals = torch.cat(next_states[5], dim=0).to(self.device)

        # Current Q values; the encoder input is each episode's start snapshot s0
        states[0], states[1], states[2], states[3] = batch_reshape(
            states[0], states[1], states[2], states[3], nodes_sizes)
        states[0] = states[0].to(self.device)
        states[1] = states[1].to(self.device)
        states[2] = states[2].to(self.device)
        states[3] = states[3].to(self.device)
        qs = self.qnet(states[0], states[1], states[2], states[3],
                       t_nodes_sizes, elapsed=t_elapsed, goal_mask=t_goals)[0]
        batch_size = qs.size(0)
        indices = torch.arange(batch_size).to(self.device)
        q = qs[indices, actions]

        # Target Q values (Double DQN). The mask applies to both the action
        # selection and the evaluation, so time-window violations and missing
        # directions are excluded.
        with torch.no_grad():
            next_mask = self._mask_tensor(next_action_masks, len(experiences))
            next_states[0], next_states[1], next_states[2], next_states[3] = \
                batch_reshape(next_states[0], next_states[1],
                              next_states[2], next_states[3], nodes_sizes)
            next_states[0] = next_states[0].to(self.device)
            next_states[1] = next_states[1].to(self.device)
            next_states[2] = next_states[2].to(self.device)
            next_states[3] = next_states[3].to(self.device)
            # Online network selects the next action (after masking)
            next_qs_online = self.qnet(next_states[0], next_states[1],
                                       next_states[2], next_states[3],
                                       t_nodes_sizes, elapsed=t_next_elapsed,
                                       goal_mask=t_next_goals)[0]
            next_actions = next_qs_online.masked_fill(~next_mask, -1e9).argmax(1, keepdim=True)
            # Target network evaluates the selected action (masked likewise)
            next_qs = self.qnet_target(next_states[0], next_states[1],
                                       next_states[2], next_states[3],
                                       t_nodes_sizes, elapsed=t_next_elapsed,
                                       goal_mask=t_next_goals)[0]
            next_q = next_qs.masked_fill(~next_mask, -1e9).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        # Update priorities with TD errors and compute importance-sampling weights (PER)
        td_errors = (target - q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(buffer_indices, td_errors)

        beta = 0.4 + 0.6 * (self.current_step / 10000)
        weights = (self.batch_size * probs) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        loss = (weights * (q - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
