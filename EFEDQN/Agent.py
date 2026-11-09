import torch
import torch.nn as nn
import torch.optim as optim
from GAT import GAT_DQN
import numpy as np
from Replaybuffer import ReplayBuffer

def batch_reshape(node_attr, edge_indices, edge_attrs, agent_pos,node_size):
    """
          Forward pass that batches multiple graphs.

          Args:
          - node_attr: node features, shape [batch_size, num_nodes, input_dim]
          - edge_indices: edge indices per graph, list of [2, num_edges] tensors
          - edge_attrs: edge features, shape [batch_size, num_edges, edge_dim]
          - agent_pos: agent position index per graph, shape [batch_size]
          """
    batch_size = len(node_attr)

    # Flatten node features
    # Offset edge_index and flatten edge_attr
    x_flat = []
    edge_index_list = []
    edge_attr_list = []
    agent_pos = torch.stack(agent_pos).squeeze(1)
    edge_offset = 0

    for i in range(batch_size):
        x_flat.append(node_attr[i].clone())
        # Shift node indices of edge_index
        ei = edge_indices[i].clone()
        ei += edge_offset
        edge_index_list.append(ei)

        # Collect edge features
        edge_attr_list.append(edge_attrs[i].clone())

        agent_pos[i] += edge_offset

        edge_offset += node_size[i]

    # Concatenate edge_index and edge_attr of all graphs
    x_flat1 = torch.cat(x_flat, dim=0) if edge_attrs is not None else None
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attrs is not None else None

    return x_flat1,edge_index,edge_attr,agent_pos


class DQNAgent:
    def __init__(self,action_size):
        self.gamma = 0.99#discount factor
        self.lr = 0.00001#learning rate
        self.epsilon = 1#exploration rate
        self.buffer_size = 1000000#replay buffer capacity
        self.batch_size = 16#mini-batch size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet = GAT_DQN(0, 2, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target = GAT_DQN(0, 2, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.lr)
        self.tau = 0.01
        self.current_step = 0

    def _sanitize_mask(self, action_mask):
        """Normalise the action mask: None allows every action; an all-forbidden mask also allows every action (avoids deadlock; the environment penalises instead)"""
        if action_mask is None:
            return [True] * self.action_size
        mask = [bool(m) for m in action_mask]
        if not any(mask):
            return [True] * self.action_size
        return mask

    def _masked_argmax(self, qs, mask):
        """Greedy action selection: set Q of forbidden actions to -inf, then take the argmax"""
        qs = qs.view(-1).clone()
        if not all(mask):
            invalid = torch.tensor([not m for m in mask], dtype=torch.bool, device=qs.device)
            qs[invalid] = float('-inf')
        return torch.argmax(qs).item()

    def _forward_single(self, state, nodes_size):
        """Single-state forward pass, returns (qs, weight)"""
        t_state = [state[0].to(self.device), state[1].to(self.device),
                   state[2].to(self.device), state[3].to(self.device)]
        nodes_size = torch.tensor([nodes_size], dtype=torch.int32).to(self.device)
        return self.qnet(t_state[0], t_state[1], t_state[2], t_state[3], nodes_size)

    def get_action(self, state, nodes_size, is_test=False, action_mask=None):
        # Mask: exclude actions that violate an edge time window or have no road
        mask = self._sanitize_mask(action_mask)
        with torch.no_grad():
            if is_test:
                qs, weight = self._forward_single(state, nodes_size)
                return self._masked_argmax(qs, mask), weight
            else:
                if np.random.rand() < self.epsilon:
                    # Exploration: random choice among legal actions only
                    valid_actions = [a for a, m in enumerate(mask) if m]
                    action = int(np.random.choice(valid_actions))
                    return action, 0
                else:
                    qs, weight = self._forward_single(state, nodes_size)
                    return self._masked_argmax(qs, mask), weight

    def get_actions(self, states, nodes_sizes, action_masks=None):
        """
        Batched action selection: one forward pass yields actions for several environments,
        so all transitions in a step come from the same network version.
        Each environment explores independently; exploring environments skip the forward pass.
        action_masks: per-environment action masks (True = legal); both exploration and
        greedy selection only consider legal actions.
        Returns a list of actions (int).
        """
        batch = len(states)
        masks = [None] * batch if action_masks is None else action_masks
        actions = [0] * batch
        explore = np.random.rand(batch) < self.epsilon
        for i in range(batch):
            mask = self._sanitize_mask(masks[i])
            if explore[i]:
                # Exploration: random choice among legal actions only
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
        g_masks = [self._sanitize_mask(masks[i]) for i in greedy_idx]
        with torch.no_grad():
            x0, x1, x2, x3 = batch_reshape(s0, s1, s2, s3, sizes)
            t_sizes = torch.tensor(sizes, dtype=torch.int32).to(self.device)
            qs, _ = self.qnet(x0.to(self.device), x1.to(self.device), x2.to(self.device), x3.to(self.device), t_sizes)
            # Greedy selection is masked: forbidden actions get Q = -inf before the argmax
            mask_t = self._mask_tensor(g_masks, len(greedy_idx))
            greedy_actions = qs.masked_fill(~mask_t, float('-inf')).argmax(dim=1).tolist()
        for k, i in enumerate(greedy_idx):
            actions[i] = int(greedy_actions[k])
        return actions

    def _mask_tensor(self, masks, batch_size):
        """Convert batched masks to a boolean tensor [batch_size, action_size] on the device; None or all-forbidden rows allow every action"""
        t = torch.ones((batch_size, self.action_size), dtype=torch.bool)
        for i, m in enumerate(masks):
            if m is None:
                continue
            row = [bool(x) for x in m]
            if any(row):
                t[i] = torch.tensor(row, dtype=torch.bool)
        return t.to(self.device)

    def store(self, state, action, reward, next_state, done, nodes_size,
              action_mask=None, next_action_mask=None):
        """Store one transition in the replay buffer without a gradient update"""
        self.replay_buffer.add(state, action, reward, next_state, done, nodes_size,
                               action_mask, next_action_mask)

    def update(self, state, action, reward, next_state, done, nodes_size,
               action_mask=None, next_action_mask=None):
        """Store one transition and run a single gradient update"""
        self.store(state, action, reward, next_state, done, nodes_size,
                   action_mask, next_action_mask)
        return self.train_step()

    def train_step(self):
        """Sample from the replay buffer and run one gradient update; returns the loss (None if the buffer holds fewer than one batch)"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        # Sample transitions (returns samples, indices and probabilities)
        experiences, buffer_indices, probs = self.replay_buffer.sample()
        states1, actions, rewards, next_states1, dones, nodes_sizes, action_masks, next_action_masks = zip(*experiences)

        states = [[],[],[],[]]
        for s in states1:
            states[0].append(s[0])
            states[1].append(s[1])
            states[2].append(s[2])
            states[3].append(s[3])

        actions = torch.tensor(list(actions), dtype=torch.int).to(self.device)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(list(dones), dtype=torch.float32).to(self.device)
        next_states = [[],[],[],[]]
        for s in next_states1:
            next_states[0].append(s[0])
            next_states[1].append(s[1])
            next_states[2].append(s[2])
            next_states[3].append(s[3])

        t_nodes_sizes = torch.tensor(list(nodes_sizes), dtype=torch.int32).to(self.device)

        # Current Q values
        states[0], states[1], states[2], states[3] = batch_reshape(states[0],states[1],states[2],states[3],nodes_sizes)
        states[0] = states[0].to(self.device)
        states[1] = states[1].to(self.device)
        states[2] = states[2].to(self.device)
        states[3] = states[3].to(self.device)
        qs = self.qnet(states[0],states[1],states[2],states[3],t_nodes_sizes)[0]
        batch_size = qs.size(0)
        indices = torch.arange(batch_size).to(self.device)  # use torch.arange instead of np.arange
        q = qs[indices, actions]

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Next-state action mask: neither action selection nor evaluation may pick an
            # action that violates a time window or has no road
            next_mask = self._mask_tensor(next_action_masks, len(experiences))
            # The online network selects the next action (masked)
            next_states[0], next_states[1], next_states[2], next_states[3] = batch_reshape(next_states[0],next_states[1],next_states[2],next_states[3],nodes_sizes)
            next_states[0] = next_states[0].to(self.device)
            next_states[1] = next_states[1].to(self.device)
            next_states[2] = next_states[2].to(self.device)
            next_states[3] = next_states[3].to(self.device)
            next_qs_online = self.qnet(next_states[0],next_states[1],next_states[2],next_states[3],t_nodes_sizes)[0]
            next_actions = next_qs_online.masked_fill(~next_mask, -1e9).argmax(1, keepdim=True)
            # The target network evaluates that action (also masked)
            next_qs = self.qnet_target(next_states[0],next_states[1],next_states[2],next_states[3],t_nodes_sizes)[0]
            next_q = next_qs.masked_fill(~next_mask, -1e9).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        # TD errors (used to update priorities)
        td_errors = (target - q).detach().cpu().numpy()  # to numpy for the priority update

        # Update the priorities of the sampled transitions
        self.replay_buffer.update_priorities(buffer_indices, td_errors)

        # Loss with importance-sampling correction
        # Importance-sampling weights
        beta = 0.4 + 0.6 * (self.current_step / 10000) # annealed from 0.4 to 1.0 during training
        weights = (self.batch_size * probs) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        loss = (weights * (q - target) ** 2).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_qnet(self):
        with torch.no_grad():
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def reset_qnet(self):
        self.qnet = GAT_DQN(0, 2, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.epsilon = 1
        self.current_step = 0
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.lr)