import torch
import torch.optim as optim
from Net import Net_DQN
import numpy as np
from Replaybuffer import ReplayBuffer

def batch_reshape(node_attr, edge_indices, edge_attrs, agent_pos,node_size):
    """
          前向传播方法，处理批量图数据

          参数:
          - node_attr: 节点特征，形状为 [batch_size, num_nodes, input_dim]
          - edge_indices: 每个图的边索引，列表，每个元素形状为 [2, num_edges]
          - edge_attrs: 边特征，形状为 [batch_size, num_edges, edge_dim]
          - agent_pos: 每个图中代理的位置索引，形状为 [batch_size]
          """
    batch_size = len(node_attr)

    # 扁平化节点特征
    # 调整edge_index并扁平化edge_attr
    x_flat = []
    edge_index_list = []
    edge_attr_list = []
    agent_pos = torch.stack(agent_pos).squeeze(1)
    edge_offset = 0

    for i in range(batch_size):
        x_flat.append(node_attr[i].clone())
        # 调整edge_index的节点索引
        ei = edge_indices[i].clone()
        ei += edge_offset
        edge_index_list.append(ei)

        # 收集边特征
        edge_attr_list.append(edge_attrs[i].clone())

        agent_pos[i] += edge_offset

        edge_offset += node_size[i]

    # 合并所有图的edge_index和edge_attr
    x_flat1 = torch.cat(x_flat, dim=0) if edge_attrs is not None else None
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attrs is not None else None

    return x_flat1,edge_index,edge_attr,agent_pos


class DQNAgent:
    def __init__(self,action_size):
        self.gamma = 0.99#折扣率
        self.lr = 0.00001#学习率
        self.epsilon = 1#探索率
        self.buffer_size = 1000000#经验回放池大小
        self.batch_size = 16#小批量
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet = Net_DQN(0, 4, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target = Net_DQN(0, 4, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.tau = 0.01
        self.current_step = 0

    def get_action(self, state, nodes_size, is_test=False):
        with torch.no_grad():
            if is_test:
                t_state = [None,None,None,None]
                t_state[0] = state[0].to(self.device)
                t_state[1] = state[1].to(self.device)
                t_state[2] = state[2].to(self.device)
                t_state[3] = state[3].to(self.device)
                nodes_size = torch.tensor([nodes_size], dtype=torch.int32).to(self.device)
                qs,weight = self.qnet(t_state[0],t_state[1],t_state[2],t_state[3],nodes_size)
                return torch.argmax(qs).item(),weight
            else:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_size)
                    return action,0
                else:
                    t_state = [None, None, None, None]
                    t_state[0] = state[0].to(self.device)
                    t_state[1] = state[1].to(self.device)
                    t_state[2] = state[2].to(self.device)
                    t_state[3] = state[3].to(self.device)
                    nodes_size = torch.tensor([nodes_size], dtype=torch.int32).to(self.device)
                    qs,weight = self.qnet(t_state[0],t_state[1],t_state[2],t_state[3],nodes_size)
                    return torch.argmax(qs).item(),weight

    def update(self, state, action, reward, next_state, done, nodes_size):
        self.replay_buffer.add(state, action, reward, next_state, done,nodes_size)
        if len(self.replay_buffer) < self.batch_size:
            return None
        # 2. 采样经验（现在会返回样本、索引和概率）
        experiences, buffer_indices, probs = self.replay_buffer.sample()
        states1, actions, rewards, next_states1, dones,nodes_sizes = zip(*experiences)

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

        #计算当前Q值
        states[0], states[1], states[2], states[3] = batch_reshape(states[0],states[1],states[2],states[3],nodes_sizes)
        states[0] = states[0].to(self.device)
        states[1] = states[1].to(self.device)
        states[2] = states[2].to(self.device)
        states[3] = states[3].to(self.device)
        qs = self.qnet(states[0],states[1],states[2],states[3],t_nodes_sizes)[0]
        batch_size = qs.size(0)
        indices = torch.arange(batch_size).to(self.device)  # 使用 torch.arange 而不是 np.arange
        # 提取 qs[indices, actions]
        q = qs[indices, actions]

        #计算目标Q值（Double DQN）
        with torch.no_grad():
            # 使用在线网络选择下一个状态的最优动作
            next_states[0], next_states[1], next_states[2], next_states[3] = batch_reshape(next_states[0],next_states[1],next_states[2],next_states[3],nodes_sizes)
            next_states[0] = next_states[0].to(self.device)
            next_states[1] = next_states[1].to(self.device)
            next_states[2] = next_states[2].to(self.device)
            next_states[3] = next_states[3].to(self.device)
            next_actions = self.qnet(next_states[0],next_states[1],next_states[2],next_states[3],t_nodes_sizes)[0].argmax(1, keepdim=True)
            # 使用目标网络评估该动作的Q值
            next_qs = self.qnet_target(next_states[0],next_states[1],next_states[2],next_states[3],t_nodes_sizes)[0]
            next_q = next_qs.gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        # 6. 计算TD误差（用于更新优先级）
        td_errors = (target - q).detach().cpu().numpy()  # 转为numpy数组用于更新回放池

        # 7. 更新回放池中对应经验的优先级
        self.replay_buffer.update_priorities(buffer_indices, td_errors)

        # 8. 计算损失函数（可选：添加重要性采样权重修正偏差）
        # 计算重要性采样权重 (IS Weights)
        beta = 0.4 + 0.6 * (self.current_step / 10000) # 通常初始值为0.4，随训练逐渐增加到1.0
        weights = (self.batch_size * probs) ** (-beta)
        weights = weights / weights.max()  # 归一化
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # 使用重要性采样权重计算加权损失
        loss = (weights * (q - target) ** 2).mean()  # 加权MSE损失

        # 9. 梯度下降（保持不变）
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
        self.qnet = Net_DQN(0, 4, 32, 0, 32, 0, 4, 22, 8).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.epsilon = 1
        self.current_step = 0
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)