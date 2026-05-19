import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, epsilon=1e-6):
        """
        优先级经验回放池
        :param buffer_size: 最大存储容量
        :param batch_size: 默认采样批量大小
        :param alpha: 优先级指数（0~1），控制优先级的影响程度
        :param epsilon: 小常数，避免优先级为0
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # 通常设为0.6左右
        self.epsilon = epsilon  # 防止优先级为0

        self.memory = np.empty((buffer_size, 6), dtype=object)  # 存储经验元组
        self.priorities = np.zeros(buffer_size)  # 存储优先级
        self._size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, nodes_size):
        """
        添加经验到回放池
        新经验初始优先级设为当前最大优先级或初始值
        """
        # 存储经验元组
        self.memory[self.position] = (state, action, reward, next_state, done,nodes_size)

        # 为新经验设置初始优先级
        if self._size == 0:
            # 第一个经验，使用初始值
            self.priorities[self.position] = (self.epsilon) ** self.alpha
        else:
            # 后续经验使用当前最大优先级，确保新经验能被采样到
            max_priority = self.priorities[:self._size].max()
            self.priorities[self.position] = max_priority

        # 更新写入位置和缓冲区大小
        self.position = (self.position + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        """
        根据优先级采样经验
        :return: 采样样本, 采样索引, 采样概率
        """
        current_batch_size = batch_size or self.batch_size

        # 计算有效优先级（仅考虑已存储的经验）
        valid_priorities = self.priorities[:self._size]

        # 计算采样概率（防止除以零）
        if valid_priorities.sum() == 0:
            probabilities = np.ones(self._size) / self._size
        else:
            probabilities = valid_priorities / valid_priorities.sum()

        # 加权随机采样
        indices = np.random.choice(self._size, current_batch_size, p=probabilities)

        # 获取对应经验样本
        samples = self.memory[indices].tolist()
        for i in range(len(samples)):
            state, action, reward, next_state, done,nodes_size = samples[i]
            # 克隆状态（如果使用PyTorch张量）
            state1 = [t.clone() for t in state] if isinstance(state, list) else state.clone()
            next_state1 = [t.clone() for t in next_state] if isinstance(next_state, list) else next_state.clone()
            samples[i] = (state1, action, reward, next_state1, done,nodes_size)

        # 返回样本、索引和概率，用于后续更新优先级和计算重要性采样权重
        return samples, indices, probabilities[indices]

    def update_priorities(self, indices, td_errors):
        """
        根据TD误差更新指定经验的优先级
        :param indices: 要更新的经验索引
        :param td_errors: 对应的TD误差
        """
        # 计算新的优先级：(|td_error| + epsilon)^alpha
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        # 更新优先级数组
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self._size