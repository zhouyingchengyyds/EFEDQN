import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, epsilon=1e-6):
        """
        Prioritized experience replay buffer
        :param buffer_size: maximum number of stored transitions
        :param batch_size: default sample batch size
        :param alpha: priority exponent (0~1) controlling how strongly priorities matter
        :param epsilon: small constant that keeps priorities strictly positive
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # typically around 0.6
        self.epsilon = epsilon  # avoids zero priority

        self.memory = np.empty((buffer_size, 8), dtype=object)  # stored transitions (with action masks)
        self.priorities = np.zeros(buffer_size)  # priority of each slot
        self._size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, nodes_size,
            action_mask=None, next_action_mask=None):
        """
        Add one transition to the buffer.
        A new transition receives the current maximum priority or the initial value.
        action_mask / next_action_mask: action masks (True = legal action)
        """
        # Store the transition
        self.memory[self.position] = (state, action, reward, next_state, done, nodes_size,
                                      action_mask, next_action_mask)

        # Initial priority of the new transition
        if self._size == 0:
            # first transition: use the initial value
            self.priorities[self.position] = (self.epsilon) ** self.alpha
        else:
            # later transitions use the current maximum so they can be sampled
            max_priority = self.priorities[:self._size].max()
            self.priorities[self.position] = max_priority

        # Advance the write position and the size
        self.position = (self.position + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        """
        Sample transitions according to their priorities
        :return: samples, sampled indices, sampling probabilities
        """
        current_batch_size = batch_size or self.batch_size

        # Valid priorities (stored transitions only)
        valid_priorities = self.priorities[:self._size]

        # Sampling probabilities (guard against division by zero)
        if valid_priorities.sum() == 0:
            probabilities = np.ones(self._size) / self._size
        else:
            probabilities = valid_priorities / valid_priorities.sum()

        # Weighted random sampling
        indices = np.random.choice(self._size, current_batch_size, p=probabilities)

        # Collect the sampled transitions
        samples = self.memory[indices].tolist()
        for i in range(len(samples)):
            state, action, reward, next_state, done, nodes_size, action_mask, next_action_mask = samples[i]
            # Clone the states (PyTorch tensors)
            state1 = [t.clone() for t in state] if isinstance(state, list) else state.clone()
            next_state1 = [t.clone() for t in next_state] if isinstance(next_state, list) else next_state.clone()
            samples[i] = (state1, action, reward, next_state1, done, nodes_size,
                         list(action_mask) if action_mask is not None else None,
                         list(next_action_mask) if next_action_mask is not None else None)

        # Samples, indices and probabilities feed the priority update and the IS weights
        return samples, indices, probabilities[indices]

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the given transitions from their TD errors
        :param indices: indices of the transitions to update
        :param td_errors: their TD errors
        """
        # New priority: (|td_error| + epsilon)^alpha
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        # Write back the priorities
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self._size