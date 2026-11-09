import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, epsilon=1e-6):
        """
        Prioritized experience replay buffer.

        Args:
            buffer_size: maximum number of stored transitions
            batch_size: default sampling batch size
            alpha: priority exponent in (0, 1], controls the influence of priorities
            epsilon: small constant keeping priorities non-zero
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # typically around 0.6
        self.epsilon = epsilon  # prevents zero priorities

        self.memory = np.empty((buffer_size, 8), dtype=object)  # transition tuples including action masks
        self.priorities = np.zeros(buffer_size)  # priority of each slot
        self._size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, nodes_size,
            action_mask=None, next_action_mask=None):
        """
        Add a transition to the buffer.
        A new transition receives the current maximum priority (or the initial value).
        action_mask / next_action_mask: action masks (True means the action is valid)
        """
        # Store the transition tuple
        self.memory[self.position] = (state, action, reward, next_state, done, nodes_size,
                                      action_mask, next_action_mask)

        # Initial priority
        if self._size == 0:
            # First transition: use the initial value
            self.priorities[self.position] = (self.epsilon) ** self.alpha
        else:
            # Later transitions inherit the current maximum so they are sampled at least once
            max_priority = self.priorities[:self._size].max()
            self.priorities[self.position] = max_priority

        # Advance the write position and update the size
        self.position = (self.position + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        """
        Sample transitions according to their priorities.
        Returns: samples, sampled indices, sampling probabilities
        """
        current_batch_size = batch_size or self.batch_size

        # Valid priorities (stored entries only)
        valid_priorities = self.priorities[:self._size]

        # Sampling probabilities (guarding against division by zero)
        if valid_priorities.sum() == 0:
            probabilities = np.ones(self._size) / self._size
        else:
            probabilities = valid_priorities / valid_priorities.sum()

        # Weighted random sampling
        indices = np.random.choice(self._size, current_batch_size, p=probabilities)

        # Fetch the corresponding transitions
        samples = self.memory[indices].tolist()
        for i in range(len(samples)):
            state, action, reward, next_state, done, nodes_size, action_mask, next_action_mask = samples[i]
            # Clone tensors so later in-place updates do not modify stored states
            state1 = [t.clone() for t in state] if isinstance(state, list) else state.clone()
            next_state1 = [t.clone() for t in next_state] if isinstance(next_state, list) else next_state.clone()
            samples[i] = (state1, action, reward, next_state1, done, nodes_size,
                         list(action_mask) if action_mask is not None else None,
                         list(next_action_mask) if next_action_mask is not None else None)

        # Return samples, indices and probabilities for priority updates and importance weights
        return samples, indices, probabilities[indices]

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the given transitions from their TD errors.
        Args:
            indices: indices of the transitions to update
            td_errors: corresponding TD errors
        """
        # New priority: (|td_error| + epsilon)^alpha
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        # Write the new priorities
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self._size