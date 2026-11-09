import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, epsilon=1e-6):
        """
        Experience replay buffer with uniform sampling.

        :param buffer_size: maximum number of stored transitions
        :param batch_size: default sample size
        :param alpha: priority exponent in (0, 1), controlling the priority scale
        :param epsilon: small constant that keeps priorities above zero
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # typically around 0.6
        self.epsilon = epsilon  # keeps priorities above zero

        self.memory = np.empty((buffer_size, 8), dtype=object)  # transition tuples (with action masks)
        self.priorities = np.zeros(buffer_size)  # per-transition priorities
        self._size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, nodes_size,
            action_mask=None, next_action_mask=None):
        """
        Add one transition to the buffer.
        The new transition receives the current maximum priority, or the initial
        value while the buffer is still empty.
        action_mask / next_action_mask: action masks (True means the action is valid)
        """
        # Store the transition tuple
        self.memory[self.position] = (state, action, reward, next_state, done, nodes_size,
                                      action_mask, next_action_mask)

        # Initial priority of the new transition
        if self._size == 0:
            # first transition: use the initial value
            self.priorities[self.position] = (self.epsilon) ** self.alpha
        else:
            # later transitions: use the current maximum so they can be sampled
            max_priority = self.priorities[:self._size].max()
            self.priorities[self.position] = max_priority

        # Advance the write position and the size
        self.position = (self.position + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        """
        Sample transitions uniformly from the buffer.

        :return: sampled transitions, their indices, and their sampling probabilities
        """
        current_batch_size = batch_size or self.batch_size

        # Uniform sampling: every stored transition has equal probability.
        probabilities = np.ones(self._size) / self._size

        # Random draw with the given probabilities
        indices = np.random.choice(self._size, current_batch_size, p=probabilities)

        # Gather the sampled transitions
        samples = self.memory[indices].tolist()
        for i in range(len(samples)):
            state, action, reward, next_state, done, nodes_size, action_mask, next_action_mask = samples[i]
            # Clone the states so that later in-place updates cannot leak into the buffer
            state1 = [t.clone() for t in state] if isinstance(state, list) else state.clone()
            next_state1 = [t.clone() for t in next_state] if isinstance(next_state, list) else next_state.clone()
            samples[i] = (state1, action, reward, next_state1, done, nodes_size,
                         list(action_mask) if action_mask is not None else None,
                         list(next_action_mask) if next_action_mask is not None else None)

        # Return samples, indices and probabilities for the priority and weight updates
        return samples, indices, probabilities[indices]

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the given transitions from their TD errors.

        :param indices: indices of the transitions to update
        :param td_errors: corresponding TD errors
        """
        # New priority: (|td_error| + epsilon) ** alpha
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        # Write the new priorities
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self._size