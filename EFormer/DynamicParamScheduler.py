"""Dynamic scheduler for batch_size and max_steps.

Rule: max_steps = clamp(round(avg_steps * safety_factor)).
Constraints (highest to lowest priority; lower-priority constraints are
relaxed on conflict):
  1. max_steps_lb <= max_steps <= max_steps_ub
  2. max_steps * batch_size <= product_cap
  3. batch_size <= batch_ub
batch_size takes the largest value that satisfies constraints 1 and 2 first.
"""


class DynamicParamScheduler:
    def __init__(self,
                 product_cap=4800,
                 max_steps_lb=100,
                 max_steps_ub=1500,
                 batch_ub=64,
                 init_max_steps=100,
                 init_batch_size=48,
                 safety_factor=1.25):
        self.product_cap = product_cap
        self.max_steps_lb = max_steps_lb
        self.max_steps_ub = max_steps_ub
        self.batch_ub = batch_ub
        self.safety_factor = safety_factor

        self.max_steps = self._clamp_ms(init_max_steps)
        if self._valid(init_batch_size, self.max_steps):
            self.batch_size = init_batch_size
        else:
            self.batch_size = self._batch_for(self.max_steps)

    def _clamp_ms(self, ms):
        return int(min(max(ms, self.max_steps_lb), self.max_steps_ub))

    def _batch_for(self, ms):
        # Largest batch under the product cap and the batch upper bound, floor of 1
        return max(1, min(self.batch_ub, self.product_cap // ms))

    def _valid(self, bs, ms):
        return (self.max_steps_lb <= ms <= self.max_steps_ub
                and 1 <= bs <= self.batch_ub
                and bs * ms <= self.product_cap)

    def update(self, avg_steps):
        """Called after each training batch (O(1)); returns (new_batch_size, new_max_steps)."""
        new_ms = self._clamp_ms(int(round(avg_steps * self.safety_factor)))
        new_bs = self._batch_for(new_ms)
        assert self._valid(new_bs, new_ms), "dynamic param constraint violated"
        self.max_steps, self.batch_size = new_ms, new_bs
        return new_bs, new_ms
