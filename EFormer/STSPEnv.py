"""Vectorized S-TSP-TVRA environment with POMO-style rollouts.

1. Action space: node-level selection restricted to the outgoing neighbours of
   the current node; non-neighbour nodes are masked with -inf.
2. Dynamic action masking: edges whose time window (closed interval, may wrap
   past midnight) contains the current time are masked; waiting is not allowed.
3. Steiner semantics: non-target nodes may be revisited or traversed; an episode
   succeeds once all targets are visited and the agent returns to the depot.
   It fails when the step budget is exceeded or no action remains available.
4. Time evolution: t = (t + speed / 60) % 24.
5. Reward: -(total edge length + 1000 * failure flag), used for POMO REINFORCE
   training.
"""

from dataclasses import dataclass

import torch


@dataclass
class Reset_State:
    # Encoder inputs produced by STSPProblemDef.instance_to_tensors
    x_edges: torch.Tensor
    edge_feats: torch.Tensor
    cost_mat: torch.Tensor
    node_flags: torch.Tensor
    adj: torch.Tensor
    length: torch.Tensor
    travel_time: torch.Tensor
    win_s: torch.Tensor
    win_e: torch.Tensor
    start: torch.Tensor
    start_time: torch.Tensor
    target_mask: torch.Tensor


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    current_node: torch.Tensor = None      # (B, P)
    ninf_mask: torch.Tensor = None         # (B, P, N)
    remaining: torch.Tensor = None         # (B, P, N) remaining unvisited targets
    time: torch.Tensor = None              # (B, P) current time (hours)


class STSPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.pomo_size = env_params["pomo_size"]
        self.max_steps = env_params.get("max_steps", 100)
        self.fail_penalty = env_params.get("fail_penalty", 1000.0)

        self.batch_size = None
        self.problem_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        # Static instance data
        self.data = None
        # Dynamic state
        self.step_count = None
        self.current = None
        self.time = None
        self.remaining = None
        self.done = None        # (B,P) episode finished (success or failure)
        self.success = None     # (B,P)
        self.cost = None        # (B,P) accumulated edge length
        self.step_state = None

    def load_problems(self, tensors):
        """tensors: output dict of STSPProblemDef.instance_to_tensors."""
        self.data = tensors
        self.batch_size = tensors["start"].size(0)
        self.problem_size = tensors["adj"].size(1)
        self.BATCH_IDX = torch.arange(self.batch_size, device=tensors["adj"].device)[:, None] \
            .expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=tensors["adj"].device)[None, :] \
            .expand(self.batch_size, self.pomo_size)

    def reset(self):
        d = self.data
        b, p, n = self.batch_size, self.pomo_size, self.problem_size
        self.step_count = 0
        self.current = d["start"][:, None].expand(b, p).clone()   # (B,P) start node
        self.time = d["start_time"][:, None].expand(b, p).clone()  # (B,P)
        self.remaining = d["target_mask"][:, None, :].expand(b, p, n).clone()
        self.done = torch.zeros(b, p, dtype=torch.bool, device=d["adj"].device)
        self.success = torch.zeros(b, p, dtype=torch.bool, device=d["adj"].device)
        self.cost = torch.zeros(b, p, device=d["adj"].device)
        # Steps at which each rollout actually finished (for dynamic scheduling; -1 = not yet)
        self.done_steps = torch.full((b, p), -1, dtype=torch.long, device=d["adj"].device)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.current_node = None
        self.step_state.remaining = self.remaining
        self.step_state.time = self.time
        self.step_state.ninf_mask = self._build_mask()
        return Reset_State(**d), None, False

    def pre_step(self):
        return self.step_state, None, False

    def step(self, selected):
        """selected: (B,P) next node chosen by the model.

        Step 0 is the start placeholder (selected = start; no move, no cost).
        max_steps bounds the number of actual moves (default 100).
        """
        d = self.data
        moved = self.step_state.current_node is not None   # step 0 is not counted
        if moved:
            self.step_count += 1
        active = ~self.done                              # (B,P)

        # Move (only for active rollouts; finished ones stay in place)
        # Vectorized gather of length[b, cur[b,p], sel[b,p]]:
        b_idx = torch.arange(self.batch_size, device=selected.device)[:, None].expand_as(selected)
        cur = self.current
        edge_len = d["length"][b_idx, cur, selected]      # (B,P) 0 when sel == cur (diagonal)
        edge_tt = d["travel_time"][b_idx, cur, selected]  # (B,P)
        self.cost = self.cost + torch.where(active, edge_len, torch.zeros_like(edge_len))
        self.time = (self.time + torch.where(active, edge_tt, torch.zeros_like(edge_tt))) % 24.0
        self.current = torch.where(active, selected, cur)

        # Update remaining targets: drop targets that were just reached
        onehot = torch.zeros_like(self.remaining)
        onehot[b_idx, self.POMO_IDX, self.current] = True
        self.remaining = self.remaining & ~onehot

        # Success: all targets visited and back at the start
        finished = (self.remaining.sum(dim=2) == 0) & (self.current == d["start"][:, None]) & active
        self.success = self.success | finished
        self.done = self.done | finished

        # Build the next-step mask (dead ends are marked as failed episodes)
        self.step_state.current_node = self.current
        self.step_state.remaining = self.remaining
        self.step_state.time = self.time
        self.step_state.ninf_mask = self._build_mask()

        # Record actual steps of rollouts that just finished (after _build_mask,
        # because dead ends are marked done during mask construction)
        newly_done = self.done & (self.done_steps < 0)
        self.done_steps = torch.where(newly_done,
                                      torch.full_like(self.done_steps, self.step_count),
                                      self.done_steps)

        # Termination: all rollouts finished, or the move budget is exhausted
        # (timeouts count as failures)
        all_done = bool(self.done.all())
        timeout = moved and self.step_count >= self.max_steps
        done = all_done or timeout
        reward = None
        if done:
            reward = self._get_reward()
        return self.step_state, reward, done

    def _build_mask(self):
        """Dynamic action mask (B,P,N): 0 = allowed, -inf = blocked (float, added to logits).

        - Nodes that are not outgoing neighbours of the current node are blocked.
        - Edges whose time window contains the current time are blocked (closed
          interval, may wrap past midnight).
        - A dead end (no allowed action, episode not finished) marks the episode failed.
        - Finished episodes park at the current node (one-hot), so softmax yields 1.
        """
        d = self.data
        b, p, n = self.batch_size, self.pomo_size, self.problem_size
        device = d["adj"].device
        b_idx = torch.arange(b, device=device)[:, None].expand(b, p)      # (B,P)
        cur = self.current                                                # (B,P)

        # adj[b, cur[b,p], :] -> (B,P,N)
        adj_cur = torch.gather(d["adj"], 1, cur.unsqueeze(-1).expand(b, p, n))
        ws_cur = torch.gather(d["win_s"], 1, cur.unsqueeze(-1).expand(b, p, n))
        we_cur = torch.gather(d["win_e"], 1, cur.unsqueeze(-1).expand(b, p, n))

        t = self.time.unsqueeze(-1)                                       # (B,P,1)
        has_win = ws_cur >= 0
        # Closed interval [s,e]; s > e denotes a window crossing midnight
        in_win = torch.where(ws_cur < we_cur,
                             (t >= ws_cur) & (t <= we_cur),
                             (t >= ws_cur) | (t <= we_cur))
        mask = adj_cur & ~(has_win & in_win)                              # (B,P,N) bool

        # Dead end: no allowed action and the episode is not finished -> failure
        deadend = (mask.sum(dim=2) == 0) & (~self.done)
        self.done = self.done | deadend

        # Finished episodes stay at the current node
        park = torch.zeros_like(mask)
        park[b_idx, self.POMO_IDX, self.current] = True
        mask = torch.where(self.done.unsqueeze(-1), park, mask)

        # Convert to a float mask of 0 / -inf
        ninf = torch.zeros_like(mask, dtype=torch.float32)
        ninf = ninf.masked_fill(~mask, float("-inf"))
        return ninf

    def _get_reward(self):
        """-(total edge length + failure penalty). POMO advantage is relative
        within a group, so the scale does not matter."""
        return -(self.cost + self.fail_penalty * (~self.success).float())

    def episode_stats(self):
        """Call after all episodes finish: returns (mean actual steps, timeout rate, success rate).

        Rollouts truncated by max_steps (done_steps = -1) are counted with the
        final step_count, i.e. the full budget. Timeout rate is the fraction of
        truncated rollouts. Success rate is the fraction of rollouts that visited
        all targets and returned to the start.
        """
        ds = self.done_steps
        steps = torch.where(ds >= 0, ds, torch.full_like(ds, self.step_count))
        avg_steps = steps.float().mean().item()
        timeout_rate = (ds < 0).float().mean().item()
        success_rate = self.success.float().mean().item()
        return avg_steps, timeout_rate, success_rate

    # Test / evaluation helpers
    def rollout_result(self):
        """Return (cost, success, steps) as numpy arrays for tester statistics."""
        return (self.cost.detach().cpu().numpy(),
                self.success.detach().cpu().numpy(),
                self.step_count)
