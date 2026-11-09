"""EFormer trainer for S-TSP-TVRA (POMO-style REINFORCE).

Training protocol:
- Instances follow the training distribution (n=100, target=n/3,
  random non-target start, departure time U(0,24), random number of time
  windows). Every batch generates fresh instances that are consumed once and
  never reused across batches or epochs.
- Optimizer: Adam(lr=1e-4, weight_decay=1e-6).
- Each instance uses pomo_size sampled rollouts with the group mean as the shared
  baseline. The start is fixed (depot) or sampled, so POMO's multi-start trick
  does not apply and the standard REINFORCE variant with sampled rollouts and a
  shared baseline is used instead.
- Reward: -(total edge length + 1000 * failure).
- Periodic validation: every test_interval epochs, greedy evaluation on a fixed
  validation set built once at initialization (independent seed, test protocol:
  start = 0, departure time = 0), so scores are comparable across epochs.
"""

import logging
import os

import numpy as np
import torch
from torch.optim import Adam as Optimizer

from STSPEnv import STSPEnv as Env
from STSPModel import STSPModel as Model
from STSPProblemDef import (generate_train_instance, generate_test_instance,
                            instance_to_tensors)
from utils import TimeEstimator, AverageMeter


class STSPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = logging.getLogger(name="trainer")
        self.result_folder = trainer_params["result_folder"]
        self.device = torch.device(trainer_params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Fix the seed for reproducibility; no pre-generated instance pool, each
        # batch samples fresh instances that are consumed once (no reuse across
        # batches or epochs)
        import random as _random
        import numpy as _np
        seed = trainer_params.get("seed", 2024)
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)          # GPU-side random source (sampled rollouts)
        torch.backends.cudnn.deterministic = True  # deterministic cuDNN algorithms
        torch.backends.cudnn.benchmark = False     # disable autotuning (it adds randomness)
        self.problem_size = env_params["problem_size"]
        self.k_num = env_params.get("k_num")

        # Fixed validation set: built once at initialization with an independent
        # seed and reused by every periodic test, so validation scores stay
        # comparable across epochs. The RNG state is saved before generation and
        # restored afterwards, so the validation set does not affect the training
        # instance stream.
        self.test_interval = trainer_params.get("test_interval", 10)
        test_size = trainer_params.get("test_size", 32)
        # Evaluation-only step budget (--eval-max-steps): independent of the
        # training max_steps and the dynamic scheduler, and equal to the final test
        # protocol, so validation scores are comparable across epochs and configs
        self.eval_max_steps = trainer_params.get("eval_max_steps", 100)
        rng_states = (_random.getstate(), _np.random.get_state(),
                      torch.get_rng_state())
        _random.seed(seed + 1)
        _np.random.seed(seed + 1)
        torch.manual_seed(seed + 1)
        self.logger.info("generating fixed validation set (size=%d, n=%d, seed=%d) ...",
                         test_size, self.problem_size, seed + 1)
        self.test_instances = [
            generate_test_instance(self.problem_size,
                                   k_num=env_params.get("k_num"),
                                   target_num=self.problem_size // 3)
            for _ in range(test_size)
        ]
        _random.setstate(rng_states[0])
        _np.random.set_state(rng_states[1])
        torch.set_rng_state(rng_states[2])

        # Early stopping on the mean cost of the fixed validation set (lower is
        # better). Training stops after patience consecutive tests without
        # improvement; every new best is saved to checkpoint-best.pt.
        # patience=0 or test_interval=0 disables early stopping.
        self.patience = trainer_params.get("patience", 5)
        self.best_test_score = None
        self.best_epoch = None
        self._no_improve_count = 0

        # Components
        self.model = Model(**model_params).to(self.device)
        self.env = Env(**env_params)
        self.optimizer = Optimizer(self.model.parameters(),
                                   **optimizer_params["optimizer"])

        # Dynamic parameter scheduling (joint batch_size / max_steps adjustment,
        # see DynamicParamScheduler)
        self.dynamic_params = trainer_params.get("dynamic_params", False)
        if self.dynamic_params:
            from DynamicParamScheduler import DynamicParamScheduler
            self.scheduler = DynamicParamScheduler(
                product_cap=trainer_params.get("dyn_product_cap", 4800),
                max_steps_lb=trainer_params.get("dyn_max_steps_lb", 100),
                max_steps_ub=trainer_params.get("dyn_max_steps_ub", 1500),
                batch_ub=trainer_params.get("dyn_batch_ub", 64),
                init_max_steps=env_params.get("max_steps", 100),
                init_batch_size=trainer_params.get("train_batch_size", 48))
            self.logger.info(
                "DynamicParamScheduler enabled: init max_steps=%d batch_size=%d "
                "(constraints by priority: %d<=ms<=%d > ms*bs<=%d > 1<=bs<=%d)",
                self.scheduler.max_steps, self.scheduler.batch_size,
                self.scheduler.max_steps_lb, self.scheduler.max_steps_ub,
                self.scheduler.product_cap, self.scheduler.batch_ub)
        else:
            self.scheduler = None

        # Checkpoint resume
        self.start_epoch = 1
        model_load = trainer_params.get("model_load", {"enable": False})
        if model_load.get("enable"):
            checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.logger.info("Saved Model Loaded !!")

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info("=================================================================")
            train_score, train_loss = self._train_one_epoch(epoch)
            elapsed, remain = self.time_estimator.get_est_string(epoch, epochs)
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}], "
                             "Score(best-pomo cost):{:.4f}, Loss:{:.4f}".format(
                                 epoch, epochs, elapsed, remain, train_score, train_loss))

            if self.test_interval > 0 and \
                    ((epoch % self.test_interval == 0) or epoch == epochs):
                test_score, test_feasible = self._evaluate_fixed()
                self.logger.info("Epoch {:3d}: Test on fixed set  Score(avg cost):{:.4f}, "
                                 "Feasible:{:.2%}".format(epoch, test_score, test_feasible))
                if self._update_early_stopping(epoch, test_score):
                    break  # early stopping triggered

            if (epoch % self.trainer_params.get("model_save_interval", 50)) == 0 \
                    or epoch == epochs:
                ckpt_path = os.path.join(self.result_folder, "checkpoint-{}.pt".format(epoch))
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, ckpt_path)
                self.logger.info("Saved trained_model to %s", ckpt_path)

        self.logger.info(" *** Training Done *** ")

    def _update_early_stopping(self, epoch, test_score):
        """Early-stopping criterion and best-model saving. Returns True to stop.

        - Lower mean validation cost is better; each new best saves checkpoint-best.pt
        - patience consecutive tests without improvement trigger early stopping
        - patience=0 only tracks and saves the best model without stopping
        """
        if self.best_test_score is None or test_score < self.best_test_score:
            self.best_test_score = test_score
            self.best_epoch = epoch
            self._no_improve_count = 0
            best_path = os.path.join(self.result_folder, "checkpoint-best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "test_score": test_score,
            }, best_path)
            self.logger.info("Epoch %3d: New best test score %.4f, saved to %s",
                             epoch, test_score, best_path)
            return False
        self._no_improve_count += 1
        self.logger.info("Epoch %3d: No improvement for %d test(s) "
                         "(best %.4f @ epoch %d)",
                         epoch, self._no_improve_count,
                         self.best_test_score, self.best_epoch)
        if self.patience > 0 and self._no_improve_count >= self.patience:
            self.logger.info("Early stopping triggered at epoch %d "
                             "(best test score %.4f at epoch %d)",
                             epoch, self.best_test_score, self.best_epoch)
            return True
        return False

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        while episode < train_num_episode:
            remaining_eps = train_num_episode - episode
            base_batch = (self.scheduler.batch_size if self.scheduler is not None
                          else self.trainer_params["train_batch_size"])
            batch_size = min(base_batch, remaining_eps)
            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

        self.logger.info("Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}".format(
            epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))
        return score_AM.avg, loss_AM.avg

    def _generate_instances(self, batch_size):
        """Generate batch_size fresh training instances on the fly (single use)."""
        return [generate_train_instance(self.problem_size, k_num=self.k_num)
                for _ in range(batch_size)]

    def _evaluate_fixed(self):
        """Greedy rollout per instance on the fixed validation set (pomo=1, argmax).

        The set is built once in __init__ and only read here (never regenerated).
        Returns (mean cost, feasible rate). The MatNet-style col_emb of the
        Precoder calls torch.rand on every pre_forward, so this evaluation
        advances the GPU random stream, but the whole RNG consumption sequence is
        fixed by the seed, so training reproducibility and validation determinism
        are preserved.
        """
        self.model.eval()
        was_pomo = self.env.pomo_size
        was_max_steps = self.env.max_steps
        self.env.pomo_size = 1
        # Evaluation always uses the dedicated protocol step budget (--eval-max-steps),
        # unaffected by dynamic scheduling, so scores stay comparable across epochs
        self.env.max_steps = self.eval_max_steps
        costs, feas = [], []
        try:
            with torch.no_grad():
                for inst in self.test_instances:
                    tensors = instance_to_tensors([inst], self.device)
                    self.env.load_problems(tensors)
                    reset_state, _, _ = self.env.reset()
                    self.model.pre_forward(reset_state)
                    state, reward, done = self.env.pre_step()
                    while not done:
                        selected, _ = self.model(state)
                        state, reward, done = self.env.step(selected)
                    costs.append(float(self.env.cost[0, 0].item()))
                    feas.append(bool(self.env.success[0, 0].item()))
        finally:
            self.env.pomo_size = was_pomo
            self.env.max_steps = was_max_steps
        return float(np.mean(costs)), float(np.mean(feas))

    def _train_one_batch(self, batch_size):
        self.model.train()
        if self.scheduler is not None:
            self.env.max_steps = self.scheduler.max_steps
        tensors = instance_to_tensors(self._generate_instances(batch_size), self.device)
        self.env.load_problems(tensors)
        reset_state, _, _ = self.env.reset()

        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.device)
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # POMO REINFORCE: group mean as shared baseline
        advantage = reward - reward.float().mean(dim=1, keepdims=True)   # (B,P)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        # Score: best rollout cost per instance (without the failure penalty)
        best_cost, _ = self.env.cost.min(dim=1)
        score_mean = best_cost.float().mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # Post-batch report: pass rate (read-only stats), memory usage and dynamic
        # parameter scheduling.
        # pass = fraction of rollouts that finished naturally within max_steps;
        # success = fraction that completed the task feasibly
        avg_steps, timeout_rate, success_rate = self.env.episode_stats()
        pass_rate = 1.0 - timeout_rate
        if str(self.device).startswith("cuda"):
            mem_now = torch.cuda.memory_allocated() / 1024 ** 3
            mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            mem_str = "mem={:.2f}G peak={:.2f}G".format(mem_now, mem_peak)
        else:
            mem_str = "mem=n/a"
        if self.scheduler is not None:
            new_bs, new_ms = self.scheduler.update(avg_steps)
            self.logger.info(
                "DynParam: avg_steps=%.1f, pass=%.1f%%, success=%.1f%%, %s -> "
                "next max_steps=%d, batch_size=%d",
                avg_steps, 100.0 * pass_rate, 100.0 * success_rate, mem_str,
                new_ms, new_bs)
        else:
            self.logger.info(
                "Batch: avg_steps=%.1f, pass=%.1f%%, success=%.1f%%, %s "
                "(max_steps=%d, batch_size=%d)",
                avg_steps, 100.0 * pass_rate, 100.0 * success_rate, mem_str,
                self.env.max_steps, batch_size)
        return score_mean.item(), loss_mean.item()
