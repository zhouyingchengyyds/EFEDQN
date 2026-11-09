"""Smoke test for EFormer-STSP (CPU only, a few minutes).

Checks:
1. Instance generation (train/test distributions) and feature tensor construction;
2. Dynamic masking, time evolution and termination conditions of the environment;
3. Certified-path replay: feed the Gurobi-certified optimal path of each benchmark
   instance node by node into STSPEnv and verify that (a) every action is allowed
   by the dynamic mask, (b) the replay cost equals the solver-certified cost, and
   (c) the episode is marked successful;
4. Model forward / backward pass (random init, small scale);
5. End-to-end run of the tester in pool mode (random weights, pipeline only).

Usage (from the STSP directory):
    python smoke_test.py [--benchmark-dir ../../long_reward/benchmark]
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from STSPProblemDef import (generate_train_instance, generate_test_instance,  # noqa: E402
                            instance_from_json, instance_to_tensors)
from STSPEnv import STSPEnv  # noqa: E402
from STSPModel import STSPModel  # noqa: E402


def model_params_small():
    """Reduced model parameters for the smoke test (train.py uses the full scale)."""
    return {
        "embedding_dim": 64,
        "sqrt_embedding_dim": 64 ** (1 / 2),
        "head_num": 4,
        "encoder_layer_num": 2,
        "qkv_dim": 16,
        "sqrt_qkv_dim": 16 ** (1 / 2),
        "logit_clipping": 10,
        "ff_hidden_dim": 128,
        "eval_type": "argmax",
        "hidden_dim": 64,
        "GCN_dim": 2,
        "mlp_layers": 2,
        "aggregation": "mean",
        "ms_hidden_dim": 8,
        "ms_layer1_init": (1 / 2) ** (1 / 2),
        "ms_layer2_init": (1 / 16) ** (1 / 2),
        "edge_feat_dim": 8,
    }


def check(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok -", msg)


def test_instance_generation():
    print("[1] 实例生成与张量构建")
    inst = generate_train_instance(25)
    n = inst["n"]
    check(1 <= inst["start"] < n, "训练实例起点为非目标随机节点")
    check(inst["start"] not in inst["targets"], "起点不是目标")
    check(len(inst["targets"]) == 25 // 3, "目标数 = n/3")
    out_deg = inst["adj"].sum(axis=1)
    check(int(out_deg.max()) <= 4, "出度 <= 4")
    check(int(out_deg.min()) >= 1, "弱连通保证出度 >= 1")
    lens = inst["length"][inst["adj"]]
    check(1 <= lens.min() and lens.max() <= 50, "边长在 [1,50]")
    n_win = int((inst["win_s"] >= 0).sum())
    check(n_win >= 16, "时间窗数量 >= 16（正态截断下界）")
    # Test instance: start 0, time 0, targets exclude 0
    tinst = generate_test_instance(25, 16, 8)
    check(tinst["start"] == 0 and tinst["start_time"] == 0.0, "测试实例起点=0, 时间=0")
    check(0 not in tinst["targets"], "测试目标不含仓库")

    tensors = instance_to_tensors([inst], "cpu")
    n_ = tensors["x_edges"].size(1)
    check(tensors["x_edges"].shape == (1, n_, n_), "x_edges 形状")
    check(bool((tensors["x_edges"][:, torch.arange(n_), torch.arange(n_)] == 2).all()),
          "对角 tag=2")
    cm = tensors["cost_mat"]
    check(float(cm.max()) <= 1.5 and bool((cm[:, torch.arange(n_), torch.arange(n_)] == 0).all()),
          "代价矩阵：对角0，哨兵<=1.5")
    check(tensors["edge_feats"].shape == (1, n_, n_, 8), "边特征 8 通道")
    check(bool((tensors["edge_feats"][..., 2] == tensors["adj"].float()).all()),
          "存在性通道与邻接一致")


def test_env_rollout_random():
    print("[2] 随机策略 rollout（掩码/终止/计步）")
    torch.manual_seed(0)
    inst = generate_test_instance(25, 16, 8)
    env = STSPEnv(pomo_size=2, max_steps=100)
    tensors = instance_to_tensors([inst], "cpu")
    env.load_problems(tensors)
    env.reset()
    state, _, done = env.pre_step()
    steps = 0
    while not done:
        mask = state.ninf_mask                      # (1,2,N)
        # Pick a random allowed action
        allowed = (mask[0, 0] == 0).nonzero().squeeze(-1)
        check_step = allowed.numel() > 0
        if not check_step:
            break
        sel = allowed[torch.randint(allowed.numel(), (1,))].unsqueeze(0)  # (1,1)
        sel = sel.expand(1, 2)
        state, reward, done = env.step(sel)
        steps += 1
    check(steps <= env.max_steps + 1, "步数受限（含第0步占位）")
    print("  rollout 结束: steps=%d success=%s cost=%.1f"
          % (env.step_count, bool(env.success[0, 0]), float(env.cost[0, 0])))


def test_replay_certified(benchmark_dir, max_instances=8):
    print("[3] 认证路径回放（核心正确性校验）")
    files = sorted(glob.glob(os.path.join(benchmark_dir, "n*", "n*.json")))
    if not files:
        print("  [skip] 未找到 benchmark 实例:", benchmark_dir)
        return
    env = STSPEnv(pomo_size=1, max_steps=100)
    n_checked = 0
    for path in files[:max_instances]:
        inst = instance_from_json(path)
        env.max_steps = inst.get("max_steps", 100)  # replay with the certified step budget
        solver = inst["solver"]
        if not solver or not solver.get("path"):
            continue
        opt_path = solver["path"]
        tensors = instance_to_tensors([inst], "cpu")
        env.load_problems(tensors)
        env.reset()
        state, _, _ = env.pre_step()

        # Step 0: start placeholder
        state, reward, done = env.step(torch.tensor([[inst["start"]] * 1]))
        # Replay step by step
        for j in opt_path[1:]:
            allowed = (state.ninf_mask[0, 0] == 0)
            assert bool(allowed[j]), \
                "%s: 节点%d 在掩码中被禁止（动态掩码与认证路径冲突）" % (inst["id"], j)
            state, reward, done = env.step(torch.tensor([[j]]))

        assert bool(env.success[0, 0]), "%s: 回放未判定成功" % inst["id"]
        cost = float(env.cost[0, 0])
        ref = float(solver["cost"])
        assert abs(cost - ref) < 1e-6, \
            "%s: 回放成本 %.4f != 认证成本 %.4f" % (inst["id"], cost, ref)
        assert env.step_count == len(opt_path) - 1, \
            "%s: 步数 %d != %d" % (inst["id"], env.step_count, len(opt_path) - 1)
        print("  ok - %s: cost=%.1f steps=%d 与 Gurobi 认证一致"
              % (inst["id"], cost, env.step_count))
        n_checked += 1
    check(n_checked > 0, "至少回放校验了 %d 个认证实例" % n_checked)


def test_model_train_step():
    print("[4] 模型前向/反向（POMO REINFORCE 单批）")
    torch.manual_seed(0)
    insts = [generate_train_instance(25) for _ in range(4)]
    tensors = instance_to_tensors(insts, "cpu")
    model = STSPModel(**model_params_small())
    env = STSPEnv(pomo_size=4, max_steps=100)
    env.load_problems(tensors)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    check(model.encoded_nodes.shape == (4, 25, 64), "节点编码形状")
    check(model.encoded_graph.shape == (4, 25, 64), "图编码形状")

    model.train()
    prob_list = torch.zeros(4, env.pomo_size, 0)
    state, reward, done = env.pre_step()
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
    advantage = reward - reward.mean(dim=1, keepdims=True)
    loss = -(advantage * prob_list.log().sum(dim=2)).mean()
    loss.backward()
    check(torch.isfinite(loss).item(), "损失有限 (%.4f)" % loss.item())
    grads = [p.grad.abs().sum().item() for p in model.parameters()
             if p.grad is not None]
    check(sum(grads) > 0, "梯度非全零（%d 个参数有梯度）" % len(grads))

    # Greedy evaluation branch
    model.eval()
    env2 = STSPEnv(pomo_size=1, max_steps=100)
    tensors1 = instance_to_tensors([insts[0]], "cpu")
    env2.load_problems(tensors1)
    reset_state, _, _ = env2.reset()
    with torch.no_grad():
        model.pre_forward(reset_state)
        state, reward, done = env2.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env2.step(selected)
    print("  ok - 贪心 rollout: success=%s cost=%.1f steps=%d"
          % (bool(env2.success[0, 0]), float(env2.cost[0, 0]), env2.step_count))


def test_tester_end2end(benchmark_dir):
    print("[5] 评估器池模式端到端（随机权重，仅验证流程）")
    import logging
    logging.basicConfig(level=logging.WARNING)
    from STSPTester import STSPTester
    files = sorted(glob.glob(os.path.join(benchmark_dir, "n10", "n*.json")))
    if not files:
        print("  [skip] 无 n10 实例")
        return
    tester = STSPTester(
        env_params={"pomo_size": 1, "max_steps": 100},
        model_params=model_params_small(),
        tester_params={"model_load": {"enable": False}, "device": "cpu",
                       "mode": "pool", "instances_dir": benchmark_dir,
                       "aug_k": 1, "output_dir": "results_smoke"})
    rows = []
    for path in files[:3]:
        inst = instance_from_json(path)
        cost, success, steps, t_ms = tester._greedy_rollout(inst)
        rows.append((inst["id"], cost, success, steps, t_ms))
    for r in rows:
        print("  %s: cost=%.1f feasible=%s steps=%d t=%.0fms" % r)
    check(len(rows) == 3, "评估器 rollout 正常返回")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str,
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(__file__), "..", "..",
                            "long_reward", "benchmark")))
    args = parser.parse_args()

    torch.manual_seed(2024)
    np.random.seed(2024)
    import random
    random.seed(2024)

    test_instance_generation()
    test_env_rollout_random()
    test_replay_certified(args.benchmark_dir)
    test_model_train_step()
    test_tester_end2end(args.benchmark_dir)
    print("\n=== 冒烟测试全部通过 ===")


if __name__ == "__main__":
    main()
