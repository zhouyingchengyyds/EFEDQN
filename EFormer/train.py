"""EFormer-STSP training entry point.

Usage:
    python train.py                       # default config (n=100, shared training distribution)
    python train.py --epochs 200
    python train.py --smoke               # smoke mode (very small scale, quick run)

Notes:
- Training distribution: n=100, target=n/3, random
  non-target start, departure time U(0,24), random number of time windows.
- pomo_size is the number of parallel sampled rollouts per instance (REINFORCE
  with a shared baseline).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import create_logger, get_result_folder  # noqa: E402
from STSPTrainer import STSPTrainer  # noqa: E402


def build_params(args):
    # model_params: EFormer scale (embedding_dim=256, 6 encoder layers, 16 heads)
    model_params = {
        "embedding_dim": 256,
        "sqrt_embedding_dim": 256 ** (1 / 2),
        "head_num": 16,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "sqrt_qkv_dim": 16 ** (1 / 2),
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",

        "hidden_dim": 256,
        "GCN_dim": 6,
        "mlp_layers": 3,
        "aggregation": "mean",

        "ms_hidden_dim": 16,
        "ms_layer1_init": (1 / 2) ** (1 / 2),
        "ms_layer2_init": (1 / 16) ** (1 / 2),

        "edge_feat_dim": 8,
    }

    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": args.pomo_size,
        "max_steps": args.train_max_steps,  # training step budget (scheduler initial value when dynamic)
        "fail_penalty": 1000.0,
        "k_num": None,             # None -> normally distributed window count
    }

    optimizer_params = {
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-6,
        },
    }

    trainer_params = {
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_episodes": args.train_episodes,
        "train_batch_size": args.batch_size,
        "test_interval": args.test_interval,
        "test_size": args.test_size,
        "eval_max_steps": args.eval_max_steps,
        "patience": args.patience,
        "model_save_interval": args.save_interval,
        "result_folder": get_result_folder(),
        "model_load": {"enable": False},
        "dynamic_params": args.dynamic_params,
        "dyn_product_cap": args.dyn_product_cap,
        "dyn_max_steps_lb": args.dyn_max_steps_lb,
        "dyn_max_steps_ub": args.dyn_max_steps_ub,
        "dyn_batch_ub": args.dyn_batch_ub,
    }
    return env_params, model_params, optimizer_params, trainer_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--pomo-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=20 * 1000,
                        help="每 epoch 的实例数")
    parser.add_argument("--train-max-steps", type=int, default=100,
                        help="训练步数上限：--no-dynamic-params 时全程生效；"
                             "--dynamic-params 时仅作调度器初始值（随后被动态调整）")
    parser.add_argument("--batch-size", type=int, default=48,
                        help="训练批大小：--no-dynamic-params 时全程生效；"
                             "--dynamic-params 时仅作调度器初始值")
    parser.add_argument("--test-interval", type=int, default=10,
                        help="每多少个 epoch 在固定验证集上测试一次（0=关闭）")
    parser.add_argument("--test-size", type=int, default=32,
                        help="固定验证集实例数（初始化时一次性生成）")
    parser.add_argument("--eval-max-steps", type=int, default=1000,
                        help="验证评估专用步数上限（与最终测试协议一致），"
                             "不受动态调度影响")
    parser.add_argument("--patience", type=int, default=5,
                        help="早停：连续多少次固定集测试无改善则终止（0=关闭，"
                             "但仍保存 checkpoint-best.pt）")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu；缺省自动选择")
    parser.add_argument("--dynamic-params", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="启用 batch_size/max_steps 动态调度"
                             "（max_steps=round(avg_steps*1.25)，约束边界见 "
                             "--dyn-max-steps-lb/ub、--dyn-product-cap、--dyn-batch-ub；"
                             "默认开启；--no-dynamic-params 时训练全程使用 "
                             "--train-max-steps 与 --batch-size 的静态值）")
    parser.add_argument("--dyn-product-cap", type=int, default=20000,
                        help="动态调度约束（优先级2）：max_steps*batch_size 的上限")
    parser.add_argument("--dyn-max-steps-lb", type=int, default=100,
                        help="动态调度约束（优先级1）：max_steps 下限")
    parser.add_argument("--dyn-max-steps-ub", type=int, default=900,
                        help="动态调度约束（优先级1）：max_steps 上限")
    parser.add_argument("--dyn-batch-ub", type=int, default=64,
                        help="动态调度约束（优先级3）：batch_size 上限")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试模式")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        args.epochs = 2
        args.train_episodes = 32
        args.batch_size = 8
        args.test_interval = 1
        args.test_size = 4
        args.patience = 0  # disable early stopping in smoke mode
        args.pomo_size = 4
        args.problem_size = 25

    create_logger({"desc": "train_eformer_stsp", "filename": "train_log.txt"})
    env_params, model_params, optimizer_params, trainer_params = build_params(args)

    trainer = STSPTrainer(env_params=env_params, model_params=model_params,
                          optimizer_params=optimizer_params,
                          trainer_params=trainer_params)
    trainer.run()


if __name__ == "__main__":
    main()
