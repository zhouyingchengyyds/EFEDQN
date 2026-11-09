"""Evaluation entry point for EFormer-STSP (greedy evaluation protocol).

Usage:
    # Pool mode: certified instance pool with paired gaps (shared instances)
    python test.py --model-path result/xxx/checkpoint-500.pt \
                   --instances-dir benchmark

    # Pool mode with the best-of-10 sampling protocol (equal search budget)
    python test.py --model-path ... --instances-dir ... --aug-k 10

    # Random mode: generate instances for the five CONDITIONS
    python test.py --model-path ... --mode random --num-instances 100

Without --model-path a randomly initialized model is used (pipeline validation).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from utils import create_logger  # noqa: E402
from STSPTester import run_tester  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
                        help="checkpoint 路径（含 checkpoint-{}.pt 模板时用 "
                             "--epoch 指定；直接给 .pt 文件则原样加载）")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--mode", type=str, default="pool", choices=["pool", "random"])
    parser.add_argument("--instances-dir", type=str,
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(__file__), "..", "..", "long_reward", "benchmark")))
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--aug-k", type=int, default=1,
                        help=">1 时额外输出 best-of-K 增强协议结果")
    parser.add_argument("--aug-mode", type=str, default="seed",
                        choices=["seed", "sample"],
                        help="K>1 的多样性来源: seed=原版 EFormer 实例增强"
                             "(副本独立重抽 col_emb 种子 + argmax 贪心, 默认); "
                             "sample=逐步采样解码")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "problem_size": 100,       # overridden by the actual instance size in pool mode
        "pomo_size": 1,            # greedy evaluation
        "max_steps": 100,
        "fail_penalty": 1000.0,
    }
    tester_params = {
        "device": args.device,
        "mode": args.mode,
        "instances_dir": args.instances_dir,
        "num_instances": args.num_instances,
        "aug_k": args.aug_k,
        "aug_mode": args.aug_mode,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "model_load": {"enable": args.model_path is not None},
    }
    if args.model_path is not None:
        if args.epoch is not None:
            tester_params["model_load"].update({"path": args.model_path,
                                                "epoch": args.epoch})
        else:
            # A direct checkpoint file: build path/epoch so the template matches
            path, fname = os.path.split(os.path.abspath(args.model_path))
            token = os.path.splitext(fname)[0].split("-")[-1]
            # The token of checkpoint-best.pt (best early-stopping model) is not an
            # integer; passing the string "best" into checkpoint-{epoch}.pt matches it
            epoch = int(token) if token.isdigit() else token
            tester_params["model_load"].update({"path": path, "epoch": epoch})

    create_logger({"desc": "test_eformer_stsp", "filename": "test_log.txt"})
    run_tester(env_params, model_params, tester_params)


if __name__ == "__main__":
    main()
