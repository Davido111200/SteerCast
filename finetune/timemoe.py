#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""Fine-tune Time-MoE on one benchmark dataset (the FT backbone used by SteerCast).

The training data are all rows before the test split of the given CSV (see
``time_moe.datasets.benchmark_train_dataset``), one sequence per channel, each
z-normalised. Defaults reproduce the fine-tuning recipe of the paper.

    python -m finetune.timemoe -d dataset/ETT-small/ETTh1.csv -o checkpoints/timemoe/ETTh1

For multi-GPU training launch the same module with ``torchrun``.
"""
import argparse

from time_moe.runner import TimeMoeRunner


def main():
    parser = argparse.ArgumentParser("Fine-tune Time-MoE")
    parser.add_argument("--data_path", "-d", type=str, required=True,
                        help="Benchmark CSV, or a folder / file in a format supported by TimeMoEDataset")
    parser.add_argument("--model_path", "-m", type=str, default="Maple728/TimeMoE-50M",
                        help="Pre-trained model to start from")
    parser.add_argument("--output_path", "-o", type=str, required=True,
                        help="Directory the fine-tuned model is written to")
    parser.add_argument("--max_length", type=int, default=4096, help="Training window length")
    parser.add_argument("--stride", type=int, default=None,
                        help="Step of the sliding training window (default: max_length)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_steps", type=int, default=None,
                        help="Number of training steps (overrides --num_train_epochs)")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--normalization_method", type=str, choices=["none", "zero", "max"], default="zero")
    parser.add_argument("--seed", type=int, default=9899)
    parser.add_argument("--attn_implementation", type=str, choices=["auto", "eager", "flash_attention_2"],
                        default="auto")
    parser.add_argument("--lr_scheduler_type", type=str,
                        choices=["constant", "linear", "cosine", "constant_with_warmup"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], type=str, default="fp32")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--from_scratch", action="store_true", help="Train from a random initialisation")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_strategy", choices=["steps", "epoch", "no"], type=str, default="no")
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--save_only_model", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--evaluation_strategy", choices=["steps", "epoch", "no"], type=str, default="no")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.normalization_method == "none":
        args.normalization_method = None

    runner = TimeMoeRunner(model_path=args.model_path, output_path=args.output_path, seed=args.seed)
    runner.train_model(
        from_scratch=args.from_scratch,
        max_length=args.max_length,
        stride=args.stride,
        data_path=args.data_path,
        normalization_method=args.normalization_method,
        attn_implementation=args.attn_implementation,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_steps=args.train_steps,
        num_train_epochs=args.num_train_epochs,
        precision=args.precision,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
    )


if __name__ == "__main__":
    main()
