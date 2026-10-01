#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""Fine-tune Timer-XL on one benchmark dataset (the FT backbone used by SteerCast).

All channels are trained independently with shared weights on sliding windows
of the (standardised) training split, with an MSE loss on the next ``pred_len``
points. A checkpoint is written after every epoch to
``{output_dir}/{data}/epoch-{n}``; the paper uses ``epoch-1``.

    python -m finetune.timerxl -d dataset/ETT-small/ETTh1.csv --data ETTh1 --output_dir checkpoints/timerxl
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDatasetTrain

logging.basicConfig(level=logging.INFO)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)

    # Timer consumes patches of `input_token_len` points: use a multiple of it as context.
    patch_len = int(getattr(model.config, "input_token_len", 512))
    if args.seq_len is None:
        args.seq_len = 30 * patch_len
    else:
        args.seq_len = max(patch_len, (args.seq_len // patch_len) * patch_len)
    logging.info(f"patch_len={patch_len}, seq_len={args.seq_len}, pred_len={args.pred_len}")

    if not args.data_path.endswith(".csv"):
        raise ValueError("Only CSV data is supported.")
    args.label_len = 0
    dataset_train = BenchmarkEvalDatasetTrain(
        args, args.data_path, seq_len=args.seq_len, pred_len=args.pred_len,
        max_train_samples=args.pool_number,
    )
    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            series_in = batch["inputs"].to(device=device, dtype=torch.float32)  # [B, seq_len]
            target = batch["labels"].to(device=device, dtype=torch.float32)     # [B, pred_len]

            out = model(series_in, use_cache=False, max_output_length=args.pred_len, return_dict=True)
            preds = out.logits
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            loss = loss_fn(preds, target[:, :args.pred_len])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += float(loss.item())
            if step % args.log_interval == 0:
                pbar.set_postfix(loss=f"{running_loss / args.log_interval:.6f}")
                running_loss = 0.0

        ckpt_dir = os.path.join(args.output_dir, f"{args.data}/epoch-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tune Timer-XL (all channels independently)")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Benchmark CSV path")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (checkpoint sub-directory)")
    parser.add_argument("--model", "-m", type=str, default="thuml/timer-base-84m", help="Pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-c", type=int, default=None,
                        help="Context length, rounded down to a multiple of the patch length "
                             "(default: 30 patches = 2880 points)")
    parser.add_argument("--pred_len", "-p", type=int, default=96)
    parser.add_argument("--pool_number", type=int, default=10000,
                        help="Train on the last `pool_number` time steps of the training split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9899,
                        help="Seed for the data order (the paper's checkpoints were trained unseeded)")
    train(parser.parse_args())
