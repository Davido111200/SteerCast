#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Fine-tune TimesFM 2.5 on one benchmark dataset and horizon (the FT backbone used by SteerCast).

Every channel of the (standardised) training split is cut into sliding windows
of ``seq_len + pred_len`` points; the model is trained with an MSE loss on its
autoregressive point forecast. The checkpoint with the lowest validation MSE is
written to ``--out``.

    python -m finetune.timesfm -d dataset/ETT-small/ETTh1.csv --data ETTh1 -c 512 -p 96 --out checkpoints/timesfm/ETTh1_p96
"""
import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from steercast.backbones.timesfm import TimesFMBackbone
from time_moe.datasets.benchmark_dataset import compute_split_borders


def load_scaled_splits(csv_path: str, seq_len: int):
    """Train / validation splits (all channels), standardised with training statistics."""
    df = pd.read_csv(csv_path)
    border1s, border2s, _ = compute_split_borders(csv_path, seq_len)
    cols = list(df.columns[1:])
    values = df[cols].values.astype(np.float32)
    train_data = values[border1s[0]:border2s[0]]
    val_data = values[border1s[1]:border2s[1]]

    scaler = StandardScaler()
    scaler.fit(train_data)
    return {
        "train": scaler.transform(train_data).astype(np.float32),
        "val": scaler.transform(val_data).astype(np.float32),
        "columns": cols,
    }


class SlidingWindowMultiSeries(Dataset):
    """Windows of ``context_len + horizon_len`` points from every column of a [T, N] array."""

    def __init__(self, data: np.ndarray, context_len: int, horizon_len: int, stride: int = 1):
        assert data.ndim == 2, f"Expected [T, N], got {data.shape}"
        self.data = data
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        self.stride = int(stride)

        T, N = data.shape
        max_start = T - (self.context_len + self.horizon_len)
        if max_start < 0:
            raise ValueError(
                f"Series too short for context={self.context_len} horizon={self.horizon_len}: got T={T}"
            )
        self.positions_per_series = (max_start // self.stride) + 1
        self.num_series = N

    def __len__(self):
        return self.num_series * self.positions_per_series

    def __getitem__(self, idx):
        series_idx = idx // self.positions_per_series
        start = (idx % self.positions_per_series) * self.stride
        x = self.data[start:start + self.context_len, series_idx]
        y = self.data[start + self.context_len:start + self.context_len + self.horizon_len, series_idx]
        return {
            "inputs": torch.tensor(x, dtype=torch.float32),
            "labels": torch.tensor(y, dtype=torch.float32),
        }


@torch.no_grad()
def evaluate(model: TimesFMBackbone, loader: DataLoader, device: torch.device):
    model.model.eval()
    se_sum, ae_sum, count = 0.0, 0.0, 0
    for batch in loader:
        x = batch["inputs"].to(device=device, dtype=torch.float32)
        y = batch["labels"].to(device=device, dtype=torch.float32)
        err = model.predict_series(x, pred_len=y.shape[1]) - y
        se_sum += float((err * err).sum().item())
        ae_sum += float(err.abs().sum().item())
        count += int(y.numel())
    denom = max(count, 1)
    return {"mse": se_sum / denom, "mae": ae_sum / denom}


def save_checkpoint(model: TimesFMBackbone, out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    model.wrapper.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser("Fine-tune TimesFM 2.5 on a benchmark dataset")
    parser.add_argument("--model", "-m", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--data_path", "-d", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint directory")
    parser.add_argument("--seq_len", "-c", type=int, default=512)
    parser.add_argument("--pred_len", "-p", type=int, default=96)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=1, help="Stride between training windows")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9899,
                        help="Seed for the data order (the paper's checkpoints were trained unseeded)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    splits = load_scaled_splits(args.data_path, seq_len=args.seq_len)
    train_ds = SlidingWindowMultiSeries(splits["train"], args.seq_len, args.pred_len, stride=args.stride)
    val_ds = SlidingWindowMultiSeries(splits["val"], args.seq_len, args.pred_len, stride=args.stride)

    prefetch_factor = 2 if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, prefetch_factor=prefetch_factor, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, prefetch_factor=prefetch_factor, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesFMBackbone(args.model, device=device)
    optim = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.model.train()
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["inputs"].to(device=device, dtype=torch.float32)
            y = batch["labels"].to(device=device, dtype=torch.float32)
            loss = F.mse_loss(model.predict_series_trainable(x, pred_len=y.shape[1]), y, reduction="mean")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=(running / steps))

        val_metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            save_checkpoint(model, args.out, {
                "data": args.data,
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "columns": splits["columns"],
                "args": vars(args),
            })

    print(f"Done. Best val_mse={best_val:.6f}. Model saved to: {args.out}")


if __name__ == "__main__":
    main()
