#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""Shared helpers for the baseline scripts (FT, RAF, RAFT)."""
import argparse
import json
import math
import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from steercast.metrics import MAEMetric, MSEMetric
from time_moe.datasets.benchmark_dataset import (
    BenchmarkEvalDataset,
    BenchmarkEvalDatasetTrain,
    GeneralEvalDataset,
)

mp.set_sharing_strategy("file_system")


# --------------------------------------------------------------------------- #
# Arguments / setup
# --------------------------------------------------------------------------- #
def common_parser(description: str, default_model: str = None, with_database: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--model", "-m", type=str, default=default_model, help="Model path or Hugging Face id")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Benchmark CSV path")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (used for cache/result paths)")
    parser.add_argument("--seq_len", "-c", type=int, default=None,
                        help="Look-back length (default: 512 for horizons 96/192/336/720, else 96)")
    parser.add_argument("--pred_len", "-p", type=int, default=96, help="Forecast horizon")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0)
    if with_database:
        parser.add_argument("--num_channels", type=int, required=True, help="Number of variates in the CSV")
        parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser


def finalize_args(args):
    if args.seq_len is None:
        args.seq_len = 512 if args.pred_len in (96, 192, 336, 720) else 96
    # Read by BenchmarkEvalDatasetTrain.
    args.label_len = 0
    return args


def get_device():
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype


def load_timemoe(model_path: str, device):
    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        model = TimeMoeForPrediction.from_pretrained(model_path, device_map=device, torch_dtype="auto")
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype="auto", trust_remote_code=True,
        )
    model.eval()
    return model


def load_timerxl(model_path: str, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def eval_dataset(args):
    if args.data_path.endswith(".csv"):
        return BenchmarkEvalDataset(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len)
    return GeneralEvalDataset(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len)


def test_loader(args):
    return DataLoader(
        dataset=eval_dataset(args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
        persistent_workers=False,
        pin_memory=False,
    )


def to_2d(x: torch.Tensor) -> torch.Tensor:
    """[B, T, 1] or [B, 1, T] -> [B, T]."""
    if x.ndim == 3 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def iterate_train_windows(args):
    """Yield ``(channel_id, global_idx, x [B, P], y [B, F])`` batches over the training split.

    Windows come from ``BenchmarkEvalDatasetTrain.data_loader`` (channel-major
    order, batch size ``args.batch_size``); the channel of a window follows from
    its running index.
    """
    if not args.data_path.endswith(".csv"):
        raise ValueError("Only CSV data is supported for building the retrieval database.")
    dataset_train = BenchmarkEvalDatasetTrain(
        args, args.data_path, seq_len=args.seq_len, pred_len=args.pred_len,
        max_train_samples=args.pool_number,
    )
    num_each = max(1, len(dataset_train) // args.num_channels)
    base_idx = 0
    for x, y, _x_mark, _y_mark in tqdm(dataset_train.data_loader, desc="Building database", ncols=100):
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if y.ndim == 3 and y.shape[1] == 1:
            y = y.squeeze(1)
        B = x.shape[0]
        channel_ids = [min((base_idx + b) // num_each, args.num_channels - 1) for b in range(B)]
        yield channel_ids, base_idx, x, y
        base_idx += B


# --------------------------------------------------------------------------- #
# RAF: prepend the nearest training history (in the backbone's latent space)
# --------------------------------------------------------------------------- #
def build_or_load_raf_cache(args, backbone: str, rep_fn):
    """Per-channel list of ``(global_idx, key fp16 [H], history fp16 [P])``.

    ``rep_fn(x)`` maps a [B, P] float tensor to its retrieval key [B, H].
    """
    cache_path = os.path.join(
        args.cache_dir, backbone, "raf", args.data, f"{args.seq_len}_{args.pred_len}",
        f"raf_cache_pool{args.pool_number}.pt",
    )
    if os.path.exists(cache_path):
        print(f"Loaded RAF cache: {cache_path}")
        return torch.load(cache_path)

    cache = {i: [] for i in range(args.num_channels)}
    with torch.no_grad():
        for channel_ids, base_idx, x, _y in iterate_train_windows(args):
            r_cpu = rep_fn(x).detach().cpu().to(torch.float16).contiguous()
            x_cpu = x.detach().cpu().to(torch.float16).contiguous()
            for b, c in enumerate(channel_ids):
                cache[c].append((base_idx + b, r_cpu[b], x_cpu[b].clone()))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved RAF cache to {cache_path}")
    return cache


def raf_pools(cache, device):
    """Stack each channel's keys (on ``device``) and histories (on CPU)."""
    reps, inps = {}, {}
    for c, items in cache.items():
        if len(items) == 0:
            reps[c], inps[c] = None, None
            continue
        reps[c] = torch.stack([item[1].float() for item in items], dim=0).to(device)  # [N, H]
        inps[c] = torch.stack([item[2].float() for item in items], dim=0)             # [N, P]
    return reps, inps


def augment_context(test_inp: torch.Tensor, retrieved_inp: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Z-normalise both windows, shift the retrieved one so that it ends where the
    query starts, and prepend it: returns a [2 * P] context."""
    r_mean = retrieved_inp.mean()
    r_std = retrieved_inp.std() + eps
    r_norm = (retrieved_inp - r_mean) / r_std

    t_mean = test_inp.mean()
    t_std = test_inp.std() + eps
    t_norm = (test_inp - t_mean) / t_std

    r_norm = r_norm + (t_norm[0] - r_norm[-1])
    return torch.cat([r_norm, t_norm], dim=0)


def raf_contexts(test_inputs_1d, test_reps, channel_ids, reps_pool, inps_pool, seq_len: int,
                 dtype=torch.float32, device="cpu"):
    """Top-1 (Euclidean) retrieval per query and context augmentation: [B, 2 * seq_len]."""
    B = test_inputs_1d.shape[0]
    augmented = torch.zeros(B, 2 * seq_len, dtype=dtype, device=device)
    for b in range(B):
        c = int(channel_ids[b])
        pool = reps_pool.get(c)
        if pool is None or len(pool) == 0:
            t = test_inputs_1d[b].detach().float().reshape(-1)[:seq_len]
            augmented[b] = torch.cat([t.to(device), t.to(device)], dim=0)
            continue
        q = test_reps[b].float()
        dists = torch.norm(pool - q.unsqueeze(0), dim=1)
        best_idx = int(dists.argmin().item())
        retrieved = inps_pool[c][best_idx].float()
        test_ctx = test_inputs_1d[b].detach().float().reshape(-1)[:seq_len]
        augmented[b] = augment_context(test_ctx, retrieved).to(device=device, dtype=dtype)
    return augmented


# --------------------------------------------------------------------------- #
# RAFT: correlation-weighted average of retrieved futures, mixed with the backbone
# --------------------------------------------------------------------------- #
RAFT_TAU = 0.1


def build_or_load_raft_cache(args, backbone: str):
    """Per-channel list of ``(global_idx, history fp16 [P], future fp16 [F])``."""
    cache_path = os.path.join(
        args.cache_dir, backbone, "raft", args.data, f"{args.seq_len}_{args.pred_len}",
        f"raft_cache_pool{args.pool_number}.pt",
    )
    if os.path.exists(cache_path):
        print(f"Loaded RAFT cache: {cache_path}")
        return torch.load(cache_path)

    cache = {i: [] for i in range(args.num_channels)}
    for channel_ids, base_idx, x, y in iterate_train_windows(args):
        x_cpu = x.detach().cpu().to(torch.float16).contiguous()
        y_cpu = y.detach().cpu().to(torch.float16).contiguous()
        for b, c in enumerate(channel_ids):
            cache[c].append((base_idx + b, x_cpu[b].clone(), y_cpu[b].clone()))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved RAFT cache to {cache_path}")
    return cache


def build_raft_bank(cache, device, eps: float = 1e-8):
    """Per-channel keys (offset-subtracted, centred, unit-norm histories) and values."""
    bank = {}
    for c, items in cache.items():
        if len(items) == 0:
            bank[c] = None
            continue
        xs = torch.stack([item[1].detach().float().reshape(-1) for item in items], dim=0)  # [N, P]
        ys = torch.stack([item[2].detach().float().reshape(-1) for item in items], dim=0)  # [N, F]
        xs = xs.to(device=device, dtype=torch.float32)
        ys = ys.to(device=device, dtype=torch.float32)

        x_last = xs[:, -1:].clone()
        xs_hat = xs - x_last
        xs_hat = xs_hat - xs_hat.mean(dim=1, keepdim=True)
        xs_hat = xs_hat / (xs_hat.norm(dim=1, keepdim=True) + eps)
        bank[c] = {"xs_hat": xs_hat, "x_last": x_last, "ys": ys}
    return bank


def raft_predict_batch(batch_inputs, channel_ids, bank, *, top_m: int, pred_len: int, device,
                       tau: float = RAFT_TAU, eps: float = 1e-8):
    """RAFT forecast for a batch: softmax(top-m Pearson correlations / tau)-weighted
    average of the retrieved futures, with key- and query-offset correction."""
    batch_inputs = to_2d(batch_inputs)
    B = batch_inputs.size(0)
    raft_preds = torch.empty((B, pred_len), device=device, dtype=torch.float32)

    by_ch = {}
    for b in range(B):
        by_ch.setdefault(int(channel_ids[b]), []).append(b)

    for c, idxs in by_ch.items():
        entry = bank.get(c)
        if entry is None:
            raft_preds[idxs] = 0.0
            continue
        xs_hat, x_last, ys = entry["xs_hat"], entry["x_last"], entry["ys"]

        q = batch_inputs[idxs].detach().float().reshape(len(idxs), -1).to(device=device, dtype=torch.float32)
        q_last = q[:, -1:].clone()
        q_hat = q - q_last
        q_hat = q_hat - q_hat.mean(dim=1, keepdim=True)
        q_hat = q_hat / (q_hat.norm(dim=1, keepdim=True) + eps)

        corrs = q_hat @ xs_hat.t()                                   # [Bc, N]
        m = min(top_m, corrs.size(1))
        top_corr, top_idx = torch.topk(corrs, k=m, dim=1, largest=True)
        w = torch.softmax(top_corr / tau, dim=1)

        y_hat = (w.unsqueeze(-1) * (ys[top_idx] - x_last[top_idx])).sum(dim=1) + q_last  # [Bc, F]
        if y_hat.size(1) > pred_len:
            y_hat = y_hat[:, -pred_len:]
        elif y_hat.size(1) < pred_len:
            y_hat = torch.cat([y_hat, y_hat[:, -1:].repeat(1, pred_len - y_hat.size(1))], dim=1)
        raft_preds[idxs] = y_hat
    return raft_preds


def run_raft(args, backbone: str, predict_fn, device):
    """Evaluate ``(1 - gamma) * backbone + gamma * RAFT`` on the test split.

    ``predict_fn(batch)`` returns the backbone forecast [B, pred_len]; it is not
    called when ``gamma >= 1`` (pure retrieval).
    """
    cache = build_or_load_raft_cache(args, backbone)
    bank = build_raft_bank(cache, device=device)

    evaluator = Evaluator()
    with torch.no_grad():
        for batch in tqdm(test_loader(args)):
            labels = to_2d(batch["labels"].to(device=device).float())
            H = min(args.pred_len, labels.size(1))
            labels = labels[:, -H:]
            channel_ids = batch["channel_id"].detach().cpu().tolist()

            raft_preds = raft_predict_batch(batch["inputs"], channel_ids, bank, top_m=args.k,
                                            pred_len=H, device=device)
            if args.gamma >= 1.0:
                preds = raft_preds
            else:
                base_preds = to_2d(predict_fn(batch).to(device=device).float())[:, -H:]
                if args.gamma <= 0.0:
                    preds = base_preds
                else:
                    preds = (1.0 - args.gamma) * base_preds + args.gamma * raft_preds
            evaluator.push(preds, labels)

    save_results(args, backbone, "raft", f"k{args.k}_gamma{args.gamma}", evaluator.result())


def add_raft_args(parser, default_k: int):
    parser.add_argument("--k", type=int, default=default_k, help="Number of retrieved neighbours (top-m)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Weight of the retrieval forecast (0 = backbone only, 1 = retrieval only)")
    parser.add_argument("--pool_number", type=int, default=10000,
                        help="Use the last `pool_number` time steps of the training split as the database")
    return parser


# --------------------------------------------------------------------------- #
# Evaluation loop / results
# --------------------------------------------------------------------------- #
class Evaluator:
    """Accumulates MSE/MAE over forecast points and skips (and counts) NaN batches."""

    def __init__(self):
        self.metrics = [MSEMetric(name="mse"), MAEMetric(name="mae")]
        self.count = 0
        self.nan_batches = 0

    def push(self, preds, labels):
        if math.isnan(((preds - labels) ** 2).mean().item()):
            self.nan_batches += 1
            return
        for metric in self.metrics:
            metric.push(preds, labels)
        self.count += int(preds.numel())

    def result(self):
        out = {m.name: float(m.value / self.count) for m in self.metrics}
        out["nan_batches"] = self.nan_batches
        return out


def save_results(args, backbone: str, method: str, tag: str, result: dict):
    print(f"{backbone} {method} {args.data} {args.seq_len}->{args.pred_len}: "
          f"mse={result['mse']:.6f} mae={result['mae']:.6f} (NaN batches skipped: {result['nan_batches']})")
    if int(os.getenv("RANK") or 0) != 0:
        return
    out_dir = os.path.join(args.output_dir, backbone, method, args.data, f"{args.seq_len}_{args.pred_len}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), **result}, f, indent=2)
    print(f"Saved results to {out_path}")
