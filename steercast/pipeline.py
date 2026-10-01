"""Database construction and steered evaluation shared by the three backbone runners.

A backbone adapter (see ``run_steercast_*.py``) provides:

* ``name``, ``device``, ``hidden_size``
* ``predict(inputs, pred_len) -> [B, pred_len]`` point forecasts on ``device``
* ``key(inputs) -> [B, H]`` retrieval keys (Eq. 1)
* ``key_and_states(inputs, full_series, pred_len) -> ([B, H], [B, L*H])``: the key of
  ``inputs`` and one hidden state per layer for ``full_series = [inputs; continuation]``
* ``steer_targets() -> list[nn.Module]``: one module per transformer block whose
  output is steered
* ``num_leading_states``: states in ``key_and_states`` that precede the first block
  (1 when the embedding output is included, else 0)
* ``query(keys) -> Tensor``: how a test mini-batch is matched against the database
"""
import argparse
import hashlib
import json
import os
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from steercast.metrics import MAEMetric, MSEMetric
from steercast.retrieval import retrieve, stack_keys
from steercast.steering import attach_steering, detach_steering
from steercast.vectors import build_steering_vector
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, BenchmarkEvalDatasetTrain

mp.set_sharing_strategy("file_system")

STANDARD_HORIZONS = (96, 192, 336, 720)


def add_common_args(parser: argparse.ArgumentParser, *, default_model=None, whiten_default=False):
    g = parser.add_argument_group("data")
    g.add_argument("--model", "-m", type=str, default=default_model, required=default_model is None,
                   help="Fine-tuned backbone (local path or Hugging Face id)")
    g.add_argument("--data_path", "-d", type=str, required=True, help="Benchmark CSV file")
    g.add_argument("--data", type=str, required=True, help="Dataset name used for cache and result paths")
    g.add_argument("--seq_len", "-c", type=int, default=None,
                   help="Look-back length (default: 512, or 96 for non-standard horizons such as Illness)")
    g.add_argument("--pred_len", "-p", type=int, default=96, help="Forecast horizon")
    g.add_argument("--batch_size", "-b", type=int, default=512,
                   help="Mini-batch size. Retrieval is done once per test mini-batch, so this "
                        "affects results; use the values of the provided scripts to reproduce the paper")
    g.add_argument("--num_workers", type=int, default=2)
    g.add_argument("--split", type=str, default="test", choices=["test", "val"],
                   help="Evaluation split; use `val` to select hyper-parameters such as --k")
    g.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    g = parser.add_argument_group("database")
    g.add_argument("--pool_number", type=int, default=10000,
                   help="Build the database from the last `pool_number` time steps of the training split")
    g.add_argument("--db_pred_len", type=int, default=None,
                   help="Continuation length used to build the database (default: --pred_len). "
                        "Set to 96 to reuse the 512->96 database for every horizon (fixed look-back protocol)")
    g.add_argument("--db_buffer", type=int, default=0,
                   help="Drop the last N training windows of every channel (strict temporal isolation)")
    g.add_argument("--cache_dir", type=str, default="./cache", help="Where databases are stored")

    g = parser.add_argument_group("steering")
    g.add_argument("--k", type=int, default=1, help="Number of retrieved neighbours")
    g.add_argument("--lam", type=float, default=0.01, help="Steering strength lambda")
    g.add_argument("--beta", type=float, default=0.0,
                   help="0 = plain norm-preserving update, 1 = cosine-gated update (Eqs. 4-5), "
                        "in between = interpolation")
    g.add_argument("--gate_b", type=float, default=0.1, help="Gate floor b")
    g.add_argument("--gate_m", type=float, default=0.1, help="Gate margin m")
    g.add_argument("--gate_p", type=float, default=1.25, help="Gate power p")
    g.add_argument("--retrieval", type=str, default="euclidean", choices=["euclidean", "cosine"])
    g.add_argument("--fast_retrieval", action="store_true",
                   help="Matrix-product Euclidean distances (faster; may reorder near-ties)")
    g.add_argument("--whiten", type=int, default=int(whiten_default), choices=[0, 1],
                   help="Whiten the steering vector by the per-dimension spread of h_pred over the neighbours")

    g = parser.add_argument_group("output")
    g.add_argument("--output_dir", type=str, default="./results")
    return parser


def finalize_args(args):
    if args.seq_len is None:
        args.seq_len = 512 if args.pred_len in STANDARD_HORIZONS else 96
    if args.db_pred_len is None:
        args.db_pred_len = args.pred_len
    if not args.data_path.endswith(".csv"):
        raise ValueError("SteerCast builds its database from a CSV benchmark file.")
    return args


def _model_tag(model: str) -> str:
    base = os.path.basename(os.path.normpath(model)) or model
    base = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in base)
    path = os.path.realpath(os.path.expanduser(model)) if os.path.exists(os.path.expanduser(model)) else model
    return f"{base}_{hashlib.md5(path.encode('utf-8')).hexdigest()[:8]}"


def database_dir(args, backbone) -> str:
    sub = f"{args.seq_len}_{args.db_pred_len}"
    if args.db_buffer:
        sub += f"_buf{args.db_buffer}"
    if args.pool_number != 10000:
        sub += f"_pool{args.pool_number}"
    return os.path.join(args.cache_dir, backbone.name, args.data, sub, _model_tag(args.model))


def _complete(db, num_channels):
    return isinstance(db, dict) and all(i in db for i in range(num_channels))


@torch.no_grad()
def build_database(backbone, args):
    """Build (or load) the per-channel database ``(gt, pred)``.

    ``gt[c]`` / ``pred[c]`` list one entry per training window of channel ``c``:
    ``(global_idx, key [H], states [L*H], look-back window)``, where the states
    come from continuing the window with the ground truth / the model's own
    forecast. Steering vectors are formed on the fly as ``h_gt - h_pred``.
    """
    out_dir = database_dir(args, backbone)
    gt_path = os.path.join(out_dir, "gt_cache.pt")
    pred_path = os.path.join(out_dir, "pred_cache.pt")

    dataset = BenchmarkEvalDatasetTrain(
        args, args.data_path, seq_len=args.seq_len, pred_len=args.db_pred_len,
        max_train_samples=args.pool_number, temporal_buffer=args.db_buffer,
    )
    num_channels = dataset.num_sequences
    num_each = len(dataset) // num_channels

    def _add(db, offset, rep, hs, inputs):
        rep = rep.detach().cpu().to(torch.float16).contiguous()
        hs = hs.detach().cpu().to(torch.float16).contiguous()
        inputs = inputs.detach().cpu().to(torch.float16).contiguous().clone()
        for b in range(rep.shape[0]):
            global_idx = offset + b
            channel = min(global_idx // num_each, num_channels - 1)
            db[channel].append((global_idx, rep[b], hs[b], inputs[b].clone()))
        return offset + rep.shape[0]

    start = time.perf_counter()
    pred = torch.load(pred_path) if os.path.exists(pred_path) else None
    if not _complete(pred, num_channels):
        pred = {c: [] for c in range(num_channels)}
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, prefetch_factor=2 if args.num_workers > 0 else None)
        offset = 0
        for batch in tqdm(loader, desc="Database (model continuation)", ncols=100):
            inputs = batch["inputs"]
            preds = backbone.predict(inputs, args.db_pred_len)
            full = torch.cat([inputs.to(device=preds.device), preds], dim=1)
            rep, hs = backbone.key_and_states(inputs, full, args.db_pred_len)
            offset = _add(pred, offset, rep, hs, inputs)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(pred, pred_path)
        print(f"Saved {pred_path}")
    else:
        print(f"Loaded {pred_path}")

    gt = torch.load(gt_path) if os.path.exists(gt_path) else None
    if not _complete(gt, num_channels):
        gt = {c: [] for c in range(num_channels)}
        offset = 0
        for x, y, _, _ in tqdm(dataset.data_loader, desc="Database (ground-truth continuation)", ncols=100):
            x = x.transpose(1, 2).squeeze(1)  # [B, seq_len]
            y = y.transpose(1, 2).squeeze(1)  # [B, db_pred_len]
            rep, hs = backbone.key_and_states(x, torch.cat([x, y], dim=-1), args.db_pred_len)
            offset = _add(gt, offset, rep, hs, x)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(gt, gt_path)
        print(f"Saved {gt_path}")
    else:
        print(f"Loaded {gt_path}")

    print(f"Database ready in {time.perf_counter() - start:.2f} s "
          f"({sum(len(v) for v in gt.values())} entries, {num_channels} channels)")
    return gt, pred


@torch.no_grad()
def evaluate(backbone, args, gt, pred):
    """Steered forecasting on the test split; returns ``{'mse', 'mae', ...}``."""
    dataset = BenchmarkEvalDataset(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers > 0 else None)
    keys = {c: stack_keys(entries) for c, entries in gt.items()}
    targets = backbone.steer_targets()
    skip = backbone.num_leading_states

    metrics = [MSEMetric("mse"), MAEMetric("mae")]
    count, nan_batches, infer_time, n_batches = 0, 0, 0.0, 0
    for batch in tqdm(loader, desc="SteerCast", ncols=100):
        t0 = time.perf_counter()
        # One retrieval per mini-batch, from the database of the batch's first channel.
        channel = int(batch["channel_id"][0])
        query = backbone.query(backbone.key(batch["inputs"]))
        _, idx = retrieve(query, keys[channel], args.k, metric=args.retrieval, fast=args.fast_retrieval)

        vector = build_steering_vector(
            [gt[channel][j] for j in idx], [pred[channel][j] for j in idx],
            hidden_size=backbone.hidden_size, beta=args.beta, whiten=bool(args.whiten),
        )
        vector = vector[skip:].to(device=backbone.device, dtype=backbone.dtype)

        attach_steering(targets, vector, lam=args.lam, beta=args.beta,
                        gate_b=args.gate_b, gate_m=args.gate_m, gate_p=args.gate_p, max_frac=1.0)
        try:
            preds = backbone.predict(batch["inputs"], args.pred_len)
        finally:
            detach_steering(targets)
        infer_time += time.perf_counter() - t0
        n_batches += 1

        labels = batch["labels"].to(preds.device)
        if preds.ndim > labels.ndim:
            labels = labels[..., None]
        elif labels.ndim > preds.ndim:
            labels = labels.squeeze(-1)
        if torch.isnan(preds).any():
            nan_batches += 1
            continue
        for m in metrics:
            m.push(preds, labels)
        count += preds.numel()

    if nan_batches:
        print(f"WARNING: skipped {nan_batches} mini-batches with NaN forecasts")
    result = {m.name: float(m.value / count) for m in metrics}
    result.update(nan_batches=nan_batches, sec_per_batch=infer_time / max(1, n_batches))
    return result


def save_result(args, backbone, result, method="steercast"):
    tag = f"k{args.k}_lam{args.lam}_beta{args.beta}"
    if args.beta > 0:
        tag += f"_b{args.gate_b}_m{args.gate_m}_p{args.gate_p}"
    if args.db_pred_len != args.pred_len:
        tag += f"_db{args.db_pred_len}"
    if args.db_buffer:
        tag += f"_buf{args.db_buffer}"
    if args.split != "test":
        tag += f"_{args.split}"
    out_dir = os.path.join(args.output_dir, backbone.name, method, args.data, f"{args.seq_len}_{args.pred_len}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), **result}, f, indent=2)
    print(json.dumps(result))
    print(f"Saved {path}")


def run(backbone, args):
    gt, pred = build_database(backbone, args)
    result = evaluate(backbone, args, gt, pred)
    save_result(args, backbone, result)
    return result
