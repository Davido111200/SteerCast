#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""RAF baseline with a Timer-XL backbone.

Retrieval-augmented forecasting without training: for every test window, the
nearest training history (Euclidean distance between mean-pooled final-layer
hidden states) is normalised, aligned and prepended to the query context, and
the backbone forecasts from the doubled context.

    python -m baselines.raf_timerxl -m <ckpt> -d ETTh1.csv --data ETTh1 --num_channels 7 -p 96 -b 512
"""
import torch
from tqdm import tqdm

from baselines.common import (
    Evaluator, build_or_load_raf_cache, common_parser, finalize_args, get_device, load_timerxl,
    model_device_dtype, raf_contexts, raf_pools, save_results, test_loader, to_2d,
)


def get_rep(model, series: torch.Tensor) -> torch.Tensor:
    """Mean-pooled final-layer hidden state, [B, H]."""
    device, dtype = model_device_dtype(model)
    out = model(series.to(device=device, dtype=dtype), output_hidden_states=True, use_cache=False, return_dict=True)
    return out.hidden_states[-1].mean(dim=1).contiguous()


def predict(model, inputs: torch.Tensor, pred_len: int) -> torch.Tensor:
    device, dtype = model_device_dtype(model)
    outputs = model.generate(inputs=inputs.to(device=device, dtype=dtype), max_new_tokens=pred_len)
    return outputs[:, -pred_len:].float()


def evaluate(args):
    device = get_device()
    model = load_timerxl(args.model, str(device))
    cache = build_or_load_raf_cache(args, "timerxl", lambda x: get_rep(model, x))
    reps_pool, inps_pool = raf_pools(cache, device)
    dev, dt = model_device_dtype(model)

    evaluator = Evaluator()
    with torch.no_grad():
        for batch in tqdm(test_loader(args)):
            labels = to_2d(batch["labels"].to(device=device).float())
            channel_ids = batch["channel_id"].detach().cpu().tolist()
            test_inputs = to_2d(batch["inputs"])

            test_reps = get_rep(model, test_inputs)
            augmented = raf_contexts(test_inputs, test_reps, channel_ids, reps_pool, inps_pool,
                                     args.seq_len, dtype=dt, device=dev)
            preds = predict(model, augmented, pred_len=args.pred_len).to(device=device).float()

            H = min(args.pred_len, labels.size(1))
            evaluator.push(to_2d(preds)[:, -H:], labels[:, -H:])

    save_results(args, "timerxl", "raf", "top1", evaluator.result())


if __name__ == "__main__":
    parser = common_parser("RAF baseline (Timer-XL)", default_model="thuml/timer-base-84m")
    parser.add_argument("--pool_number", type=int, default=10000,
                        help="Use the last `pool_number` time steps of the training split as the database")
    evaluate(finalize_args(parser.parse_args()))
