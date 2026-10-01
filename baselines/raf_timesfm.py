#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""RAF baseline with a TimesFM 2.5 backbone.

Retrieval-augmented forecasting without training: for every test window, the
nearest training history (Euclidean distance between final-block hidden states
mean-pooled over patches) is normalised, aligned and prepended to the query
context, and the backbone forecasts from the doubled context.

    python -m baselines.raf_timesfm -m google/timesfm-2.5-200m-pytorch -d ETTh1.csv --data ETTh1 --num_channels 7 -p 96 -b 512
"""
import torch
from tqdm import tqdm

from baselines.common import (
    Evaluator, build_or_load_raf_cache, common_parser, finalize_args, get_device, raf_contexts,
    raf_pools, save_results, test_loader, to_2d,
)
from steercast.backbones.timesfm import TimesFMBackbone


def evaluate(args):
    device = get_device()
    model = TimesFMBackbone(args.model, device=device)
    cache = build_or_load_raf_cache(args, "timesfm", model.get_rep)
    reps_pool, inps_pool = raf_pools(cache, device)

    evaluator = Evaluator()
    with torch.no_grad():
        for batch in tqdm(test_loader(args)):
            labels = to_2d(batch["labels"].to(device=device).float())
            channel_ids = batch["channel_id"].detach().cpu().tolist()
            test_inputs = to_2d(batch["inputs"])

            test_reps = model.get_rep(test_inputs)
            augmented = raf_contexts(test_inputs, test_reps, channel_ids, reps_pool, inps_pool, args.seq_len)
            preds, _ = model.predict({"inputs": augmented, "labels": batch["labels"]}, pred_len=args.pred_len)
            preds = preds.to(device=device).float()

            H = min(args.pred_len, labels.size(1))
            evaluator.push(to_2d(preds)[:, -H:], labels[:, -H:])

    save_results(args, "timesfm", "raf", "top1", evaluator.result())


if __name__ == "__main__":
    parser = common_parser("RAF baseline (TimesFM 2.5)", default_model="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--pool_number", type=int, default=10000,
                        help="Use the last `pool_number` time steps of the training split as the database")
    evaluate(finalize_args(parser.parse_args()))
