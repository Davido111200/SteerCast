#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""FT baseline: evaluate a (fine-tuned) Time-MoE checkpoint without retrieval.

For Timer-XL and TimesFM the FT numbers are obtained with the SteerCast runners
and ``--lam 0`` (no steering).

    python -m baselines.ft_timemoe -m <ckpt> -d ETTh1.csv --data ETTh1 -p 96 -b 512
"""
import torch
from tqdm import tqdm

from baselines.common import (
    Evaluator, common_parser, finalize_args, get_device, load_timemoe, save_results, test_loader,
)


def evaluate(args):
    device = get_device()
    model = load_timemoe(args.model, str(device))

    evaluator = Evaluator()
    with torch.no_grad():
        for batch in tqdm(test_loader(args)):
            outputs = model.generate(inputs=batch["inputs"].to(device).to(model.dtype), max_new_tokens=args.pred_len)
            preds = outputs[:, -args.pred_len:]
            labels = batch["labels"].to(device)
            if len(preds.shape) > len(labels.shape):
                labels = labels[..., None]
            evaluator.push(preds, labels)

    save_results(args, "timemoe", "ft", "ft", evaluator.result())


if __name__ == "__main__":
    parser = common_parser("FT baseline (Time-MoE)", default_model="Maple728/TimeMoE-50M", with_database=False)
    evaluate(finalize_args(parser.parse_args()))
