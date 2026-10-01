#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""RAFT baseline with a Timer-XL backbone.

RAFT retrieves the top-k training histories most correlated with the query
(Pearson correlation of offset-subtracted windows), averages their futures with
softmax weights, and mixes the result with the backbone forecast:
``(1 - gamma) * backbone + gamma * retrieval``.

    python -m baselines.raft_timerxl -m <ckpt> -d ETTh1.csv --data ETTh1 --num_channels 7 -p 96 -b 1024
"""
from baselines.common import (
    add_raft_args, common_parser, finalize_args, get_device, load_timerxl, model_device_dtype, run_raft,
)


def evaluate(args):
    device = get_device()
    model = load_timerxl(args.model, device)
    dev, dt = model_device_dtype(model)

    def predict_fn(batch):
        outputs = model.generate(inputs=batch["inputs"].to(device=dev, dtype=dt), max_new_tokens=args.pred_len)
        return outputs[:, -args.pred_len:]

    run_raft(args, "timerxl", predict_fn, device)


if __name__ == "__main__":
    parser = common_parser("RAFT baseline (Timer-XL)", default_model="thuml/timer-base-84m")
    add_raft_args(parser, default_k=20)
    evaluate(finalize_args(parser.parse_args()))
