#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""RAFT baseline with a Time-MoE backbone.

RAFT retrieves the top-k training histories most correlated with the query
(Pearson correlation of offset-subtracted windows), averages their futures with
softmax weights, and mixes the result with the backbone forecast:
``(1 - gamma) * backbone + gamma * retrieval``.

    python -m baselines.raft_timemoe -m <ckpt> -d ETTh1.csv --data ETTh1 --num_channels 7 -p 96 -b 512
"""
from baselines.common import (
    add_raft_args, common_parser, finalize_args, get_device, load_timemoe, model_device_dtype, run_raft,
)


def evaluate(args):
    device = get_device()
    model = load_timemoe(args.model, str(device))
    dev, dt = model_device_dtype(model)

    def predict_fn(batch):
        outputs = model.generate(inputs=batch["inputs"].to(dev).to(dt), max_new_tokens=args.pred_len)
        return outputs[:, -args.pred_len:]

    run_raft(args, "timemoe", predict_fn, device)


if __name__ == "__main__":
    parser = common_parser("RAFT baseline (Time-MoE)", default_model="Maple728/TimeMoE-50M")
    add_raft_args(parser, default_k=4)
    evaluate(finalize_args(parser.parse_args()))
