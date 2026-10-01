#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""RAFT baseline with a TimesFM 2.5 backbone.

RAFT retrieves the top-k training histories most correlated with the query
(Pearson correlation of offset-subtracted windows), averages their futures with
softmax weights, and mixes the result with the backbone forecast:
``(1 - gamma) * backbone + gamma * retrieval``.

    python -m baselines.raft_timesfm -m <ckpt> -d ETTh1.csv --data ETTh1 --num_channels 7 -p 96 -b 512
"""
from baselines.common import add_raft_args, common_parser, finalize_args, get_device, run_raft
from steercast.backbones.timesfm import TimesFMBackbone


def evaluate(args):
    device = get_device()
    model = TimesFMBackbone(args.model, device=device)

    def predict_fn(batch):
        preds, _ = model.predict(batch, pred_len=args.pred_len)
        return preds

    run_raft(args, "timesfm", predict_fn, device)


if __name__ == "__main__":
    parser = common_parser("RAFT baseline (TimesFM 2.5)", default_model="google/timesfm-2.5-200m-pytorch")
    add_raft_args(parser, default_k=4)
    evaluate(finalize_args(parser.parse_args()))
