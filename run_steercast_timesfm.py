#!/usr/bin/env python
"""SteerCast on a fine-tuned TimesFM 2.5 forecaster.

Example:
    python run_steercast_timesfm.py -m checkpoints/timesfm/ETTh1_p96 \
        -d dataset/ETT-small/ETTh1.csv --data ETTh1 -p 96 --k 4 --lam 0.01 -b 512
"""
import argparse

from steercast.backbones.timesfm import TimesFMBackbone
from steercast.pipeline import add_common_args, finalize_args, run


class TimesFMSteerBackbone:
    name = "timesfm"
    num_leading_states = 0  # states are collected at the output of each block

    def __init__(self, model_path, device):
        self.fm = TimesFMBackbone(model_path, device=device)
        self.device = self.fm.device
        self.dtype = self.fm.dtype
        self.hidden_size = self.fm.hidden_size

    def predict(self, inputs, pred_len):
        return self.fm.predict_series(inputs, pred_len)

    def key(self, inputs):
        return self.fm.get_rep(inputs)

    def key_and_states(self, inputs, full_series, pred_len):
        return self.fm.get_rep_with_hidden_states(inputs, full_series)

    def steer_targets(self):
        return self.fm.blocks

    def query(self, keys):
        return keys[0]  # the mini-batch is matched by its first window


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser("SteerCast / TimesFM 2.5"),
                             default_model="google/timesfm-2.5-200m-pytorch", whiten_default=False)
    parser.set_defaults(num_workers=0)
    args = finalize_args(parser.parse_args())
    run(TimesFMSteerBackbone(args.model, args.device), args)
