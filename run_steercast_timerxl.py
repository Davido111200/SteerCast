#!/usr/bin/env python
"""SteerCast on a fine-tuned Timer-XL forecaster.

Example:
    python run_steercast_timerxl.py -m checkpoints/timerxl/ETTh1/epoch-1 \
        -d dataset/ETT-small/ETTh1.csv --data ETTh1 -p 96 --k 1 --lam 0.01 -b 1024
"""
import argparse

import torch
from transformers import AutoModelForCausalLM

from steercast.pipeline import add_common_args, finalize_args, run
from steercast.steering import decoder_blocks, find_ffn

SHORT_HORIZONS = (24, 36, 48, 60)


class TimerXLBackbone:
    name = "timerxl"
    num_leading_states = 1  # hidden_states[0] is the embedding output

    def __init__(self, model_path, device):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.dtype = next(model.parameters()).dtype
        self.hidden_size = model.config.hidden_size

    def predict(self, inputs, pred_len):
        outputs = self.model.generate(inputs=inputs.to(device=self.device, dtype=self.dtype),
                                      max_new_tokens=pred_len)
        return outputs[:, -pred_len:]

    def _hidden_states(self, x):
        out = self.model(x.to(device=self.device, dtype=self.dtype), output_hidden_states=True,
                         output_attentions=False, use_cache=False, return_dict=True)
        return out.hidden_states

    def key(self, inputs):
        return self._hidden_states(inputs)[-1].mean(dim=1).contiguous()

    def key_and_states(self, inputs, full_series, pred_len):
        x = full_series.to(device=self.device, dtype=self.dtype)
        if pred_len in SHORT_HORIZONS:
            # Timer-XL consumes 96-step tokens: right-pad short continuations to one full token.
            x = torch.cat([x, torch.zeros((x.shape[0], 96 - pred_len), device=x.device, dtype=x.dtype)], dim=1)
        hidden = self._hidden_states(x)
        states = torch.stack([h[:, -1, :] for h in hidden], dim=1)  # last token of every layer
        states = states.reshape(states.size(0), -1).contiguous()
        return self.key(inputs), states

    def steer_targets(self):
        return [find_ffn(block) for block in decoder_blocks(self.model)]

    def query(self, keys):
        return keys[0]  # the mini-batch is matched by its first window


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser("SteerCast / Timer-XL"), whiten_default=False)
    args = finalize_args(parser.parse_args())
    run(TimerXLBackbone(args.model, args.device), args)
