#!/usr/bin/env python
"""SteerCast on a fine-tuned Time-MoE forecaster.

Example:
    python run_steercast_timemoe.py -m checkpoints/timemoe/ETTh1 \
        -d dataset/ETT-small/ETTh1.csv --data ETTh1 -p 96 --k 1 --lam 0.01 -b 512
"""
import argparse

import torch
from transformers import AutoModelForCausalLM

from steercast.pipeline import add_common_args, finalize_args, run
from steercast.steering import decoder_blocks, find_ffn


class TimeMoEBackbone:
    name = "timemoe"
    num_leading_states = 1  # hidden_states[0] is the embedding output

    def __init__(self, model_path, device):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(model_path, device_map=device, torch_dtype="auto")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device, torch_dtype="auto", trust_remote_code=True)
        self.model = model.eval()
        self.device = torch.device(device)
        self.dtype = model.dtype
        self.hidden_size = model.config.hidden_size
        print(f">>> Model dtype: {model.dtype}; attention: {model.config._attn_implementation}")

    def predict(self, inputs, pred_len):
        outputs = self.model.generate(inputs=inputs.to(self.device).to(self.dtype), max_new_tokens=pred_len)
        return outputs[:, -pred_len:]

    def _hidden_states(self, x):
        out = self.model(x.to(device=self.device, dtype=self.dtype), output_hidden_states=True,
                         output_attentions=False, use_cache=False, return_dict=True)
        return out.hidden_states

    def key(self, inputs):
        return self._hidden_states(inputs)[-1].mean(dim=1).contiguous()

    def key_and_states(self, inputs, full_series, pred_len):
        hidden = self._hidden_states(full_series)
        states = torch.stack([h[:, -1, :] for h in hidden], dim=1)  # last token of every layer
        states = states.reshape(states.size(0), -1).contiguous()
        return self.key(inputs), states

    def steer_targets(self):
        return [find_ffn(block) for block in decoder_blocks(self.model)]

    def query(self, keys):
        return keys  # the mini-batch is matched as a whole


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser("SteerCast / Time-MoE"), whiten_default=True)
    args = finalize_args(parser.parse_args())
    run(TimeMoEBackbone(args.model, args.device), args)
