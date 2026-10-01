#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Thin wrapper around TimesFM 2.5 (PyTorch) exposing what SteerCast needs.

TimesFM is imported from the installed ``timesfm`` package. If it is not
installed, set ``TIMESFM_SRC`` to the ``src`` directory of a clone of
https://github.com/google-research/timesfm.
"""
import os
import sys
from pathlib import Path

import torch


def _import_timesfm():
    try:
        import timesfm  # noqa: F401
    except ImportError:
        src = os.environ.get("TIMESFM_SRC")
        if not src or not os.path.isdir(src):
            raise ImportError(
                "TimesFM is not installed. Install it from "
                "https://github.com/google-research/timesfm (with the torch extra), "
                "or set TIMESFM_SRC to the `src` directory of a local clone."
            )
        sys.path.insert(0, src)
    import timesfm
    from timesfm.torch import util as timesfm_torch_util
    return timesfm, timesfm_torch_util


def _find_snapshot_dir(repo_id: str):
    """Locate a downloaded Hugging Face snapshot of ``repo_id`` in the local cache."""
    repo_key = repo_id.replace("/", "--")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_roots = [Path(os.environ.get("HF_HUB_CACHE", Path(hf_home) / "hub"))]
    for cache_root in cache_roots:
        snapshot_root = cache_root / f"models--{repo_key}" / "snapshots"
        if not snapshot_root.is_dir():
            continue
        for snapshot_dir in sorted(snapshot_root.iterdir(), reverse=True):
            if (snapshot_dir / "model.safetensors").is_file():
                return snapshot_dir
    return None


def _resolve_local_checkpoint(model_path: str):
    path = Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    if path.is_dir() and (path / "model.safetensors").is_file():
        return path / "model.safetensors"
    snapshot_dir = _find_snapshot_dir(model_path)
    if snapshot_dir is not None:
        return snapshot_dir / "model.safetensors"
    return None


class TimesFMBackbone:
    """TimesFM 2.5 (200M, PyTorch) with hidden-state access and point forecasts.

    ``model_path`` is either a local fine-tuned checkpoint (a directory containing
    ``model.safetensors`` or the file itself) or a Hugging Face repo id such as
    ``google/timesfm-2.5-200m-pytorch``.
    """

    def __init__(self, model_path: str, device):
        timesfm, timesfm_torch_util = _import_timesfm()
        self.timesfm = timesfm
        self.timesfm_torch_util = timesfm_torch_util
        self.device = torch.device(device)

        local_checkpoint = _resolve_local_checkpoint(model_path)
        if local_checkpoint is not None:
            wrapper = timesfm.TimesFM_2p5_200M_torch()
            try:
                wrapper.model.load_checkpoint(str(local_checkpoint), torch_compile=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to load TimesFM 2.5 weights from {local_checkpoint}. "
                    "This backbone expects the official TimesFM 2.5 PyTorch checkpoint "
                    "(for example `google/timesfm-2.5-200m-pytorch`)."
                ) from exc
        else:
            wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path, torch_compile=False)

        self.wrapper = wrapper
        self.model = wrapper.model.to(self.device)
        self.model.eval()

        param = next(self.model.parameters())
        self.dtype = param.dtype
        self.patch_len = int(self.model.p)
        self.output_patch_len = int(self.model.o)
        self.num_quantiles = int(self.model.q)
        self.pred_index = int(getattr(self.model, "aridx", 5))
        self.hidden_size = int(self.model.md)
        self.num_layers = int(len(self.model.stacked_xf))

    @property
    def blocks(self):
        """Transformer blocks; SteerCast steers the output of each block."""
        return list(self.model.stacked_xf)

    def _to_2d_series(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        elif x.ndim != 2:
            raise ValueError(f"Expected [B,T] or [B,T,1], got {tuple(x.shape)}")
        return x.to(device=self.device, dtype=torch.float32)

    def _prepare_inputs(self, x: torch.Tensor):
        """Left-pad the context to a multiple of the patch length and build the padding mask."""
        x = self._to_2d_series(x)
        batch_size, context_len = x.shape
        pad_left = (self.patch_len - (context_len % self.patch_len)) % self.patch_len
        if pad_left > 0:
            pad = torch.zeros((batch_size, pad_left), device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
        mask = torch.zeros((batch_size, x.shape[1]), device=x.device, dtype=torch.bool)
        if pad_left > 0:
            mask[:, :pad_left] = True
        return x, mask, context_len

    def _patch_and_normalize(self, x: torch.Tensor, mask: torch.Tensor):
        B = x.shape[0]
        patched_inputs = x.view(B, -1, self.patch_len)
        patched_masks = mask.view(B, -1, self.patch_len)

        n = torch.zeros(B, device=x.device)
        mu = torch.zeros(B, device=x.device)
        sigma = torch.zeros(B, device=x.device)
        patch_mu = []
        patch_sigma = []
        for i in range(patched_inputs.shape[1]):
            (n, mu, sigma), _ = self.timesfm_torch_util.update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        normed_inputs = self.timesfm_torch_util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, torch.zeros_like(normed_inputs), normed_inputs)
        return normed_inputs, patched_masks, context_mu, context_sigma

    def _collect_hidden_states(self, series: torch.Tensor):
        """Outputs of every transformer block for one forward pass: L x [B, N_patches, H]."""
        x, mask, _ = self._prepare_inputs(series)
        normed_inputs, patched_masks, _, _ = self._patch_and_normalize(x, mask)

        hidden_states = []
        handles = []

        def _hook(_module, _inputs, output):
            y = output[0] if isinstance(output, tuple) else output
            hidden_states.append(y.detach())

        for layer in self.model.stacked_xf:
            handles.append(layer.register_forward_hook(_hook))
        try:
            self.model(normed_inputs, patched_masks, decode_caches=None)
        finally:
            for handle in handles:
                handle.remove()
        return hidden_states

    def _forward_point_outputs(self, series: torch.Tensor) -> torch.Tensor:
        x, mask, _ = self._prepare_inputs(series)
        normed_inputs, patched_masks, context_mu, context_sigma = self._patch_and_normalize(x, mask)
        (_emb, _hidden, output_ts, _quant), _decode_caches = self.model(
            normed_inputs, patched_masks, decode_caches=None,
        )
        renormed_outputs = self.timesfm_torch_util.revin(
            output_ts, context_mu, context_sigma, reverse=True,
        ).reshape(x.shape[0], -1, self.output_patch_len, self.num_quantiles)
        return renormed_outputs

    @torch.no_grad()
    def predict_series(self, series: torch.Tensor, pred_len: int) -> torch.Tensor:
        x, mask, _ = self._prepare_inputs(series)
        pf_outputs, _quantiles, ar_outputs = self.model.decode(pred_len, x, mask)
        pieces = [pf_outputs[:, -1, ...]]
        if ar_outputs is not None:
            pieces.append(ar_outputs.reshape(x.shape[0], -1, self.model.q))
        full_forecast = torch.cat(pieces, dim=1)
        return full_forecast[:, :pred_len, self.pred_index]

    def predict_series_trainable(self, series: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Differentiable autoregressive point forecast (used for fine-tuning)."""
        current = self._to_2d_series(series)
        chunks = []
        while sum(chunk.shape[1] for chunk in chunks) < pred_len:
            renormed_outputs = self._forward_point_outputs(current)
            next_patch = renormed_outputs[:, -1, :, self.pred_index]
            chunks.append(next_patch)
            current = torch.cat([current, next_patch], dim=1)
        return torch.cat(chunks, dim=1)[:, :pred_len]

    @torch.no_grad()
    def predict(self, batch: dict, pred_len: int):
        preds = self.predict_series(batch["inputs"], pred_len)
        labels = batch["labels"].to(device=self.device)
        if labels.ndim == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        return preds, labels

    @torch.no_grad()
    def get_rep(self, series: torch.Tensor) -> torch.Tensor:
        """Retrieval key: final-block hidden states mean-pooled over patches, [B, H]."""
        hidden_states = self._collect_hidden_states(series)
        return hidden_states[-1].mean(dim=1).contiguous()

    @torch.no_grad()
    def get_rep_with_hidden_states(self, input_only: torch.Tensor, full_series: torch.Tensor):
        """Return ``(key, h)`` for one database entry.

        ``key`` is the retrieval key of ``input_only`` ([B, H]); ``h`` stacks one
        state per block for ``full_series`` ([B, L*H]). TimesFM attends over
        patches without a privileged last position, so each block's state is
        mean-pooled over patches rather than read at the last token.
        """
        hidden_full = self._collect_hidden_states(full_series)
        hidden_input = self._collect_hidden_states(input_only)

        patch_means = [h.mean(dim=1) for h in hidden_full]          # L x [B, H]
        hs_layers = torch.stack(patch_means, dim=1).contiguous()    # [B, L, H]
        hs = hs_layers.view(hs_layers.size(0), -1).contiguous()     # [B, L*H]

        rep = hidden_input[-1].mean(dim=1).contiguous()
        return rep, hs
