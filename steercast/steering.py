"""Inference-time injection of steering vectors into a decoder-only forecaster.

A :class:`SteeringTail` is appended to a module inside every transformer block
(the feed-forward sub-layer for Time-MoE / Timer-XL, the whole block for
TimesFM). It edits that module's output at every position and every
autoregressive step, using the per-layer steering direction ``delta_l``.

Two update rules are implemented and mixed with ``beta`` in [0, 1]:

* plain (``beta = 0``), norm-preserving::

      y   = lam * (1 + ReLU(-cos(h, delta_l))) * delta_l / ||delta_l||
      h'  = ||h|| * normalize(normalize(h) + y)

* gated (``beta = 1``), the cosine gate of Eq. (4) and normalised update of
  Eq. (5) in the paper::

      alpha = b + ReLU(-cos(h, delta_l) + m) ** p
      h'    = (h + lam * alpha * delta_l / ||delta_l||) * (1 + s * alpha)

  where the update is clamped element-wise to ``max_frac * ||h||`` and ``s`` is
  a small output stretch (``stretch``, 0.1 by default).

For ``0 < beta < 1`` the two outputs are interpolated: ``(1 - beta) * plain + beta * gated``.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

FFN_KEYWORDS = ("ffn_layer", "mlp", "feedforward")


class SteeringTail(nn.Module):
    """Edits a hidden-state tensor ``[B, T, H]`` (or ``[B, H]``) along a steering direction.

    Args:
        vector: steering vector(s) for this layer, ``[H]`` or ``[K, H]``.
        lam: global steering strength (lambda).
        dtype: dtype of the returned hidden states.
        beta: mix between the plain (0) and gated (1) update rules.
        gate_b, gate_m, gate_p: floor ``b``, margin ``m`` and power ``p`` of the cosine gate.
        renorm: post-processing of the gated output: ``"stretch"`` (default) or ``"keep"``
            (rescale to the original norm).
        stretch: stretch coefficient ``s`` used when ``renorm="stretch"``.
        max_frac: clamp of the gated update relative to ``||h||`` (``None`` disables it).
    """

    def __init__(self, vector: torch.Tensor, lam, dtype=torch.float32, *, beta: float = 0.0,
                 gate_b: float = 0.1, gate_m: float = 0.1, gate_p: float = 1.25,
                 renorm: str = "stretch", stretch: float = 0.10, max_frac=None, eps: float = 1e-8):
        super().__init__()
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        assert vector.dim() == 2, f"vector must be [K, H] or [H], got {tuple(vector.shape)}"
        self.register_buffer("vector", vector)

        if isinstance(lam, (list, tuple)):
            lam = torch.tensor(lam, dtype=torch.float32)
        elif not isinstance(lam, torch.Tensor):
            lam = torch.tensor([float(lam)], dtype=torch.float32)
        if lam.dim() == 0:
            lam = lam.view(1)
        self.register_buffer("lam", lam)

        self.dtype = dtype
        self.beta = float(beta)
        self.gate_b = float(gate_b)
        self.gate_m = float(gate_m)
        self.gate_p = float(gate_p)
        self.renorm = renorm
        self.stretch = float(stretch)
        self.max_frac = max_frac
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.dim() == 2
        xb = x.float().unsqueeze(1) if squeezed else x.float()
        if xb.dim() != 3:
            raise ValueError(f"x must be [B, H] or [B, T, H], got {tuple(x.shape)}")
        B, T, H = xb.shape
        orig_norm = xb.norm(dim=-1, keepdim=True) + self.eps

        # Unit steering directions [K, H] and their cosine with every hidden state [K, B, T].
        d = self.vector.float()
        d = d / (d.norm(dim=-1, keepdim=True) + self.eps)
        K = d.shape[0]
        dk = d.view(K, 1, 1, H).expand(-1, B, T, -1)
        cos = F.cosine_similarity(xb.unsqueeze(0).expand(K, -1, -1, -1), dk, dim=-1)
        cos_neg = -cos

        # Plain rule: norm-preserving rotation towards the steering direction.
        lam = self.lam.to(xb.device, dtype=torch.float32)
        y_plain = (lam.view(K, 1, 1) * (1.0 + F.relu(cos_neg))).unsqueeze(-1) * dk
        y_plain = y_plain.mean(dim=0)
        out_plain = F.normalize(F.normalize(xb, dim=-1) + y_plain, dim=-1) * orig_norm
        if self.beta <= 0.0:
            return self._restore(out_plain, squeezed)

        # Gated rule: alpha = b + ReLU(-cos + m)^p, additive update along the unit direction.
        gate = self.gate_b + F.relu(cos_neg + self.gate_m).pow(self.gate_p)
        w = lam.view(K)
        if K == 1:
            global_gain = float(w.item())
            w = torch.ones_like(w)
        else:
            global_gain = 1.0
            w = w / (w.sum() + self.eps)
        gate = gate * w.view(K, 1, 1)
        delta = (gate.unsqueeze(-1) * dk).sum(dim=0) * global_gain
        if self.max_frac is not None:
            max_delta = self.max_frac * orig_norm
            delta = torch.clamp(delta, min=-max_delta, max=max_delta)

        out_gated = xb + delta
        if self.renorm == "keep":
            out_gated = F.normalize(out_gated, dim=-1) * orig_norm
        elif self.renorm == "stretch":
            factor = (1.0 + self.stretch * gate.mean(dim=0).unsqueeze(-1)).clamp(min=0.0)
            out_gated = out_gated * factor
        if self.beta >= 1.0:
            return self._restore(out_gated, squeezed)

        return self._restore((1.0 - self.beta) * out_plain + self.beta * out_gated, squeezed)

    def _restore(self, out, squeezed):
        return (out.squeeze(1) if squeezed else out).to(self.dtype)


def _longest_modulelist(module: nn.Module):
    best, best_len = None, 0
    for _, child in module.named_modules():
        if isinstance(child, nn.ModuleList) and len(child) > best_len:
            best, best_len = child, len(child)
    return best


def decoder_blocks(model: nn.Module):
    """Transformer blocks of a Hugging Face decoder (the longest ``nn.ModuleList``)."""
    blocks = _longest_modulelist(model)
    if blocks is None:
        raise ValueError(f"No nn.ModuleList found in {type(model).__name__}")
    return list(blocks)


def find_ffn(block: nn.Module) -> nn.Module:
    """Feed-forward sub-layer of a transformer block (``ffn_layer`` for Time-MoE and Timer-XL)."""
    if isinstance(getattr(block, "ffn_layer", None), nn.Module):
        return block.ffn_layer
    for name, module in block.named_modules():
        if any(k in name for k in FFN_KEYWORDS):
            return module
    raise ValueError(f"Could not find a feed-forward module in {type(block).__name__}")


def _forward_with_tail(self, *args, **kwargs):
    out = self._steercast_orig_forward(*args, **kwargs)
    tail = getattr(self, "steer_tail", None)
    if tail is None:
        return out
    if isinstance(out, tuple):
        return (tail(out[0]), *out[1:])
    return tail(out)


def attach_steering(targets, vectors, *, lam, beta=0.0, gate_b=0.1, gate_m=0.1, gate_p=1.25,
                    max_frac=1.0, skip_zero=True):
    """Attach one :class:`SteeringTail` per module in ``targets``.

    Args:
        targets: list of modules whose outputs are steered, one per transformer block.
        vectors: per-layer steering vectors, ``[L, H]`` or ``[L, K, H]`` (``L == len(targets)``).
        skip_zero: leave a module untouched when its steering vector is all zeros.
    """
    if len(vectors) != len(targets):
        raise ValueError(f"Got {len(vectors)} steering vectors for {len(targets)} layers.")
    for module, v in zip(targets, vectors):
        if skip_zero and v.abs().max().item() < 1e-8:
            continue
        p = next(module.parameters())
        dev, dt = p.device, p.dtype
        module.steer_tail = SteeringTail(
            v.to(device=dev, dtype=dt), lam, dtype=dt, beta=beta,
            gate_b=gate_b, gate_m=gate_m, gate_p=gate_p, max_frac=max_frac,
        ).to(device=dev, dtype=dt)
        if not hasattr(module, "_steercast_orig_forward"):
            module._steercast_orig_forward = module.forward
            module.forward = types.MethodType(_forward_with_tail, module)


def detach_steering(targets):
    """Undo :func:`attach_steering`."""
    for module in targets:
        if hasattr(module, "_steercast_orig_forward"):
            module.forward = module._steercast_orig_forward
            del module._steercast_orig_forward
        if hasattr(module, "steer_tail"):
            del module.steer_tail
