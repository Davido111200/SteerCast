"""Steering vectors from retrieved database entries (Eqs. 2-3 of the paper)."""
from typing import Sequence, Tuple

import torch

# A database entry: (global_idx, key [H], per-layer states [L*H], look-back window).
Entry = Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]


def _as_layers(h: torch.Tensor, num_layers: int, hidden_size: int) -> torch.Tensor:
    """View a packed ``[L*H]`` state as ``[L, H]`` (zero-pad / crop to ``num_layers``)."""
    flat = h.to(dtype=torch.float32).view(-1)
    assert flat.numel() % hidden_size == 0, (
        f"State with {flat.numel()} elements cannot be reshaped with H={hidden_size}")
    rows = flat.view(-1, hidden_size)
    if rows.shape[0] == num_layers:
        return rows
    if rows.shape[0] < num_layers:
        pad = torch.zeros(num_layers - rows.shape[0], hidden_size, dtype=rows.dtype, device=rows.device)
        return torch.cat([rows, pad], dim=0)
    return rows[:num_layers]


def build_steering_vector(gt_entries: Sequence[Entry], pred_entries: Sequence[Entry], hidden_size: int, *,
                          beta: float = 0.0, whiten: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Aggregate the steering vectors of the retrieved neighbours into one ``[L, H]`` tensor.

    ``gt_entries[i]`` and ``pred_entries[i]`` hold the per-layer last-token states of
    the same training history continued with the ground truth (``h_gt``) and with
    the model's own forecast (``h_pred``). The steering vector is the summed
    difference ``sum_i (h_gt_i - h_pred_i)``.

    Args:
        beta: same mixing weight as in :class:`steercast.steering.SteeringTail`. For
            ``beta > 0`` the vector is mixed with its per-layer unit-normalised copy.
        whiten: divide element-wise by ``sqrt(Var(h_pred) + eps)`` over the neighbours.

    Every downstream use normalises the vector per layer, so only its direction
    matters when ``beta`` is 0 or 1.
    """
    assert len(gt_entries) == len(pred_entries) > 0, (
        f"Mismatched or empty neighbour lists ({len(gt_entries)} vs {len(pred_entries)}).")
    assert 0.0 <= beta <= 1.0

    h0 = pred_entries[0][2]
    total = h0.view(-1).numel()
    assert total % hidden_size == 0, f"Packed state of length {total} is not divisible by H={hidden_size}"
    L, H = total // hidden_size, hidden_size
    device = h0.device

    V = torch.zeros(L, H, device=device)
    if whiten:
        pred_sum = torch.zeros(L, H, device=device)
        pred_sq_sum = torch.zeros(L, H, device=device)

    for gt_entry, pred_entry in zip(gt_entries, pred_entries):
        h_gt = _as_layers(gt_entry[2], L, H)
        h_pred = _as_layers(pred_entry[2], L, H)
        V += h_gt - h_pred
        if whiten:
            pred_sum += h_pred
            pred_sq_sum += h_pred * h_pred

    if whiten:
        n = float(len(pred_entries))
        pred_mean = pred_sum / n
        pred_var = torch.clamp(pred_sq_sum / n - pred_mean * pred_mean, min=0.0)
        V = V / torch.sqrt(pred_var + eps)

    if beta == 0.0:
        return V
    V_unit = V / (V.norm(dim=1, keepdim=True) + eps)
    if beta == 1.0:
        return V_unit
    return (1.0 - beta) * V + beta * V_unit
