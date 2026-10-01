"""Nearest-neighbour retrieval over a per-channel SteerCast database."""
import numpy as np
import torch


def stack_keys(entries) -> np.ndarray:
    """``[N, H]`` float32 matrix of the retrieval keys of a list of database entries."""
    return np.stack([e[1].detach().float().cpu().numpy().reshape(-1) for e in entries]).astype(np.float32)


def _cosine_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 1.0
    return 1.0 - float(np.dot(u, v) / (nu * nv + eps))


def retrieve(query: torch.Tensor, keys: np.ndarray, k: int, metric: str = "euclidean", fast: bool = False):
    """Return ``(distances, top_k_indices)`` of the ``k`` keys closest to ``query``.

    ``query`` is a ``[H]`` key or a ``[B, H]`` batch of keys. A batch is matched
    as a whole: for ``metric="euclidean"`` the distance to key ``r`` is
    ``||Q - r||_F = sqrt(sum_b ||q_b - r||^2)``; for ``metric="cosine"`` the batch
    keys are averaged first.

    Args:
        keys: ``[N, H]`` database keys, see :func:`stack_keys`.
        fast: compute Euclidean distances with one matrix product instead of one
            norm per key. Much faster on large databases, but rounding differs
            slightly, which can reorder near-ties.
    """
    q = query.detach().float().cpu().numpy()
    H = keys.shape[1]

    if metric == "euclidean":
        if fast:
            Q = q.reshape(-1, H).astype(np.float32)
            sq = float((Q ** 2).sum())
            cross = keys @ Q.sum(axis=0)
            d = np.sqrt(np.maximum(sq - 2.0 * cross + Q.shape[0] * (keys ** 2).sum(axis=1), 0.0))
        else:
            d = [np.linalg.norm(q - keys[j]) for j in range(len(keys))]
    elif metric == "cosine":
        u = np.asarray(q, dtype=np.float32).reshape(-1)
        if u.size != H:
            u = u.reshape(-1, H).mean(axis=0)
        d = [_cosine_distance(u, keys[j]) for j in range(len(keys))]
    else:
        raise ValueError(f"Unknown retrieval metric: {metric}")

    dists = np.asarray(d, dtype=float)
    order = np.argsort(dists)[:min(int(k), len(dists))]
    return dists, order
