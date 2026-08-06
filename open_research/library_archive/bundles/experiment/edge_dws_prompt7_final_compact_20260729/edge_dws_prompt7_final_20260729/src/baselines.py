from __future__ import annotations

import numpy as np


def invariant_features(weights: np.ndarray, layer_observables: np.ndarray | None = None) -> np.ndarray:
    """Permutation-invariant weight summaries, one row per example."""
    n, depth, width, _ = weights.shape
    feats = []
    for i in range(n):
        row = []
        for l in range(depth):
            w = weights[i, l].astype(np.float64, copy=False)
            z = (w - w.mean()) / max(w.std(), 1e-12)
            rn = np.linalg.norm(w, axis=1)
            cn = np.linalg.norm(w, axis=0)
            row.extend([
                w.mean(), w.std(), np.mean(np.abs(w)), np.mean(z ** 3), np.mean(z ** 4) - 3,
                rn.mean(), rn.std(), rn.min(), rn.max(),
                cn.mean(), cn.std(), cn.min(), cn.max(),
            ])
            if layer_observables is not None:
                o = layer_observables[i, l].astype(np.float64, copy=False)
                row.extend([o.mean(), o.std(), o.min(), o.max()])
        feats.append(row)
    x = np.asarray(feats, dtype=np.float64)
    return x


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, np.ndarray | float]:
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    gram = z.T @ z
    reg = ridge * np.eye(gram.shape[0])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(gram + reg, z.T @ y)
    return {"mu": mu, "sd": sd, "coef": coef, "ridge": float(ridge)}


def ridge_predict(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    z = (x - model["mu"]) / model["sd"]
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    return z @ model["coef"]


def fit_anchor_shrink(anchor: np.ndarray, e0: np.ndarray, j: np.ndarray, grid: np.ndarray | None = None) -> float:
    if grid is None:
        grid = np.linspace(-0.5, 1.5, 401)
    best_alpha, best = 0.0, float("inf")
    for alpha in grid:
        err = e0 + np.einsum("nod,nd->no", j, alpha * anchor)
        loss = float(np.mean(err * err))
        if loss < best:
            best, best_alpha = loss, float(alpha)
    return best_alpha
