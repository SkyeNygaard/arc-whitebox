"""MLP generation and Monte-Carlo reference computation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLP:
    Ws: tuple[np.ndarray, ...]  # each (n, n): h_{l+1} = W_l @ a_l
    width: int
    depth: int
    seed: int

    @property
    def n(self) -> int:
        return self.width

    @property
    def L(self) -> int:
        return self.depth


def make_mlp(width: int = 256, depth: int = 32, seed: int = 0, dtype=np.float32) -> MLP:
    """He-Gaussian init: W_ij ~ N(0, 2/n)."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / width)
    Ws = tuple(
        (rng.standard_normal((width, width)) * scale).astype(dtype) for _ in range(depth)
    )
    return MLP(Ws=Ws, width=width, depth=depth, seed=seed)


def forward_means(mlp: MLP, X: np.ndarray, accum: np.ndarray | None = None) -> np.ndarray:
    """Sum of post-ReLU activations over the batch X (B, n) -> (L, n)."""
    H = X
    out = []
    for W in mlp.Ws:
        H = np.maximum(H @ W.T, 0.0)
        out.append(H.sum(axis=0))
    S = np.stack(out)
    return S if accum is None else accum + S


def reference_mc(
    mlp: MLP,
    n_samples: int,
    seed: int = 12345,
    chunk: int = 8192,
    dtype=np.float32,
    halves: bool = True,
) -> dict[str, np.ndarray]:
    """Monte-Carlo reference for Y[l,i] = E[ReLU(h_{l,i})].

    Returns two *independent* halves in addition to the pooled mean.  The two
    halves enable an unbiased MSE estimator that is immune to reference noise:

        MSE_hat = mean_i (yhat_i - yA_i) * (yhat_i - yB_i)

    which has expectation (yhat_i - y_i)^2 exactly, since E[yA] = E[yB] = y and
    the halves are independent.
    """
    rng = np.random.default_rng(seed)
    n = mlp.width
    Ws = [W.astype(dtype) for W in mlp.Ws]
    mlp_c = MLP(tuple(Ws), mlp.width, mlp.depth, mlp.seed)

    per_half = n_samples // 2 if halves else n_samples
    accs = []
    for _ in range(2 if halves else 1):
        acc = np.zeros((mlp.depth, n), dtype=np.float64)
        done = 0
        while done < per_half:
            b = min(chunk, per_half - done)
            X = rng.standard_normal((b, n)).astype(dtype)
            acc += forward_means(mlp_c, X).astype(np.float64)
            done += b
        accs.append(acc / per_half)

    if halves:
        A, B = accs
        return {"Y": 0.5 * (A + B), "Y_a": A, "Y_b": B, "n_samples": n_samples}
    return {"Y": accs[0], "Y_a": accs[0], "Y_b": accs[0], "n_samples": n_samples}


def unbiased_mse(yhat: np.ndarray, ref: dict, layer: int = -1) -> float:
    """Reference-noise-free MSE estimate on one layer (default: final)."""
    y = yhat[layer] if yhat.ndim == 2 else yhat
    a = ref["Y_a"][layer]
    b = ref["Y_b"][layer]
    return float(np.mean((y - a) * (y - b)))


def unbiased_mse_all(yhat: np.ndarray, ref: dict) -> np.ndarray:
    """Per-layer unbiased MSE, shape (L,)."""
    return np.mean((yhat - ref["Y_a"]) * (yhat - ref["Y_b"]), axis=1)


def ref_noise_var(ref: dict, layer: int = -1) -> float:
    """Estimated per-neuron variance of the *pooled* reference at `layer`."""
    d = ref["Y_a"][layer] - ref["Y_b"][layer]
    return float(np.mean(d * d) / 4.0)


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------
def cache_path(width: int, depth: int, seed: int, n_samples: int, root: str) -> str:
    return os.path.join(root, f"ref_w{width}_d{depth}_s{seed}_m{n_samples}.npz")


def load_or_build_reference(
    width: int, depth: int, seed: int, n_samples: int, root: str, **kw
) -> tuple[MLP, dict]:
    mlp = make_mlp(width, depth, seed)
    p = cache_path(width, depth, seed, n_samples, root)
    if os.path.exists(p):
        z = np.load(p)
        return mlp, {"Y": z["Y"], "Y_a": z["Y_a"], "Y_b": z["Y_b"], "n_samples": int(z["m"])}
    ref = reference_mc(mlp, n_samples, **kw)
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(p, Y=ref["Y"], Y_a=ref["Y_a"], Y_b=ref["Y_b"], m=n_samples)
    return mlp, ref
