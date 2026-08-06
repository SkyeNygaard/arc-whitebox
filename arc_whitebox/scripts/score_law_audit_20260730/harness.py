"""Independent re-derivation of the shipped 129-basis Kerdock estimator.

Produces, per network, the 129 per-basis final-layer group means G (129,256)
alongside the baked ground truth, so that:
  * the baseline MSE can be reproduced exactly,
  * the "direct-output source" of the v26 ledger can be rebuilt, and
  * coefficient estimability can be tested honestly.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/Users/skyenygaard/Programming/AI-Safety")
DATA = ROOT / "arc_whitebox/data/official_phase1_mini/data"
ASSET = ROOT / "whestbench_final_estimator_20260730/kerdock_mub5_seed3.npz"

WIDTH = 256
DEPTH = 32
KERDOCK_BASES = 128
TOTAL_ROWS = 66_048
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def mean_gaussian_radius(width: int) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma((width + 1.0) / 2.0) - math.lgamma(width / 2.0)
    )


RADIUS = mean_gaussian_radius(WIDTH)
DESIGN_SCALE = RADIUS / math.sqrt(WIDTH)


def fwht_axis_one(values: np.ndarray) -> np.ndarray:
    """Batched FWHT along the direction axis -- mirrors estimator._fwht_axis_one."""
    span = 1
    while span < WIDTH:
        grouped = values.reshape((KERDOCK_BASES, WIDTH // (2 * span), 2, span, WIDTH))
        left = grouped[:, :, 0, :, :]
        right = grouped[:, :, 1, :, :]
        values = np.stack((left + right, left - right), axis=2).reshape(
            (KERDOCK_BASES, WIDTH, WIDTH)
        )
        span *= 2
    return values


def first_layer_design(w0: np.ndarray, chirps: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    eff = rotation @ w0
    weighted = chirps[:, :, None] * eff[None, :, :]
    pre = fwht_axis_one(weighted) * DESIGN_SCALE
    kerdock_rows = np.stack((pre, -pre), axis=2).reshape((-1, WIDTH))
    coord_rows = np.stack((RADIUS * eff, -RADIUS * eff), axis=1).reshape((-1, WIDTH))
    return np.maximum(np.concatenate((kerdock_rows, coord_rows), axis=0), 0.0)


def group_means(weights: list[np.ndarray], chirps, rotation) -> np.ndarray:
    """Return (129, 256) float64 per-basis final-layer means."""
    act = first_layer_design(weights[0], chirps, rotation)
    for w in weights[1:]:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
    # 129 groups of 512 consecutive rows each
    g = act.reshape(129, 512, WIDTH).sum(axis=1, dtype=np.float64) / 512.0
    return g


def load_networks(limit=None):
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    out = []
    for f in sorted(DATA.glob("mini-*.parquet")):
        t = pq.read_table(f)
        ids = t.column("mlp_id").to_pylist()
        W = np.asarray(t.column("weights").to_pylist(), dtype=np.float32)
        Y = np.asarray(t.column("final_means").to_pylist(), dtype=np.float64)
        for k, mid in enumerate(ids):
            out.append((int(mid), W[k], Y[k]))
            if limit is not None and len(out) >= limit:
                return out, chirps, rotation
    return out, chirps, rotation


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nets, chirps, rotation = load_networks(limit)
    nets.sort(key=lambda r: r[0])
    print(f"loaded {len(nets)} networks", flush=True)

    G_all, Y_all, ids = [], [], []
    t0 = time.time()
    for n, (mid, W, y) in enumerate(nets):
        weights = [np.ascontiguousarray(W[i]) for i in range(DEPTH)]
        g = group_means(weights, chirps, rotation)
        G_all.append(g)
        Y_all.append(y)
        ids.append(mid)
        if n % 10 == 0:
            print(f"  {n:3d}/{len(nets)}  {time.time()-t0:.1f}s", flush=True)

    G = np.stack(G_all)          # (M, 129, 256)
    Y = np.stack(Y_all)          # (M, 256)
    ids = np.asarray(ids)
    out = Path(__file__).parent / "groups.npz"
    np.savez_compressed(out, G=G, Y=Y, ids=ids)

    yhat = G.mean(axis=1)
    mse = ((yhat - Y) ** 2).mean(axis=1)
    print(f"\nbaseline raw MSE  mean over {len(ids)} nets : {mse.mean():.6e}")
    print(f"  first 20 networks                        : {mse[:20].mean():.6e}")
    if len(ids) >= 50:
        print(f"  IDs 0-49                                 : {mse[:50].mean():.6e}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
