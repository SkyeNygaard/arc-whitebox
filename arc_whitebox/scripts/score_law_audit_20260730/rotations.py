"""Is the residual error ISOTROPIC, or does the design's orientation matter?

The DGS bound closes full t-designs.  It does NOT close a rule adapted to the
few directions where THIS network has high-degree content.  The cheapest
decisive probe of that hypothesis: rotating the design is exactly a rotation of
its aliasing tensors A_l, so

    e(R) = sum_l <R A_l, g_l>

If the network's high-degree content were low-dimensional / structured, e(R)
would vary strongly with R and some orientations would be far better than
others.  If the content is generic, e(R) is isotropic and every rotation is
equally good -- which closes the adapted-rule direction empirically.

Any orthogonal R applied to the input space keeps the point set a valid
5-design, so all arms are legal designs of identical cost.

Also measured: whether the LEGAL magnitude signal (between-basis variance, which
correlates 0.927 with e^2) can pick the good rotation.  Selection needs only the
error's MAGNITUDE, never its sign -- so unlike every previous probe it is not
blocked by the sign wall.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import (WIDTH, DEPTH, ASSET, DATA, fwht_axis_one, RADIUS,
                     DESIGN_SCALE, KERDOCK_BASES)

N_ROT = 12
N_NETS = 8
ROWS_PER_BASIS = 512


def design_rows(w0, chirps, rot):
    eff = rot @ w0
    weighted = chirps[:, :, None] * eff[None, :, :]
    pre = fwht_axis_one(weighted) * DESIGN_SCALE
    ker = np.stack((pre, -pre), axis=2).reshape((-1, WIDTH))
    coord = np.stack((RADIUS * eff, -RADIUS * eff), axis=1).reshape((-1, WIDTH))
    return np.maximum(np.concatenate((ker, coord), axis=0), 0.0)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rot0 = asset["rotation"].astype(np.float32)
    rng = np.random.default_rng(2024)
    rots = [rot0] + [np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))[0].astype(np.float32)
                     for _ in range(N_ROT - 1)]

    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    MSE = np.zeros((N_NETS, N_ROT))
    PRED = np.zeros((N_NETS, N_ROT))
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        for j, R in enumerate(rots):
            act = design_rows(weights[0], chirps, R)
            for w in weights[1:]:
                act = np.maximum(act @ w, 0.0, dtype=np.float32)
            grp = act.reshape(129, ROWS_PER_BASIS, WIDTH).mean(1, dtype=np.float64)
            yhat = grp.mean(0)
            MSE[net, j] = np.mean((yhat - Y_all[net]) ** 2)
            PRED[net, j] = np.mean(grp.var(0)) / 129.0     # legal magnitude proxy
        print(f"  net {net}: MSE spread {MSE[net].std()/MSE[net].mean():.1%}", flush=True)

    print(f"\nDESIGN ORIENTATION: {N_NETS} networks x {N_ROT} rotations\n")
    rel = MSE / MSE.mean(1, keepdims=True)
    print(f"  mean MSE over all arms            : {MSE.mean():.4e}")
    print(f"  relative spread across rotations  : {rel.std():.1%}")
    print(f"  mean(best rotation)/mean(all)     : {(MSE.min(1)/MSE.mean(1)).mean():.4f}")
    print(f"  mean(worst)/mean(all)             : {(MSE.max(1)/MSE.mean(1)).mean():.4f}")
    print(f"  ORACLE gain from picking the best : {MSE.mean(1).mean()/MSE.min(1).mean():.3f}x")

    # can the legal magnitude proxy pick the good rotation?
    hit, sp = [], []
    for net in range(N_NETS):
        pick = int(np.argmin(PRED[net]))
        hit.append(MSE[net, pick] / MSE[net].mean())
        r1 = np.argsort(np.argsort(PRED[net])); r2 = np.argsort(np.argsort(MSE[net]))
        sp.append(np.corrcoef(r1, r2)[0, 1])
    print(f"\n  legal proxy selection: MSE ratio  : {np.mean(hit):.4f} "
          f"(1.0 = no better than random)")
    print(f"  rank correlation proxy vs truth   : {np.mean(sp):+.3f}")
    print(f"\n  A rotation costs a full design, so selecting among k arms costs kx.")
    print(f"  Even a perfect oracle needs a gain > k to pay for itself.")


if __name__ == "__main__":
    main()
