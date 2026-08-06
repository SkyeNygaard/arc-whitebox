"""The last open door: WEIGHT-AWARE prediction of the signed error.

The ledger's no-go theorem (T81) covers the finite group-output transcript only;
weight-aware and state-aware estimators are explicitly left open.  Every probe so
far used transcript features.  This one uses the weights and the full propagated
state at the final layer.

Per (network, neuron) features -- all legally computable at predict time:
  mu, sigma, skew, kurtosis of the final pre-activation
  t = mu/sigma, active fraction, the estimate yhat itself
  between-basis variance (the magnitude signal, corr 0.927 with e^2)
  ||w_i||, alignment of w_i with the top activation eigenvector
  the GAUSSIAN-CLOSURE RESIDUAL: closure prediction of E[relu] minus yhat,
    a direct local measure of non-Gaussianity at that neuron

Target: the SIGNED error e_i = yhat_i - y_i.
Validation: leave-one-network-out, so nothing is fitted on the test network.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design
from closures import relu_mean_gauss

ROWS_PER_BASIS = 512


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)

    F, T, GRP = [], [], []
    nets = 0
    for f in sorted(DATA.glob("mini-*.parquet")):
        t = pq.read_table(f)
        W = np.asarray(t.column("weights").to_pylist(), dtype=np.float32)
        Y = np.asarray(t.column("final_means").to_pylist(), dtype=np.float64)
        for k in range(W.shape[0]):
            weights = [np.ascontiguousarray(W[k][i]) for i in range(DEPTH)]
            act = first_layer_design(weights[0], chirps, rotation)
            for w in weights[1:-1]:
                act = np.maximum(act @ w, 0.0, dtype=np.float32)
            a30 = act
            p = (a30 @ weights[-1]).astype(np.float64)          # final pre-activation
            R = np.maximum(p, 0.0)
            yhat = R.mean(0)
            e = yhat - Y[k]

            mu = p.mean(0); c = p - mu
            sd = np.sqrt((c ** 2).mean(0)) + 1e-30
            sk = (c ** 3).mean(0) / sd ** 3
            ku = (c ** 4).mean(0) / sd ** 4 - 3.0
            act_frac = (p > 0).mean(0)
            grp = R.reshape(129, ROWS_PER_BASIS, WIDTH).mean(1)
            s2 = grp.var(0)
            wl = weights[-1].astype(np.float64)
            wnorm = np.linalg.norm(wl, axis=0)
            Ca = np.cov(a30.astype(np.float64), rowvar=False, bias=True)
            ev, V = np.linalg.eigh(Ca)
            v1 = V[:, -1]
            align = np.abs(wl.T @ v1)
            clos = relu_mean_gauss(mu, sd) - yhat     # Gaussian-closure residual

            F.append(np.stack([mu, sd, sk, ku, mu / sd, act_frac, yhat,
                               np.sqrt(s2), np.sqrt(s2) / (yhat + 1e-12),
                               wnorm, align, clos, clos / (sd + 1e-30),
                               np.log(s2 + 1e-300)], 1))
            T.append(e)
            GRP.append(np.full(WIDTH, nets))
            nets += 1
        print(f"  {nets} networks", flush=True)

    X = np.concatenate(F); y = np.concatenate(T); g = np.concatenate(GRP)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, g = X[ok], y[ok], g[ok]
    X = (X - X.mean(0)) / (X.std(0) + 1e-30)
    X = np.hstack([X, X ** 2, np.ones((len(X), 1))])          # allow curvature

    print(f"\nWEIGHT-AWARE signed-error prediction: {nets} networks, "
          f"{len(y):,} (network, neuron) observations, {X.shape[1]} features\n")
    base = np.mean(y ** 2)
    for lam in [1e-3, 1e-1, 1.0, 10.0, 100.0]:
        pred = np.zeros_like(y)
        for m in np.unique(g):
            tr = g != m
            A = X[tr].T @ X[tr] + lam * len(y[tr]) * np.eye(X.shape[1])
            pred[g == m] = X[g == m] @ np.linalg.solve(A, X[tr].T @ y[tr])
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)
        sign = np.mean(np.sign(pred) == np.sign(y))
        print(f"  ridge lam={lam:<7} LORO R^2 = {r2:+.4f}   sign = {sign:.4f}   "
              f"MSE ratio = {np.mean((y-pred)**2)/base:.4f}")
    print("\n  R^2 <= 0 and sign ~ 0.5 would close the weight-aware branch too.")


if __name__ == "__main__":
    main()
