"""Official-data test of the `two1` / `two2` first-layer moment transport (V62).

V62 reports, on 32 *synthetic* networks: two1 -6.24%, two2 -9.58% raw MSE, but
only 20/32 wins and a gain-ratio CI of 1.007-1.209 for two2.  The catalog's own
gate demands ">=5% adjusted improvement, CI excluding no gain after
reference-noise correction, and no severe tail", evaluated officially.  This
runs that gate on official Mini IDs 0-49.

The transport is exact-in-principle: after the Kerdock first layer, both moments
of a1 = ReLU(W1^T x) are known in closed form from W1 alone (Cho-Saul), so the
activation cloud can be mapped affinely onto them.  `two1` applies the exact map,
`two2` doubles the correction.

The relevant prior on this family is not encouraging.  The layer-2 analogue,
tested earlier today on the same official set, reported -9.0% on 10 networks and
decayed monotonically to -1.2% on 50, with a bootstrap CI straddling 1.0 and a
23/50 win rate.  Sample size is the whole story there, so this reports the
50-network number, a paired bootstrap, the win rate and the worst network.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

from exact_moments import exact_layer1_moments  # noqa: E402


def symmetric_power(matrix, power, floor=1e-12):
    vals, vecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    top = float(vals.max())
    vals = np.maximum(vals, floor * top if top > 0 else floor)
    return (vecs * np.power(vals, power)) @ vecs.T


def transport(activation, target_mean, target_cov, strength):
    """Affine map onto the exact (mean, second moment); strength 1 = exact."""
    sample_mean = activation.mean(axis=0)
    centered = activation - sample_mean
    sample_cov = (centered.T @ centered) / activation.shape[0]
    inv_root = symmetric_power(sample_cov, -0.5)
    root = symmetric_power(sample_cov, 0.5)
    g = inv_root @ target_cov @ inv_root
    a = inv_root @ symmetric_power(g, 0.5 * strength) @ root
    shift = sample_mean + strength * (target_mean - sample_mean)
    return centered @ a + shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=list(range(50)))
    ap.add_argument("--rotation-seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "two_moment_transport.json")
    args = ap.parse_args()
    if max(args.indices) >= 50:
        raise ValueError("selection protocol: official IDs 0--49 only")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    rows = _load_rows(DEFAULT_DATA, args.indices)

    # Clipping the transported cloud back to >=0 partially undoes the transport
    # (the corrected moments are no longer the ones that were matched), so both
    # conventions are reported rather than assuming one.
    variants = {
        "baseline": (None, False),
        "two1": (1.0, False), "two2": (2.0, False),
        "two1_clip": (1.0, True), "two2_clip": (2.0, True),
    }
    mses = {k: [] for k in variants}

    for idx, (name, W, tg) in zip(args.indices, rows, strict=True):
        pre = points @ (rotation @ W[0].astype(np.float64))
        base_act = np.maximum(pre, 0.0)
        tm, tc = exact_layer1_moments(W[0])
        for label, (strength, clip) in variants.items():
            act = base_act if strength is None else transport(base_act, tm, tc, strength)
            a = (np.maximum(act, 0.0) if (clip or strength is None) else act).astype(np.float32)
            for weight in W[1:]:
                a = np.maximum(a @ weight, 0.0)
            mses[label].append(
                float(np.mean(np.square(a.mean(axis=0, dtype=np.float64) - tg[-1])))
            )
        print(f"[{idx:>3}] {name[:18]:<18} " + " ".join(
            f"{k} {mses[k][-1]:.3e}" for k in variants), flush=True)

    base = np.array(mses["baseline"])
    n = len(base)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, size=(20000, n))
    print(f"\n{n} official networks (IDs {args.indices[0]}-{args.indices[-1]})")
    print(f"{'variant':<10}{'mean MSE':>12}{'ratio':>9}{'bootstrap 95%':>22}"
          f"{'wins':>9}{'worst':>9}")
    out = {}
    for label in [k for k in variants if k != "baseline"]:
        x = np.array(mses[label])
        ratio = x.mean() / base.mean()
        b = x[boot].mean(1) / base[boot].mean(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        wins = int((x < base).sum())
        worst = float((x / base).max())
        out[label] = {"ratio": ratio, "ci": [lo, hi], "wins": wins,
                      "worst": worst, "mean_mse": float(x.mean())}
        print(f"{label:<10}{x.mean():12.4e}{ratio:9.4f}   [{lo:6.4f}, {hi:6.4f}]"
              f"{wins:8d}/{n}{worst:9.2f}x")
    print(f"{'baseline':<10}{base.mean():12.4e}{1.0:9.4f}")

    print("\nCatalog gate: >=5% adjusted improvement, CI excluding no gain, no severe tail")
    for label, r in out.items():
        gate = (r["ratio"] <= 0.95) and (r["ci"][1] < 1.0) and (r["worst"] < 1.5)
        print(f"  {label}: {'PASS' if gate else 'FAIL'}"
              f"  (ratio {r['ratio']:.4f}, CI hi {r['ci'][1]:.4f}, worst {r['worst']:.2f}x)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
