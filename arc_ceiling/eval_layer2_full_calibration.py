"""Full-covariance calibration of the layer-2 preactivation cloud.

The Kerdock rule is a spherical 5-design, so it integrates every polynomial of
degree <= 5 exactly and its residual error is carried entirely by degree >= 6.
That is why every *linear* control variate has failed on top of it (a degree-1
object has nothing left to cancel), and it is also why the exact layer-2
moments are the right lever: `E[a1 a1^T]` inherits the (1-t)^{3/2} branch point
of the arc-cosine kernel, so its harmonic content reaches past degree 5 and the
design genuinely misses it.

The existing work calibrates the layer-2 cloud *marginally* -- it matches
column means and column variances, 2n constraints.  The ledger's own evidence
(M18: correcting the covariance diagonal moved sigma error only 3.70% -> 3.64%,
"error mostly off-diagonal") says that is the weaker half of the available
signal.  This module matches the FULL covariance, n(n+1)/2 constraints, via the
affine map

    h' = (h - m_hat) A(s) + m_hat + s (m_star - m_hat),
    A(s) = C_hat^{-1/2} G^{s/2} C_hat^{1/2},   G = C_hat^{-1/2} C_star C_hat^{-1/2},

which is the geodesic interpolation on the PSD cone: A(0) = I and A(1) maps the
empirical covariance exactly onto C_star.  s > 1 over-relaxes, which the
existing marginal sweep suggested was still improving when it was cut off at
1.25.

Cost.  The transform itself is free -- it folds into the previous weight matrix,
since h = a1 W2 gives h' = a1 (W2 A) + const.  The only real charge is forming
the empirical Gram, N n^2 ~ 8.7e9 FLOPs = 3.2% of the 2.72e11 budget.  The
marginal variant needs only column sums of squares (N n, negligible), so full
covariance has to beat marginal by more than ~3.2% to be worth it on score.
Both raw MSE and that score adjustment are reported.

Selection protocol follows the existing convention: official IDs 0--49 only,
the frozen holdout 50--99 is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

from exact_moments import exact_layer2_moments  # noqa: E402

BUDGET = 2.72e11
GRAM_FLOPS = 66048 * WIDTH * WIDTH * 2.0   # empirical Gram of the cloud
BASE_FLOPS = 268.4e9                       # dense Kerdock propagation (A01)


def symmetric_power(matrix: np.ndarray, power: float, floor: float = 1e-12):
    """M^power for symmetric PSD M, with relative eigenvalue flooring."""
    vals, vecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    top = float(vals.max())
    vals = np.maximum(vals, floor * top if top > 0 else floor)
    return (vecs * np.power(vals, power)) @ vecs.T


def full_covariance_map(sample_cov, target_cov, strength: float):
    """A(s) = C_hat^{-1/2} G^{s/2} C_hat^{1/2} with G = C_hat^{-1/2} C* C_hat^{-1/2}."""
    inv_root = symmetric_power(sample_cov, -0.5)
    root = symmetric_power(sample_cov, 0.5)
    g = inv_root @ target_cov @ inv_root
    return inv_root @ symmetric_power(g, 0.5 * strength) @ root


def iter_variants(pre, target_mean, target_cov, strengths):
    """Yield (name, activation) one at a time; each cloud is ~135 MB."""
    pre = pre.astype(np.float64)
    sample_mean = pre.mean(axis=0)
    centered = pre - sample_mean
    n = pre.shape[0]
    sample_cov = (centered.T @ centered) / n
    delta = target_mean - sample_mean

    target_var = np.maximum(np.diag(target_cov), 0.0)
    sample_var = np.maximum(np.diag(sample_cov), 1e-30)
    std_ratio = np.sqrt(target_var / sample_var)

    scale = np.sqrt(np.outer(target_var, target_var))
    off = ~np.eye(len(target_var), dtype=bool)
    meta = {
        "mean_rel_rms": float(
            np.sqrt(np.mean((delta / np.maximum(np.abs(target_mean), 1e-30)) ** 2))
        ),
        "var_rel_rms": float(
            np.sqrt(np.mean(((sample_var - target_var) / target_var) ** 2))
        ),
        "offdiag_cov_rel_rms": float(
            np.sqrt(np.mean(((sample_cov - target_cov) / scale)[off] ** 2))
        ),
    }

    def gen():
        yield "baseline", np.maximum(pre, 0.0)
        for s in strengths:
            # marginal: matches the existing implementation exactly
            yield (f"marginal_{s:g}",
                   np.maximum(centered * np.power(std_ratio, s)
                              + (sample_mean + s * delta), 0.0))
            a = full_covariance_map(sample_cov, target_cov, s)
            yield (f"full_{s:g}",
                   np.maximum(centered @ a + (sample_mean + s * delta), 0.0))

    return gen, meta


def propagate(activation, weights):
    for weight in weights[2:]:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.mean(axis=0, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--rotation-seed", type=int, default=3)
    ap.add_argument(
        "--strengths", type=float, nargs="+",
        default=[0.5, 1.0, 1.25, 1.5, 2.0, 2.5],
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "layer2_full_calibration.json")
    args = ap.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("selection protocol: official IDs 0--49 only")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    rows = _load_rows(args.data, args.indices)

    records = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        t0 = time.perf_counter()
        first = np.maximum(points @ (rotation @ weights[0]), 0.0)
        pre = first @ weights[1]
        target_mean, target_cov = exact_layer2_moments(weights)
        gen, meta = iter_variants(pre, target_mean, target_cov, args.strengths)
        mse = {}
        for k, v in gen():
            pred = propagate(v.astype(np.float32), weights)
            mse[k] = float(np.mean(np.square(pred - targets[-1])))
            del v
        records.append({
            "index": index, "name": name,
            "seconds": time.perf_counter() - t0,
            "metadata": meta, "final_mse": mse,
        })
        print(f"[{index:>3}] {name:<22} base {mse['baseline']:.4e}  "
              f"marg1 {mse.get('marginal_1', float('nan')):.4e}  "
              f"full1 {mse.get('full_1', float('nan')):.4e}  "
              f"offdiag_err {meta['offdiag_cov_rel_rms']:.4f}", flush=True)

    names = sorted(records[0]["final_mse"])
    base_mean = float(np.mean([r["final_mse"]["baseline"] for r in records]))
    base_med = float(np.median([r["final_mse"]["baseline"] for r in records]))
    summary = {}
    for v in names:
        vals = np.array([r["final_mse"][v] for r in records])
        # score = MSE * cost/B; marginal calibration is free, full costs one Gram
        extra = 0.0 if (v == "baseline" or v.startswith("marginal")) else GRAM_FLOPS
        cost_ratio = (BASE_FLOPS + extra) / BASE_FLOPS
        summary[v] = {
            "mean_final_mse": float(vals.mean()),
            "median_final_mse": float(np.median(vals)),
            "mean_vs_base": float(vals.mean() / base_mean),
            "median_vs_base": float(np.median(vals) / base_med),
            "cost_ratio": cost_ratio,
            "score_ratio_mean": float(vals.mean() / base_mean * cost_ratio),
            "score_ratio_median": float(np.median(vals) / base_med * cost_ratio),
            "wins": int((vals < np.array([r["final_mse"]["baseline"] for r in records])).sum()),
        }

    print(f"\n{'variant':<16}{'mean MSE':>12}{'med MSE':>12}"
          f"{'mean/base':>11}{'med/base':>10}{'score(med)':>12}{'wins':>7}")
    for v in ["baseline"] + [x for x in names if x != "baseline"]:
        s = summary[v]
        print(f"{v:<16}{s['mean_final_mse']:12.4e}{s['median_final_mse']:12.4e}"
              f"{s['mean_vs_base']:11.4f}{s['median_vs_base']:10.4f}"
              f"{s['score_ratio_median']:12.4f}{s['wins']:7d}/{len(records)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "protocol": {
            "selection_indices": args.indices,
            "holdout_loaded": False,
            "rotation_seed": args.rotation_seed,
            "gram_flops": GRAM_FLOPS,
            "base_flops": BASE_FLOPS,
        },
        "summary": summary,
        "records": records,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
