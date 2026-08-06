"""How much of the ledger's next-variance metric is reference noise?

The central white-box claims are quoted as "next-layer variance relative RMS":

    M40  Gaussian 2.564% -> third order 1.009% -> third+fourth 0.288%
    M66  3.25% -> 0.856%  (3.80x)
    M64  3.58x next-variance gain

All are synthetic.  The metric compares a predicted Var(h_{l+1,i}) against a
Monte-Carlo reference, and a variance estimated from N samples carries relative
noise of roughly sqrt((kappa4 + 2)/N) -- 0.276% at N = 262,144 under Gaussian
kurtosis, which is the same size as M40's quoted endpoint.

Measured MSE is (true error)^2 + (reference noise)^2, so reference noise makes a
good method look WORSE, not better -- it cannot manufacture a gain.  But it
saturates the metric: once the quoted number approaches the floor, the endpoint
is no longer resolvable and the reported ratio understates or overstates the
truth unpredictably, and the comparison between two good methods becomes noise.

This measures the floor directly, on real networks, by comparing two independent
references of the same size (difference / sqrt(2) is the per-reference noise),
rather than assuming Gaussian kurtosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

WIDTH = 256
CHUNK = 32768


def next_layer_variance(weights, layer, n_samples, seed):
    """Var(h_{l+1,i}) from `n_samples` fresh draws, streamed."""
    s1 = np.zeros(WIDTH)
    s2 = np.zeros(WIDTH)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            h = a @ w
            a = np.maximum(h, 0.0)
            if li == layer:
                nxt = a.astype(np.float64) @ weights[li + 1].astype(np.float64)
                s1 += nxt.sum(0)
                s2 += (nxt * nxt).sum(0)
                break
        done += b
    mean = s1 / n_samples
    return s2 / n_samples - mean * mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[262144, 1_000_000, 4_000_000, 16_000_000])
    ap.add_argument("--out", type=Path, default=HERE / "results" / "metric_noise.json")
    args = ap.parse_args()

    rows = _load_rows(DEFAULT_DATA, args.indices)
    claims = {
        "M40 Gaussian baseline": 2.564,
        "M66 baseline": 3.25,
        "M40 third order": 1.009,
        "M66 corrected": 0.856,
        "M40 third+fourth": 0.288,
    }

    print(f"Reference-noise floor on 'next-variance relative RMS', layer {args.layer + 1}")
    print("measured from two independent references of equal size\n")
    print(f"{'reference N':>14}{'noise floor':>14}{'sqrt(2/N)':>12}   resolvable claims")
    floors = {}
    for n in args.sizes:
        per = []
        for ri, (name, W, _) in enumerate(rows):
            a = next_layer_variance(W, args.layer, n, seed=1000 + ri)
            b = next_layer_variance(W, args.layer, n, seed=9000 + ri)
            mid = 0.5 * (a + b)
            good = mid > 0
            rel = np.sqrt(np.mean(((a - b) / mid)[good] ** 2)) / np.sqrt(2.0)
            per.append(rel)
        floor = float(np.mean(per))
        floors[n] = floor
        ok = [k for k, v in claims.items() if v / 100.0 > 3 * floor]
        print(f"{n:>14,}{floor:13.4%}{np.sqrt(2/n):12.4%}   "
              f"{len(ok)}/{len(claims)} above 3x floor", flush=True)

    print(f"\n{'claim':<26}{'value':>9}   " +
          "".join(f"{n//1000}k".rjust(9) for n in args.sizes))
    for k, v in sorted(claims.items(), key=lambda kv: -kv[1]):
        cells = ""
        for n in args.sizes:
            f = floors[n]
            cells += ("  RESOLVED" if v / 100.0 > 3 * f
                      else ("  marginal" if v / 100.0 > f else "  BURIED ")).rjust(9)
        print(f"{k:<26}{v:8.3f}%{cells}")

    f0 = floors[args.sizes[0]]
    print(f"""
Reading at N = {args.sizes[0]:,} (the size used by the synthetic harnesses):
  floor = {f0:.4%}.  A claim is only resolvable if it sits well ABOVE the floor.
  Claims at or below it are not measurable there -- and note this cuts against
  the METHOD, not for it: reference noise inflates measured error, so a buried
  endpoint means the true error is smaller than quoted but the ratio between
  two competing good methods is unmeasurable.""")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"floors": floors, "claims": claims}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
