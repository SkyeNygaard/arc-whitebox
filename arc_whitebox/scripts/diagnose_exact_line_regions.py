"""Count exact affine pieces of a deep ReLU MLP along Gaussian affine lines.

This is a feasibility diagnostic for the exact conditional-line identity

    E[f(X)] = d E_{v,Z} sum_r phi(t_r) (slope_r+ - slope_r-),

where v is a random unit direction, Z is Gaussian in v-perp, and t_r are all
breakpoints of t -> f(Z + t v).  The identity is attractive only if exact
piece propagation stays small.  We therefore propagate all affine pieces
layer by layer and stop at a predeclared cap.

Only official Mini selection IDs are supported by default; no challenge
holdout is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "exact_line_region_counts.json"


def child_pieces(
    lower: float,
    upper: float,
    slope: np.ndarray,
    intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero = np.abs(slope) > 1e-12
    roots = -intercept[nonzero] / slope[nonzero]
    roots = roots[(roots > lower) & (roots < upper)]
    if len(roots):
        roots = np.unique(roots)
        boundaries = np.concatenate(([lower], roots, [upper]))
    else:
        boundaries = np.asarray([lower, upper], dtype=np.float64)

    lows = boundaries[:-1]
    highs = boundaries[1:]
    tests = np.empty_like(lows)
    finite_low = np.isfinite(lows)
    finite_high = np.isfinite(highs)
    both = finite_low & finite_high
    tests[both] = 0.5 * (lows[both] + highs[both])
    left_tail = ~finite_low & finite_high
    tests[left_tail] = highs[left_tail] - np.maximum(
        1.0,
        np.abs(highs[left_tail]),
    )
    right_tail = finite_low & ~finite_high
    tests[right_tail] = lows[right_tail] + np.maximum(
        1.0,
        np.abs(lows[right_tail]),
    )
    if np.any(~finite_low & ~finite_high):
        tests[~finite_low & ~finite_high] = 0.0
    return lows, highs, tests


def count_line_pieces(
    weights: np.ndarray,
    direction: np.ndarray,
    offset: np.ndarray,
    cap: int,
) -> dict[str, object]:
    lowers = np.asarray([-np.inf], dtype=np.float64)
    uppers = np.asarray([np.inf], dtype=np.float64)
    slopes = direction[None, :].astype(np.float64)
    intercepts = offset[None, :].astype(np.float64)
    counts = []
    layer_seconds = []

    for layer, weight32 in enumerate(weights, start=1):
        start = time.perf_counter()
        weight = weight32.astype(np.float64)
        pre_slopes = slopes @ weight
        pre_intercepts = intercepts @ weight
        new_lowers = []
        new_uppers = []
        new_slopes = []
        new_intercepts = []
        overflow = False
        generated = 0
        for piece in range(len(lowers)):
            lows, highs, tests = child_pieces(
                float(lowers[piece]),
                float(uppers[piece]),
                pre_slopes[piece],
                pre_intercepts[piece],
            )
            gates = (
                tests[:, None] * pre_slopes[piece][None, :]
                + pre_intercepts[piece][None, :]
            ) > 0.0
            new_lowers.append(lows)
            new_uppers.append(highs)
            new_slopes.append(
                pre_slopes[piece][None, :] * gates
            )
            new_intercepts.append(
                pre_intercepts[piece][None, :] * gates
            )
            generated += len(lows)
            if generated > cap:
                overflow = True
                break

        counts.append(
            {
                "layer": layer,
                "input_pieces": int(len(lowers)),
                "generated_pieces_before_cap": int(generated),
                "cap_exceeded": overflow,
            }
        )
        layer_seconds.append(time.perf_counter() - start)
        if overflow:
            break
        lowers = np.concatenate(new_lowers)
        uppers = np.concatenate(new_uppers)
        slopes = np.concatenate(new_slopes)
        intercepts = np.concatenate(new_intercepts)

    result = {
        "completed_depth": len(counts) if not counts[-1]["cap_exceeded"] else len(counts) - 1,
        "cap": cap,
        "counts": counts,
        "layer_seconds": layer_seconds,
    }
    if result["completed_depth"] == len(weights):
        phi_lower = np.zeros_like(lowers)
        phi_upper = np.zeros_like(uppers)
        finite_lower = np.isfinite(lowers)
        finite_upper = np.isfinite(uppers)
        phi_lower[finite_lower] = np.exp(
            -0.5 * np.square(lowers[finite_lower])
        ) / np.sqrt(2.0 * np.pi)
        phi_upper[finite_upper] = np.exp(
            -0.5 * np.square(uppers[finite_upper])
        ) / np.sqrt(2.0 * np.pi)
        masses = ndtr(uppers) - ndtr(lowers)
        first_moments = phi_lower - phi_upper
        conditional_mean = np.sum(
            slopes * first_moments[:, None]
            + intercepts * masses[:, None],
            axis=0,
        )

        finite_breaks = lowers[1:]
        break_density = np.exp(
            -0.5 * np.square(finite_breaks)
        ) / np.sqrt(2.0 * np.pi)
        slope_jumps = slopes[1:] - slopes[:-1]
        boundary_trace = weights.shape[-1] * np.sum(
            slope_jumps * break_density[:, None],
            axis=0,
        )
        result["conditional_mean"] = conditional_mean.tolist()
        result["boundary_trace_estimate"] = boundary_trace.tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lines", type=int, default=1)
    parser.add_argument("--cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if any(index >= 50 for index in args.ids):
        raise ValueError("this diagnostic intentionally seals IDs 50--99")

    rows = _load_rows(args.data, args.ids)
    records = []
    for mlp_id, (name, weights, targets) in zip(args.ids, rows, strict=True):
        for line in range(args.lines):
            rng = np.random.default_rng(
                args.seed + 1009 * mlp_id + 65_537 * line
            )
            direction = rng.standard_normal(weights.shape[-1])
            direction /= np.linalg.norm(direction)
            gaussian = rng.standard_normal(weights.shape[-1])
            offset = gaussian - direction * np.dot(direction, gaussian)
            result = count_line_pieces(
                weights,
                direction,
                offset,
                args.cap,
            )
            record = {
                "id": mlp_id,
                "name": name,
                "line": line,
                **result,
            }
            if "conditional_mean" in result:
                target = targets[-1]
                record["conditional_mean_mse"] = float(
                    np.mean(
                        np.square(
                            np.asarray(result["conditional_mean"]) - target
                        )
                    )
                )
                record["boundary_trace_mse"] = float(
                    np.mean(
                        np.square(
                            np.asarray(result["boundary_trace_estimate"])
                            - target
                        )
                    )
                )
            records.append(record)
            last = result["counts"][-1]
            print(
                f"id={mlp_id:02d} line={line} "
                f"completed={result['completed_depth']} "
                f"last_generated={last['generated_pieces_before_cap']} "
                f"overflow={last['cap_exceeded']}",
                flush=True,
            )

    ensembles = []
    for mlp_id in args.ids:
        complete = [
            record
            for record in records
            if record["id"] == mlp_id
            and "conditional_mean" in record
        ]
        if not complete:
            continue
        target = next(
            targets[-1]
            for index, (_, _, targets) in zip(args.ids, rows, strict=True)
            if index == mlp_id
        )
        conditional = np.mean(
            [record["conditional_mean"] for record in complete],
            axis=0,
        )
        boundary = np.mean(
            [record["boundary_trace_estimate"] for record in complete],
            axis=0,
        )
        sum_input_pieces = sum(
            sum(item["input_pieces"] for item in record["counts"])
            for record in complete
        )
        dense_flops = 4 * weights.shape[-1] ** 2 * sum_input_pieces
        ensembles.append(
            {
                "id": mlp_id,
                "lines": len(complete),
                "conditional_mean_mse": float(
                    np.mean(np.square(conditional - target))
                ),
                "boundary_trace_mse": float(
                    np.mean(np.square(boundary - target))
                ),
                "dense_affine_coefficient_flops_fma2": dense_flops,
                "budget_fraction": dense_flops / 272_000_000_000,
            }
        )
    artifact = {
        "protocol": {
            "ids": args.ids,
            "lines_per_mlp": args.lines,
            "piece_cap": args.cap,
            "seed": args.seed,
            "holdout_loaded": False,
        },
        "ensembles": ensembles,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
