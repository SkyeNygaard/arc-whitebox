"""Benchmark high-throughput sampling estimators on the official Phase-1 mini set.

This is deliberately a local research harness, not a submission.  It answers
two questions that a leaderboard score cannot:

1. How much error reduction comes from the input design (IID, scrambled Sobol,
   or a randomly shifted rank-1 lattice)?
2. Does the cheap recursive mean control variate still help after QMC and exact
   radial integration?

The official mini targets use 1e9 Monte-Carlo samples, so their label noise is
small enough to compare estimators in the 1e-8--1e-7 MSE range.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import gammaln, ndtr, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "official_phase1_mini" / "data"
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _first_primes(n: int) -> np.ndarray:
    """Return the first ``n`` primes without adding another dependency."""
    limit = max(16, int(n * (math.log(max(n, 2)) + math.log(math.log(max(n, 3))))) + 16)
    while True:
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[:2] = False
        for p in range(2, int(math.sqrt(limit)) + 1):
            if sieve[p]:
                sieve[p * p :: p] = False
        primes = np.flatnonzero(sieve)
        if len(primes) >= n:
            return primes[:n]
        limit *= 2


@dataclass
class Design:
    kind: str
    n: int
    total: int
    seed: int
    antithetic: bool
    sphere: bool

    def __post_init__(self) -> None:
        self.base_total = (self.total + 1) // 2 if self.antithetic else self.total
        self.rng = np.random.default_rng(self.seed)
        self.done = 0
        self.er = float(
            math.sqrt(2.0)
            * math.exp(gammaln((self.n + 1) / 2.0) - gammaln(self.n / 2.0))
        )
        if self.kind == "sobol":
            self.engine = qmc.Sobol(d=self.n, scramble=True, seed=self.seed)
        elif self.kind == "lhs":
            self.lhs_points = qmc.LatinHypercube(d=self.n, seed=self.seed).random(
                self.base_total
            )
        elif self.kind == "lattice":
            self.generator = np.mod(np.sqrt(_first_primes(self.n)), 1.0)
            self.shift = self.rng.random(self.n)
        elif self.kind not in ("iid", "orthogonal"):
            raise ValueError(f"unknown design kind: {self.kind}")

    def next(self, output_rows: int) -> np.ndarray:
        """Return the next float32 input block, including antipodes if requested."""
        if self.antithetic:
            base_rows = min((output_rows + 1) // 2, self.base_total - self.done)
        else:
            base_rows = min(output_rows, self.base_total - self.done)
        if base_rows <= 0:
            return np.empty((0, self.n), dtype=np.float32)

        if self.kind == "orthogonal":
            blocks = []
            needed = base_rows
            while needed:
                gaussian = self.rng.standard_normal((self.n, self.n))
                q, r = np.linalg.qr(gaussian)
                # NumPy fixes QR signs deterministically. Undo that convention to
                # obtain a Haar-distributed orthogonal matrix.
                q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
                take = min(needed, self.n)
                blocks.append(q[:take])
                needed -= take
            z = np.concatenate(blocks, axis=0).astype(np.float32)
            # Rows are already directions on the unit sphere.
            z *= np.float32(self.er)
        elif self.kind == "iid":
            z = self.rng.standard_normal((base_rows, self.n), dtype=np.float32)
        else:
            if self.kind == "sobol":
                u = self.engine.random(base_rows)
            elif self.kind == "lhs":
                u = self.lhs_points[self.done : self.done + base_rows]
            else:
                k = np.arange(self.done, self.done + base_rows, dtype=np.float64)[:, None]
                u = np.mod(k * self.generator[None, :] + self.shift[None, :], 1.0)
            # A float32 Gaussian cannot usefully represent farther into the tails.
            z = ndtri(np.clip(u, 1e-7, 1.0 - 1e-7)).astype(np.float32)

        self.done += base_rows
        if self.sphere and self.kind != "orthogonal":
            scale = self.er / np.linalg.norm(z, axis=1, keepdims=True)
            z = z * scale.astype(np.float32)
        if self.antithetic:
            z = np.concatenate((z, -z), axis=0)
            remaining = self.total - 2 * (self.done - base_rows)
            z = z[: min(len(z), remaining)]
        return z


def _load_rows(data_dir: Path, indices: list[int]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards below {data_dir}")
    wanted = set(indices)
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    offset = 0
    for file in files:
        table = pq.read_table(file, columns=["mlp_name", "weights", "all_layer_means"])
        for local in range(len(table)):
            global_index = offset + local
            if global_index not in wanted:
                continue
            name = table["mlp_name"][local].as_py()
            weights = np.asarray(table["weights"][local].as_py(), dtype=np.float32)
            means = np.asarray(table["all_layer_means"][local].as_py(), dtype=np.float64)
            rows.append((name, weights, means))
        offset += len(table)
    if len(rows) != len(wanted):
        raise IndexError(f"found {len(rows)} of {len(wanted)} requested rows")
    # Parquet shards are traversed in dataset order, which is the useful stable order.
    return rows


def mean_field_sensitivity_rotation(weights: np.ndarray) -> np.ndarray:
    """Put the strongest mean-field input sensitivities in early QMC axes.

    A Gaussian input is rotationally invariant, but a finite Sobol design is
    not: its earliest coordinates generally have the best projections.  The
    expected-gate Jacobian

        J = prod_l W_l diag(P[h_l > 0])

    supplies a cheap, entirely white-box ordering of input directions.  The
    gate probabilities come from a diagonal moment chain.  Rescaling ``J``
    between layers changes neither its singular vectors nor this rotation.
    """
    _, width, _ = weights.shape
    mean = np.zeros(width, dtype=np.float64)
    var = np.ones(width, dtype=np.float64)
    jacobian = np.eye(width, dtype=np.float64)
    for weight32 in weights:
        weight = weight32.astype(np.float64)
        pre_mean = mean @ weight
        pre_var = var @ np.square(weight)
        pre_sd = np.sqrt(np.maximum(pre_var, 1e-20))
        t = pre_mean / pre_sd
        gate = ndtr(t)
        phi = np.exp(-0.5 * np.square(t)) / SQRT_2PI
        second = (
            (np.square(pre_mean) + pre_var) * gate
            + pre_mean * pre_sd * phi
        )
        mean = pre_mean * gate + pre_sd * phi
        var = np.maximum(second - np.square(mean), 1e-20)

        jacobian = jacobian @ (weight * gate[None, :])
        scale = np.linalg.norm(jacobian)
        if scale > 0.0:
            jacobian /= scale

    # x = z @ U.T makes z[:, j] the coefficient of input direction U[:, j].
    u, _, _ = np.linalg.svd(jacobian, full_matrices=True)
    return u.astype(np.float32)


def estimate(
    weights: np.ndarray,
    samples: int,
    seed: int,
    design_kind: str,
    sphere: bool,
    antithetic: bool,
    anchor: str,
    terminal: str,
    chunk: int,
    input_blocks: list[np.ndarray] | None = None,
    input_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    depth, width, _ = weights.shape
    design = (
        None
        if input_blocks is not None
        else Design(design_kind, width, samples, seed, antithetic, sphere)
    )
    sum_a = np.zeros((depth, width), dtype=np.float64)
    if terminal != "direct":
        sum_final_raw = np.zeros((4, width), dtype=np.float64)
    if anchor != "none":
        sum_h = np.zeros_like(sum_a)
        sum_h2 = np.zeros_like(sum_a)
        sum_a2 = np.zeros_like(sum_a)
        sum_gate = np.zeros_like(sum_a)

    start = time.perf_counter()
    used = 0
    block_index = 0
    while used < samples:
        if input_blocks is None:
            assert design is not None
            x = design.next(min(chunk, samples - used))
        else:
            x = input_blocks[block_index]
            block_index += 1
        if not len(x):
            break
        if input_rotation is not None:
            x = x @ input_rotation.T
        a = x
        for layer, weight in enumerate(weights):
            h = a @ weight
            a = np.maximum(h, 0.0)
            sum_a[layer] += a.sum(axis=0, dtype=np.float64)
            if layer == depth - 1 and terminal != "direct":
                hd_final = h.astype(np.float64)
                h2_final = np.square(hd_final)
                sum_final_raw[0] += hd_final.sum(axis=0)
                sum_final_raw[1] += h2_final.sum(axis=0)
                sum_final_raw[2] += (h2_final * hd_final).sum(axis=0)
                sum_final_raw[3] += np.square(h2_final).sum(axis=0)
            if anchor != "none":
                hd = h.astype(np.float64)
                ad = a.astype(np.float64)
                sum_h[layer] += hd.sum(axis=0)
                sum_h2[layer] += np.square(hd).sum(axis=0)
                sum_a2[layer] += np.square(ad).sum(axis=0)
                sum_gate[layer] += (h > 0.0).sum(axis=0)
        used += len(x)
    elapsed = time.perf_counter() - start
    if used != samples:
        raise RuntimeError(f"design produced {used} rows, expected {samples}")

    mean_a = sum_a / samples
    if terminal != "direct":
        raw = sum_final_raw / samples
        mu = raw[0]
        var = np.maximum(raw[1] - np.square(mu), 1e-20)
        sigma = np.sqrt(var)
        t = mu / sigma
        phi = np.exp(-0.5 * np.square(t)) / SQRT_2PI
        terminal_mean = mu * ndtr(t) + sigma * phi
        if terminal in ("edgeworth3", "edgeworth4"):
            k3 = raw[2] - 3.0 * mu * raw[1] + 2.0 * np.power(mu, 3)
            terminal_mean -= t * phi * k3 / (6.0 * var)
        if terminal == "edgeworth4":
            centered4 = (
                raw[3]
                - 4.0 * mu * raw[2]
                + 6.0 * np.square(mu) * raw[1]
                - 3.0 * np.power(mu, 4)
            )
            k4 = centered4 - 3.0 * np.square(var)
            terminal_mean += (
                (np.square(t) - 1.0)
                * phi
                * k4
                / (24.0 * sigma * var)
            )
        mean_a[-1] = terminal_mean
    if anchor == "none":
        return mean_a, elapsed

    mean_h = sum_h / samples
    y = np.empty_like(mean_a)
    y[0] = np.linalg.norm(weights[0], axis=0) / SQRT_2PI
    for layer in range(1, depth):
        if anchor == "gate":
            beta = sum_gate[layer] / samples
        elif anchor == "ols":
            eh2 = sum_h2[layer] / samples
            ea2 = sum_a2[layer] / samples  # ReLU(h) * h == ReLU(h)^2
            cov_ah = ea2 - mean_a[layer] * mean_h[layer]
            var_h = eh2 - np.square(mean_h[layer])
            beta = cov_ah / np.maximum(var_h, 1e-20)
        else:
            raise ValueError(anchor)
        trueish_h = y[layer - 1] @ weights[layer]
        y[layer] = mean_a[layer] + beta * (trueish_h - mean_h[layer])
    return y, elapsed


def precompute_design(
    kind: str,
    width: int,
    samples: int,
    seed: int,
    antithetic: bool,
    sphere: bool,
    chunk: int,
    moment_match: str = "none",
    frame_iterations: int = 6,
) -> list[np.ndarray]:
    design = Design(kind, width, samples, seed, antithetic, sphere)
    blocks = []
    used = 0
    while used < samples:
        block = design.next(min(chunk, samples - used))
        blocks.append(block)
        used += len(block)
    if moment_match != "none":
        sizes = [len(block) for block in blocks]
        x = np.concatenate(blocks, axis=0).astype(np.float64)
        iterations = 1 if moment_match == "whiten" else frame_iterations
        for _ in range(iterations):
            if moment_match == "sphere_frame":
                er = float(
                    math.sqrt(2.0)
                    * math.exp(gammaln((width + 1) / 2.0) - gammaln(width / 2.0))
                )
                x *= er / np.linalg.norm(x, axis=1, keepdims=True)
            covariance = (x.T @ x) / len(x)
            chol = np.linalg.cholesky(covariance)
            x = np.linalg.solve(chol, x.T).T
        if moment_match == "sphere_frame":
            x *= er / np.linalg.norm(x, axis=1, keepdims=True)
        offsets = np.cumsum([0, *sizes])
        blocks = [
            x[offsets[i] : offsets[i + 1]].astype(np.float32)
            for i in range(len(sizes))
        ]
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--design",
        choices=("iid", "sobol", "lhs", "lattice", "orthogonal"),
        default="sobol",
    )
    parser.add_argument("--sphere", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--antithetic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor", choices=("none", "gate", "ols"), default="none")
    parser.add_argument(
        "--terminal",
        choices=("direct", "gaussian", "edgeworth3", "edgeworth4"),
        default="direct",
        help="Replace the final sample mean by a fitted marginal closure.",
    )
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument(
        "--reuse-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute each seeded design once and reuse it across MLPs.",
    )
    parser.add_argument(
        "--moment-match",
        choices=("none", "whiten", "sphere_frame"),
        default="none",
        help="Offline global covariance matching for a reused design.",
    )
    parser.add_argument("--frame-iterations", type=int, default=6)
    parser.add_argument(
        "--rotation",
        choices=("none", "mean_field_sensitivity"),
        default="none",
        help="Rotate invariant Gaussian/spherical inputs before the MLP.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--save-vectors",
        action="store_true",
        help="Include final predictions and targets in the JSON research artifact.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.data, args.indices)
    design_cache = {}
    if args.reuse_inputs:
        width = rows[0][1].shape[-1]
        for seed in args.seeds:
            design_cache[seed] = precompute_design(
                args.design,
                width,
                args.samples,
                seed,
                args.antithetic,
                args.sphere,
                args.chunk,
                args.moment_match,
                args.frame_iterations,
            )

    results = []
    for row_index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        input_rotation = (
            mean_field_sensitivity_rotation(weights)
            if args.rotation == "mean_field_sensitivity"
            else None
        )
        for seed in args.seeds:
            prediction, elapsed = estimate(
                weights,
                samples=args.samples,
                seed=seed,
                design_kind=args.design,
                sphere=args.sphere,
                antithetic=args.antithetic,
                anchor=args.anchor,
                terminal=args.terminal,
                chunk=args.chunk,
                input_blocks=design_cache.get(seed),
                input_rotation=input_rotation,
            )
            final_mse = float(np.mean(np.square(prediction[-1] - targets[-1])))
            all_mse = float(np.mean(np.square(prediction - targets)))
            record = {
                "index": row_index,
                "name": name,
                "seed": seed,
                "samples": args.samples,
                "design": args.design,
                "sphere": args.sphere,
                "antithetic": args.antithetic,
                "anchor": args.anchor,
                "terminal": args.terminal,
                "reuse_inputs": args.reuse_inputs,
                "moment_match": args.moment_match,
                "rotation": args.rotation,
                "seconds": elapsed,
                "final_mse": final_mse,
                "all_layer_mse": all_mse,
                "mse_seconds": final_mse * elapsed,
            }
            if args.save_vectors:
                record["final_prediction"] = prediction[-1].tolist()
                record["final_target"] = targets[-1].tolist()
            results.append(record)
            print(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("final_prediction", "final_target")
                },
                flush=True,
            )

    summary = {
        "n_runs": len(results),
        "mean_final_mse": float(np.mean([r["final_mse"] for r in results])),
        "mean_seconds": float(np.mean([r["seconds"] for r in results])),
        "mean_mse_seconds": float(np.mean([r["mse_seconds"] for r in results])),
        "median_final_mse": float(np.median([r["final_mse"] for r in results])),
    }
    print({"summary": summary})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2))


if __name__ == "__main__":
    main()
