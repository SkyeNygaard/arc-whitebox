"""Backend-generic Kerdock suffix compiler core.

The implementation is tested with NumPy and is written against the small array
surface shared by NumPy and flopscope.numpy. It avoids in-place writes so it can
run under immutable tracked arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any, Callable, Iterable
import math
import numpy as np

WIDTH = 256
DEPTH = 32
N_POINTS = 66_048
KERDOCK_BASES_WITH_COORDINATES = 129
BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 100_000_000_000.0


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    depths: tuple[int, ...]
    pilot_cols: int
    rare_threshold: int
    final_correction_alpha: float
    guard_ratio: float | None


SPECS = {
    "baseline": CandidateSpec("baseline", (), 0, 0, 0.0, None),
    "two_layer": CandidateSpec("two_layer", (2,), 8, 1, 1.0, None),
    "fixed_three": CandidateSpec("fixed_three", (3,), 8, 8, 0.875, 0.995),
    "adaptive_2_6": CandidateSpec("adaptive_2_6", (2, 3, 4, 5, 6), 8, 1, 1.0, 0.995),
}


def pilot_indices(width: int = WIDTH, cols: int = 8) -> np.ndarray:
    """Balanced nested pilot: ``cols`` columns from all 129 signed bases."""
    if width != 256:
        raise ValueError("production pilot order is defined only for width 256")
    order = np.array(
        [0, 128, 64, 192, 32, 160, 96, 224,
         16, 144, 80, 208, 48, 176, 112, 240],
        dtype=np.int64,
    )
    if not 1 <= cols <= len(order):
        raise ValueError(f"cols must be in [1,{len(order)}]")
    selected = order[:cols]
    return np.array(
        [((basis * width + col) * 2 + sign)
         for basis in range(KERDOCK_BASES_WITH_COORDINATES)
         for col in selected
         for sign in (0, 1)],
        dtype=np.int64,
    )


def _nonzero(mask: Any, xp: Any) -> Any:
    if hasattr(xp, "flatnonzero"):
        return xp.flatnonzero(mask)
    return xp.nonzero(mask)[0]


def classify_layer(pilot_preactivation: Any, rare: int, xp: Any) -> tuple[Any, Any, Any]:
    positive = xp.sum(pilot_preactivation > 0, axis=0)
    negative = pilot_preactivation.shape[0] - positive
    stable = xp.minimum(positive, negative) <= rare
    stable_on = stable & (positive >= negative)
    stable_off = stable & ~stable_on
    kink = ~stable
    return _nonzero(stable_on, xp), _nonzero(stable_off, xp), _nonzero(kink, xp)


def cost_proxy(
    suffix_depth: int,
    classes: Iterable[tuple[Any, Any, Any]],
    pilot_rows: int,
    n_rows: int,
    width: int,
    total_depth: int = DEPTH,
) -> float:
    kink_counts = [int(c[2].shape[0]) for c in classes]
    suffix_equivalents = suffix_depth * pilot_rows / n_rows
    for layer, count in enumerate(kink_counts):
        suffix_equivalents += count / width
        for previous in range(layer):
            suffix_equivalents += kink_counts[previous] * count / (width * width)
    return (total_depth - suffix_depth + suffix_equivalents) / total_depth


def _submatrix(matrix: Any, rows: Any, cols: Any) -> Any:
    return matrix[rows[:, None], cols[None, :]]


def _scatter_partition(
    on: Any,
    kink: Any,
    off: Any,
    on_values: Any,
    kink_values: Any,
    leading_shape: tuple[int, ...],
    width: int,
    xp: Any,
) -> Any:
    """Functionally scatter partition values into original column order."""
    off_values = xp.zeros(leading_shape + (int(off.shape[0]),), dtype=on_values.dtype)
    indices = xp.concatenate((on, kink, off), axis=0)
    values = xp.concatenate((on_values, kink_values, off_values), axis=-1)
    if int(indices.shape[0]) != width:
        raise RuntimeError("classification partition does not cover all outputs")
    return values[..., xp.argsort(indices)]


def exact_pilot_suffix(anchor_pilot: Any, suffix_weights: list[Any], xp: Any) -> tuple[list[Any], Any]:
    preactivations: list[Any] = []
    activation = anchor_pilot
    for weight in suffix_weights:
        h = activation @ weight
        preactivations.append(h)
        activation = xp.maximum(h, 0.0)
    return preactivations, activation


def compile_suffix_mean(
    anchor: Any,
    suffix_weights: list[Any],
    classes: list[tuple[Any, Any, Any]],
    pilot_rows: Any,
    exact_pilot_output: Any,
    correction_alpha: float,
    xp: Any,
) -> Any:
    """Compile stable paths and exactly propagate only kink coordinates.

    ``anchor`` is the full Kerdock cloud immediately before the compiled suffix.
    ``exact_pilot_output`` is obtained by running the same pilot rows through the
    exact suffix. No full final activation is required.
    """
    n_rows, width = anchor.shape
    p_rows = int(pilot_rows.shape[0])
    if len(suffix_weights) < 2:
        raise ValueError("compiler requires at least two suffix layers")
    if len(suffix_weights) != len(classes):
        raise ValueError("weights/classes length mismatch")

    x = anchor.astype(xp.float64)
    xp_anchor = x[pilot_rows]
    mean_x = xp.mean(x, axis=0)
    residuals: list[Any] = []
    pilot_residuals: list[Any] = []

    on0, _, kink0 = classes[0]
    if int(kink0.shape[0]):
        h0 = x @ suffix_weights[0][:, kink0].astype(xp.float64)
        r0 = xp.maximum(h0, 0.0)
        rp0 = r0[pilot_rows]
    else:
        r0 = xp.empty((n_rows, 0), dtype=xp.float64)
        rp0 = xp.empty((p_rows, 0), dtype=xp.float64)
    residuals.append(r0)
    pilot_residuals.append(rp0)
    basis = suffix_weights[0][:, on0].astype(xp.float64)
    coefficients: list[Any] = []

    for layer in range(1, len(suffix_weights)):
        previous_on = classes[layer - 1][0]
        previous_kink = classes[layer - 1][2]
        on, off, kink = classes[layer]
        weight = suffix_weights[layer].astype(xp.float64)
        final = layer == len(suffix_weights) - 1

        def representation_to(target: Any) -> tuple[Any, list[Any]]:
            target_count = int(target.shape[0])
            if target_count == 0:
                return (
                    xp.zeros((width, 0), dtype=xp.float64),
                    [xp.zeros((int(r.shape[1]), 0), dtype=xp.float64) for r in residuals],
                )
            if int(previous_on.shape[0]):
                w_on_target = _submatrix(weight, previous_on, target)
                b_target = basis @ w_on_target
                c_target = [c @ w_on_target for c in coefficients]
            else:
                b_target = xp.zeros((width, target_count), dtype=xp.float64)
                c_target = [
                    xp.zeros((int(r.shape[1]), target_count), dtype=xp.float64)
                    for r in residuals[:-1]
                ]
            if int(previous_kink.shape[0]):
                c_target.append(_submatrix(weight, previous_kink, target))
            else:
                c_target.append(xp.zeros((0, target_count), dtype=xp.float64))
            return b_target, c_target

        if final:
            b_kink, c_kink = representation_to(kink)
            if int(kink.shape[0]):
                h_kink = x @ b_kink
                for residual, coefficient in zip(residuals, c_kink):
                    if int(residual.shape[1]):
                        h_kink = h_kink + residual @ coefficient
                y_kink = xp.maximum(h_kink, 0.0)
                mean_kink = xp.mean(y_kink, axis=0)
                pilot_kink = y_kink[pilot_rows]
            else:
                mean_kink = xp.empty((0,), dtype=xp.float64)
                pilot_kink = xp.empty((p_rows, 0), dtype=xp.float64)

            b_on, c_on = representation_to(on)
            mean_on = mean_x @ b_on
            for residual, coefficient in zip(residuals, c_on):
                if int(residual.shape[1]):
                    mean_on = mean_on + xp.mean(residual, axis=0) @ coefficient
            if int(on.shape[0]):
                pilot_on = xp_anchor @ b_on
                for pilot_residual, coefficient in zip(pilot_residuals, c_on):
                    if int(pilot_residual.shape[1]):
                        pilot_on = pilot_on + pilot_residual @ coefficient
            else:
                pilot_on = xp.empty((p_rows, 0), dtype=xp.float64)

            compiled_mean = _scatter_partition(
                on, kink, off, mean_on, mean_kink, (), width, xp
            )
            compiled_pilot = _scatter_partition(
                on, kink, off, pilot_on, pilot_kink, (p_rows,), width, xp
            )
            correction = xp.mean(
                exact_pilot_output.astype(xp.float64) - compiled_pilot,
                axis=0,
            )
            return compiled_mean + correction_alpha * correction

        b_kink, c_kink = representation_to(kink)
        if int(kink.shape[0]):
            h_kink = x @ b_kink
            for residual, coefficient in zip(residuals, c_kink):
                if int(residual.shape[1]):
                    h_kink = h_kink + residual @ coefficient
            next_residual = xp.maximum(h_kink, 0.0)
            next_pilot_residual = next_residual[pilot_rows]
        else:
            next_residual = xp.empty((n_rows, 0), dtype=xp.float64)
            next_pilot_residual = xp.empty((p_rows, 0), dtype=xp.float64)
        b_on, c_on = representation_to(on)
        basis = b_on
        coefficients = c_on
        residuals.append(next_residual)
        pilot_residuals.append(next_pilot_residual)

    raise RuntimeError("unreachable compiler exit")


def _timed(timings: dict[str, float], name: str, function: Callable[[], Any]) -> Any:
    started = perf_counter()
    value = function()
    timings[name] = timings.get(name, 0.0) + (perf_counter() - started)
    return value


def propagate(activation: Any, weights: list[Any], matmul: Callable[[Any, Any], Any], xp: Any) -> Any:
    out = activation
    for weight in weights:
        out = xp.maximum(matmul(out, weight), 0.0)
    return out


def run_candidate(
    first_activation: Any,
    remaining_weights: list[Any],
    spec: CandidateSpec,
    matmul: Callable[[Any, Any], Any],
    xp: Any,
    pilot_rows: Any | None = None,
    total_depth: int = DEPTH,
) -> tuple[Any, dict[str, Any]]:
    """Return final mean and detailed compiler diagnostics."""
    timings: dict[str, float] = {}
    if spec.name == "baseline":
        final = _timed(
            timings,
            "full_propagation_s",
            lambda: propagate(first_activation, remaining_weights, matmul, xp),
        )
        mean = _timed(timings, "final_reduction_s", lambda: xp.mean(final.astype(xp.float64), axis=0))
        return mean, {
            "candidate": spec.name,
            "selected_depth": 0,
            "predicted_cost_ratio": 1.0,
            "fallback": False,
            "kink_counts": [],
            "stable_on_counts": [],
            "stable_off_counts": [],
            "timings": timings,
        }

    if pilot_rows is None:
        pilot_rows = pilot_indices(cols=spec.pilot_cols)
    max_depth = max(spec.depths)
    if max_depth > len(remaining_weights):
        raise ValueError("suffix depth exceeds remaining network depth")
    prefix_weights = remaining_weights[:-max_depth]
    max_suffix_weights = remaining_weights[-max_depth:]
    anchor_max = _timed(
        timings,
        "prefix_propagation_s",
        lambda: propagate(first_activation, prefix_weights, matmul, xp),
    )

    pilot_anchor = _timed(
        timings,
        "pilot_pack_s",
        lambda: anchor_max[pilot_rows],
    )
    max_preactivations, exact_pilot_output = _timed(
        timings,
        "pilot_rollout_s",
        lambda: exact_pilot_suffix(pilot_anchor, max_suffix_weights, xp),
    )

    candidate_meta: list[tuple[float, int, list[tuple[Any, Any, Any]]]] = []
    classify_started = perf_counter()
    for depth in spec.depths:
        offset = max_depth - depth
        classes = [
            classify_layer(h, spec.rare_threshold, xp)
            for h in max_preactivations[offset:]
        ]
        ratio = cost_proxy(
            depth,
            classes,
            int(pilot_rows.shape[0]),
            int(first_activation.shape[0]),
            int(first_activation.shape[1]),
            total_depth,
        )
        candidate_meta.append((ratio, depth, classes))
    timings["classification_selection_s"] = perf_counter() - classify_started
    predicted_ratio, selected_depth, selected_classes = min(candidate_meta, key=lambda row: (row[0], row[1]))

    fallback = spec.guard_ratio is not None and predicted_ratio > spec.guard_ratio
    if fallback:
        final = _timed(
            timings,
            "fallback_suffix_s",
            lambda: propagate(anchor_max, max_suffix_weights, matmul, xp),
        )
        mean = _timed(timings, "final_reduction_s", lambda: xp.mean(final.astype(xp.float64), axis=0))
    else:
        offset = max_depth - selected_depth
        selected_anchor = _timed(
            timings,
            "selected_anchor_extension_s",
            lambda: propagate(anchor_max, max_suffix_weights[:offset], matmul, xp),
        )
        selected_weights = max_suffix_weights[offset:]
        mean = _timed(
            timings,
            "symbolic_composition_s",
            lambda: compile_suffix_mean(
                selected_anchor,
                selected_weights,
                selected_classes,
                pilot_rows,
                exact_pilot_output,
                spec.final_correction_alpha,
                xp,
            ),
        )

    return mean, {
        "candidate": spec.name,
        "spec": asdict(spec),
        "selected_depth": selected_depth,
        "predicted_cost_ratio": predicted_ratio,
        "fallback": fallback,
        "kink_counts": [int(c[2].shape[0]) for c in selected_classes],
        "stable_on_counts": [int(c[0].shape[0]) for c in selected_classes],
        "stable_off_counts": [int(c[1].shape[0]) for c in selected_classes],
        "timings": timings,
    }


def adjusted_score(raw_mse: float, effective_compute: float) -> float:
    return raw_mse * max(0.1, effective_compute / BUDGET)
