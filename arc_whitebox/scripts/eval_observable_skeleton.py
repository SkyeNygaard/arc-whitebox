"""Observable-projected leading K3/K4 Hermite skeleton correction.

The Gaussian K2 trajectory supplies the dressed two-point function.  At each
layer, connected Hermite tree diagrams approximate the marginal K3/K4 injected
by the previous ReLU.  A linearized backward response transports each local
Edgeworth mean correction to the final outputs.  Truncated SVDs test whether
that output-response family is actually compressible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "mlp_cumulant_propagation" / "src"
sys.path.insert(0, str(VENDOR))

from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402


WIDTH = 256
DEPTH = 32
RANKS = (8, 16, 32)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def build_mlp(weights: np.ndarray) -> MLP:
    depth, width, _ = weights.shape
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=depth + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=[2.0] * depth + [1.0],
        b_var=0.0,
        b_mean=0.0,
    )
    with torch.no_grad():
        for layer, weight in enumerate(weights):
            mlp.Ws[layer].weight.copy_(torch.as_tensor(weight.T))
        mlp.Ws[-1].weight.copy_(torch.eye(width))
    return mlp


def k2_trajectory(mlp: MLP) -> dict:
    return mlp_kprop(
        mlp,
        {1: torch.zeros(WIDTH), 2: torch.eye(WIDTH)},
        k_max=2,
        kind=SIMPLE,
        use_avg_metric=True,
        output_all=True,
    )


def pre_state(tower: dict, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    state = tower[f"pre{layer}"]
    return state[1].core, state[2].core


def relu_hermite_coefficients(
    mean: torch.Tensor, covariance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    variance = covariance.diagonal().clamp_min(1e-12)
    sigma = variance.sqrt()
    z = mean / sigma
    phi = torch.exp(-0.5 * z.square()) / SQRT_2PI
    cdf = torch.special.ndtr(z)
    # f(mu + sigma Z) = sum_p a_p He_p(Z), through p=3.
    a1 = sigma * cdf
    a2 = 0.5 * sigma * phi
    a3 = -(sigma * z * phi) / 6.0
    scale = sigma[:, None] * sigma[None, :]
    corr = (covariance / scale.clamp_min(1e-12)).clamp(-1.0, 1.0)
    corr.fill_diagonal_(1.0)
    return a1, a2, a3, corr


def local_tree_cumulants(
    weight: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    corr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading connected trees for K3 and K4 after a weighted sum.

    S[o,i] = sum_j W[o,j] a1[j] rho[i,j].
    K3 tree: 6 sum_i W[o,i] a2[i] S[o,i]^2.
    K4 trees: the H3-H1-H1-H1 star and H2-H2-H1-H1 path.
    """
    response = (weight * a1[None, :]) @ corr.T
    weighted_a2 = weight * a2[None, :]
    k3 = 6.0 * (weighted_a2 * response.square()).sum(dim=1)
    star = 24.0 * (weight * a3[None, :] * response.pow(3)).sum(dim=1)
    path_vector = weighted_a2 * response
    path = 48.0 * ((path_vector @ corr) * path_vector).sum(dim=1)
    return k3, star + path


def edgeworth_local(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    k3: torch.Tensor,
    k4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    variance = covariance.diagonal().clamp_min(1e-12)
    sigma = variance.sqrt()
    z = mean / sigma
    phi = torch.exp(-0.5 * z.square()) / SQRT_2PI
    d3 = -(z * phi) / variance
    d4 = ((z.square() - 1.0) * phi) / (variance * sigma)
    return (k3 / 6.0) * d3, (k4 / 24.0) * d4


def response_matrices(
    weights: torch.Tensor, tower: dict
) -> tuple[list[torch.Tensor], list[dict[int, torch.Tensor]], list[dict]]:
    """Return exact and rank-truncated responses from each activation to act31."""
    exact: list[torch.Tensor] = [torch.empty(0)] * DEPTH
    projected: list[dict[int, torch.Tensor]] = [dict() for _ in range(DEPTH)]
    diagnostics: list[dict] = [dict() for _ in range(DEPTH)]
    response = torch.eye(WIDTH)
    for layer in range(DEPTH - 1, -1, -1):
        exact[layer] = response
        u, s, vh = torch.linalg.svd(response, full_matrices=False)
        total = s.square().sum().clamp_min(1e-30)
        diagnostics[layer] = {
            "layer": layer,
            "energy": {
                str(rank): float(s[:rank].square().sum() / total)
                for rank in RANKS
            },
            "stable_rank": float(total / s[0].square().clamp_min(1e-30)),
        }
        for rank in RANKS:
            projected[layer][rank] = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        if layer > 0:
            mean, covariance = pre_state(tower, layer)
            sigma = covariance.diagonal().clamp_min(1e-12).sqrt()
            gate = torch.special.ndtr(mean / sigma)
            jacobian = gate[:, None] * weights[layer]
            response = response @ jacobian
    return exact, projected, diagnostics


def collect_one(mlp_id: int) -> dict:
    data = np.load(f"/tmp/phase1_mlp{mlp_id}.npz")
    weight_array = np.asarray(data["weights"], dtype=np.float32)
    target = np.asarray(data["means"], dtype=np.float64)[-1]
    mlp = build_mlp(weight_array)
    weights = torch.as_tensor(weight_array.transpose(0, 2, 1)).contiguous()
    start = time.perf_counter()
    tower = k2_trajectory(mlp)
    baseline = tower["act31"][1].core
    exact_response, projected_response, response_diag = response_matrices(
        weights, tower
    )
    signals = {
        "exact": {"k3": torch.zeros(WIDTH), "k4": torch.zeros(WIDTH)}
    }
    for rank in RANKS:
        signals[str(rank)] = {
            "k3": torch.zeros(WIDTH),
            "k4": torch.zeros(WIDTH),
        }
    alignment = []
    # pre0 is exactly Gaussian.  Non-Gaussian source terms begin at pre1.
    for layer in range(1, DEPTH):
        prev_mean, prev_covariance = pre_state(tower, layer - 1)
        a1, a2, a3, corr = relu_hermite_coefficients(
            prev_mean, prev_covariance
        )
        k3, k4 = local_tree_cumulants(
            weights[layer], a1, a2, a3, corr
        )
        mean, covariance = pre_state(tower, layer)
        delta3, delta4 = edgeworth_local(mean, covariance, k3, k4)
        exact3 = exact_response[layer] @ delta3
        exact4 = exact_response[layer] @ delta4
        signals["exact"]["k3"] += exact3
        signals["exact"]["k4"] += exact4
        layer_alignment = {"layer": layer}
        for rank in RANKS:
            projected3 = projected_response[layer][rank] @ delta3
            projected4 = projected_response[layer][rank] @ delta4
            signals[str(rank)]["k3"] += projected3
            signals[str(rank)]["k4"] += projected4
            denom3 = exact3.square().sum().clamp_min(1e-30)
            denom4 = exact4.square().sum().clamp_min(1e-30)
            layer_alignment[str(rank)] = {
                "k3_capture": float(
                    1.0
                    - (projected3 - exact3).square().sum() / denom3
                ),
                "k4_capture": float(
                    1.0
                    - (projected4 - exact4).square().sum() / denom4
                ),
            }
        alignment.append(layer_alignment)
    return {
        "mlp_id": mlp_id,
        "target": target,
        "baseline": baseline.detach().cpu().numpy().astype(np.float64),
        "signals": {
            key: {
                order: value.detach().cpu().numpy().astype(np.float64)
                for order, value in by_order.items()
            }
            for key, by_order in signals.items()
        },
        "response_diagnostics": response_diag,
        "alignment": alignment,
        "seconds": time.perf_counter() - start,
    }


def fit_coefficients(rows: list[dict], key: str, exclude: int | None = None):
    selected = [row for i, row in enumerate(rows) if i != exclude]
    design = np.concatenate(
        [
            np.stack(
                [row["signals"][key]["k3"], row["signals"][key]["k4"]],
                axis=1,
            )
            for row in selected
        ],
        axis=0,
    )
    residual = np.concatenate(
        [row["target"] - row["baseline"] for row in selected]
    )
    gram = design.T @ design
    ridge = 1e-10 * np.trace(gram) / 2.0
    return np.linalg.solve(gram + ridge * np.eye(2), design.T @ residual)


def mse(row: dict, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - row["target"]) ** 2))


def summarize_selection(rows: list[dict]) -> dict:
    baseline = float(np.mean([mse(row, row["baseline"]) for row in rows]))
    summaries = {}
    for key in ("exact", "8", "16", "32"):
        coefficient = fit_coefficients(rows, key)
        fitted = []
        analytic = []
        loono = []
        correlations = []
        for index, row in enumerate(rows):
            design = np.stack(
                [row["signals"][key]["k3"], row["signals"][key]["k4"]],
                axis=1,
            )
            true_residual = row["target"] - row["baseline"]
            signal = design @ coefficient
            fitted.append(mse(row, row["baseline"] + signal))
            analytic.append(mse(row, row["baseline"] + design.sum(axis=1)))
            if len(rows) > 1:
                loono_coefficient = fit_coefficients(rows, key, exclude=index)
                loono.append(
                    mse(row, row["baseline"] + design @ loono_coefficient)
                )
            correlations.append(
                float(
                    np.corrcoef(signal, true_residual)[0, 1]
                    if np.std(signal) > 0 and np.std(true_residual) > 0
                    else 0.0
                )
            )
        summaries[key] = {
            "coefficients": {"k3": coefficient[0], "k4": coefficient[1]},
            "analytic_ratio": float(np.mean(analytic) / baseline),
            "fit_ratio": float(np.mean(fitted) / baseline),
            "loono_ratio": (
                float(np.mean(loono) / baseline) if loono else None
            ),
            "mean_within_network_correlation": float(np.mean(correlations)),
        }
    response_energy = {
        str(rank): {
            "mean_all_layers": float(
                np.mean(
                    [
                        diag["energy"][str(rank)]
                        for row in rows
                        for diag in row["response_diagnostics"]
                    ]
                )
            ),
            "mean_layers_0_23": float(
                np.mean(
                    [
                        diag["energy"][str(rank)]
                        for row in rows
                        for diag in row["response_diagnostics"][:24]
                    ]
                )
            ),
            "mean_layers_24_31": float(
                np.mean(
                    [
                        diag["energy"][str(rank)]
                        for row in rows
                        for diag in row["response_diagnostics"][24:]
                    ]
                )
            ),
        }
        for rank in RANKS
    }
    return {
        "baseline_mse": baseline,
        "methods": summaries,
        "response_energy": response_energy,
    }


def flop_model() -> dict:
    n = WIDTH
    nonlinear_layers = DEPTH - 1
    k2 = 2_147_483_648
    s_matmul = nonlinear_layers * 2 * n**3
    k4_path_matmul = nonlinear_layers * 2 * n**3
    full_response_matmul = nonlinear_layers * 2 * n**3
    return {
        "convention": "multiply and add count as two FLOPs",
        "k2_forward_existing_lower_bound": k2,
        "k3_tree_S_matmuls": s_matmul,
        "k4_path_matmuls": k4_path_matmul,
        "full_backward_response_matmuls": full_response_matmul,
        "subtotal_matmuls": k2
        + s_matmul
        + k4_path_matmul
        + full_response_matmul,
        "subtotal_budget_fraction": (
            k2 + s_matmul + k4_path_matmul + full_response_matmul
        )
        / 272_000_000_000,
        "note": (
            "Excludes elementwise work and diagnostic SVDs. A deployable exact-"
            "response version needs no SVD. Rank-r transport after a chosen "
            "handoff replaces each later 2n^3 product by about 2n^2r."
        ),
    }


def serializable_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "mlp_id": row["mlp_id"],
            "seconds": row["seconds"],
            "baseline_mse": mse(row, row["baseline"]),
            "response_diagnostics": row["response_diagnostics"],
            "alignment": row["alignment"],
        }
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "observable_skeleton_selection.json",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    rows = []
    for mlp_id in args.ids:
        row = collect_one(mlp_id)
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": mlp_id,
                    "seconds": row["seconds"],
                    "baseline_mse": mse(row, row["baseline"]),
                }
            ),
            flush=True,
        )
    summary = summarize_selection(rows)
    loono_ratios = [
        method["loono_ratio"]
        for method in summary["methods"].values()
        if method["loono_ratio"] is not None
    ]
    best_loono = min(loono_ratios) if loono_ratios else math.inf
    artifact = {
        "ids": args.ids,
        "summary": summary,
        "holdout_gate_passed": best_loono <= 0.9,
        "best_loono_ratio": best_loono,
        "flops": flop_model(),
        "per_id": serializable_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
