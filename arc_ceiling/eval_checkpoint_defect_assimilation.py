"""Assimilate a Kerdock checkpoint K3 defect into the late cubic anchor.

The factorized K3 recursion represents the connected third cumulant as a
symmetrized CP tensor.  Its factor columns are append-only: columns already
present at checkpoint ``l`` are transported by expected gates, while later
columns are newly generated cumulant sources.  This lets us isolate the
inherited checkpoint contribution at layer 29 exactly.

For a checkpoint activation cloud, form its empirical connected K3 tensor in
CP form (one centered activation vector per row), transport it with the same
linear map inferred from factor columns, and replace only the inherited
factorized contribution:

    K3_corrected,29 =
        K3_factorized,29
        + T_l(K3_sample,l - K3_factorized,l).

No layer-29 state or final targets enter the correction.  Oracle layer-29
mean/covariance are used only for this first ceiling experiment, isolating
whether checkpoint defect transport improves the contracted K3 anchor at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import whest.gaussmath as gm  # noqa: E402

from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    empirical_c21_state,
    pointwise_features,
    raw_m21_from_cumulants,
)
from eval_edgeworth_cubic_anchor import contraction  # noqa: E402
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def forward_checkpoints(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    checkpoints: list[int],
    target_layer: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    wanted = set(checkpoints)
    snapshots: dict[int, np.ndarray] = {}
    pre = points @ (rotation @ weights[0].astype(np.float32))
    activation = np.maximum(pre, 0.0)
    if 0 in wanted:
        snapshots[0] = activation.copy()
    target = activation if target_layer == 0 else None
    for layer in range(1, len(weights)):
        pre = activation @ weights[layer]
        activation = np.maximum(pre, 0.0)
        if layer in wanted:
            snapshots[layer] = activation.copy()
        if layer == target_layer:
            target = activation.copy()
    if target is None or set(snapshots) != wanted:
        raise ValueError((set(snapshots), wanted, target_layer))
    return snapshots, target, activation


def factor_c21(
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    chunk: int = 4096,
) -> np.ndarray:
    """K[i,i,j] for Sym(sum_r A_ir B_jr C_kr), including its diagonal."""
    a, b, c = factors
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError((a.shape, b.shape, c.shape))
    result = np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
    for start in range(0, a.shape[1], chunk):
        stop = min(start + chunk, a.shape[1])
        aa = a[:, start:stop]
        bb = b[:, start:stop]
        cc = c[:, start:stop]
        result += (
            (aa * bb) @ cc.T
            + (aa * cc) @ bb.T
            + (bb * cc) @ aa.T
        ) / 3.0
    return result


def infer_transport(
    checkpoint_factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    ridge_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Recover the common linear tail map from inherited factor columns."""
    rank = checkpoint_factors[0].shape[1]
    maps = []
    reconstruction = []
    for source, full_target in zip(checkpoint_factors, target_factors):
        target = full_target[:, :rank]
        gram = source @ source.T
        ridge = ridge_fraction * max(float(np.trace(gram)) / len(gram), 1e-30)
        transport = (target @ source.T) @ np.linalg.inv(
            gram + ridge * np.eye(len(gram))
        )
        maps.append(transport)
        reconstruction.append(
            float(
                np.linalg.norm(transport @ source - target)
                / max(np.linalg.norm(target), 1e-30)
            )
        )
    transport = np.mean(maps, axis=0)
    disagreement = max(
        float(np.linalg.norm(one - transport) / max(np.linalg.norm(transport), 1e-30))
        for one in maps
    )
    averaged_reconstruction = [
        float(
            np.linalg.norm(transport @ source - target[:, :rank])
            / max(np.linalg.norm(target[:, :rank]), 1e-30)
        )
        for source, target in zip(checkpoint_factors, target_factors)
    ]
    return transport, {
        "individual_reconstruction_max": float(max(reconstruction)),
        "map_disagreement_max": disagreement,
        "average_map_reconstruction_max": float(max(averaged_reconstruction)),
    }


def empirical_transported_c21(
    checkpoint_activation: np.ndarray,
    transport: np.ndarray,
    basis_ids: np.ndarray,
) -> np.ndarray:
    blocks = checkpoint_activation.reshape(
        N_BASES,
        ROWS_PER_BASIS,
        WIDTH,
    )
    values = blocks[basis_ids].reshape(-1, WIDTH).astype(np.float64)
    centered = values - np.mean(values, axis=0)
    transported = centered @ transport.T
    transported -= np.mean(transported, axis=0)
    return (np.square(transported).T @ transported) / len(transported)


def expected_gate_transport(
    weights: np.ndarray,
    checkpoint: int,
    target_layer: int,
    pre_mean: np.ndarray,
    pre_second: np.ndarray,
) -> np.ndarray:
    """Row-vector Jacobian from post-checkpoint to post-target."""
    transport = np.eye(WIDTH, dtype=np.float64)
    for layer in range(checkpoint + 1, target_layer + 1):
        transport = transport @ weights[layer].astype(np.float64)
        mean = np.asarray(pre_mean[layer], dtype=np.float64)
        variance = np.maximum(
            np.diag(np.asarray(pre_second[layer], dtype=np.float64))
            - np.square(mean),
            1e-20,
        )
        gate = ndtr(mean / np.sqrt(variance))
        transport *= gate[None, :]
    return transport


def sample_initialized_tail_k2(
    weights: np.ndarray,
    checkpoint_activation: np.ndarray,
    checkpoint: int,
    target_layer: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Gaussian K2 tail initialized from the Kerdock checkpoint cloud.

    Positive homogeneity gives the exact radial conversions

        E_G[H]  = E_K[H],
        E_G[HH'] = d / E[R]^2 E_K[HH'].

    Only the angular quadrature error remains at the checkpoint.  Propagating
    five or so late layers is substantially less drift-prone than a free K2
    rollout from the input and supplies deployable gates and target marginals.
    """
    activation = np.asarray(checkpoint_activation, dtype=np.float64)
    mean = np.mean(activation, axis=0)
    second = (
        (WIDTH / np.square(radius))
        * (activation.T @ activation)
        / len(activation)
    )
    covariance = 0.5 * (
        second - np.outer(mean, mean)
        + (second - np.outer(mean, mean)).T
    )
    pre_means: dict[int, np.ndarray] = {}
    pre_seconds: dict[int, np.ndarray] = {}
    for layer in range(checkpoint + 1, target_layer + 1):
        weight = np.asarray(weights[layer], dtype=np.float64)
        pre_mean = mean @ weight
        pre_covariance = weight.T @ covariance @ weight
        pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
        pre_second = pre_covariance + np.outer(pre_mean, pre_mean)
        pre_means[layer] = pre_mean
        pre_seconds[layer] = pre_second
        mean, covariance = gm.relu_cov_from_gauss(
            pre_mean,
            pre_covariance,
            n_nodes=8,
        )
        covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance, pre_means, pre_seconds


def load_factors(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(data[f"k3_factor{index}"], dtype=np.float64)
        for index in range(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[160, 161])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--target-layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--control-ranks",
        type=int,
        nargs="+",
        default=[4],
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--transport-ridge", type=float, default=1e-12)
    parser.add_argument(
        "--transport-source",
        choices=["oracle_gates", "inferred_factors"],
        default="oracle_gates",
    )
    parser.add_argument(
        "--cheap-bases",
        type=int,
        nargs="*",
        default=[8],
    )
    parser.add_argument(
        "--defect-scales",
        type=float,
        nargs="*",
        default=[1.0],
    )
    parser.add_argument(
        "--sample-only-scales",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Also test a factor-free K3 anchor formed only from the empirical "
            "checkpoint cumulant transported to the target.  This isolates "
            "the deployable low-cost branch from the expensive full K3 base."
        ),
    )
    parser.add_argument(
        "--skip-factorized-base",
        action="store_true",
        help=(
            "Do not load or score the full factorized target K3.  Intended "
            "for the sample-only branch, which needs only a cheap expected-"
            "gate tail map."
        ),
    )
    parser.add_argument(
        "--construction-state",
        choices=["oracle", "sample_tail_k2"],
        default="oracle",
        help=(
            "State used for the expected gates and target mean/covariance. "
            "sample_tail_k2 is deployable and initializes a Gaussian tail "
            "rollout from the main Kerdock checkpoint cloud."
        ),
    )
    parser.add_argument(
        "--factor-root",
        type=Path,
        default=HERE / "results" / "factorized_k3_checkpoint_factors_smoke2",
    )
    parser.add_argument("--compact-state-dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "checkpoint_defect_assimilation_smoke2.json",
    )
    args = parser.parse_args()

    cheap_basis_sets = {}
    for count in sorted(set(args.cheap_bases)):
        if count <= 0 or count > N_BASES:
            raise ValueError(count)
        basis_ids = np.unique(
            np.rint(
                np.linspace(0, N_BASES - 1, count)
            ).astype(np.int64)
        )
        if len(basis_ids) != count:
            raise ValueError(basis_ids)
        cheap_basis_sets[f"b{count}"] = basis_ids
    full_basis_ids = np.arange(N_BASES, dtype=np.int64)

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []
    method_to_anchor: dict[str, str] = {}

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        snapshots, target_activation, final = forward_checkpoints(
            weights,
            points,
            rotation,
            args.checkpoints,
            args.target_layer,
        )
        left, right, target_sample_raw_m21 = empirical_c21_state(
            target_activation,
            args.rank,
        )
        sample_anchor = (
            np.einsum(
                "ik,ij,jk->k",
                left,
                target_sample_raw_m21,
                right,
            )
            / np.square(radius)
        )
        target_raw_features = (
            (np.square(target_activation.astype(np.float64)) @ left)
            * (target_activation.astype(np.float64) @ right)
            / np.square(radius)
        )
        target_block_means = target_raw_features.reshape(
            N_BASES,
            ROWS_PER_BASIS,
            args.rank,
        ).mean(axis=1)
        sample_anchor_se = np.std(
            target_block_means,
            axis=0,
            ddof=1,
        ) / np.sqrt(N_BASES)

        with np.load(moment_path(index)) as oracle:
            oracle_target_mean = np.asarray(
                oracle["mean"][args.target_layer],
                dtype=np.float64,
            )
            oracle_target_second = np.asarray(
                oracle["M11"][args.target_layer],
                dtype=np.float64,
            )
            oracle_target_covariance = (
                oracle_target_second
                - np.outer(oracle_target_mean, oracle_target_mean)
            )
            true_raw_m21 = np.asarray(
                oracle["M21"][args.target_layer],
                dtype=np.float64,
            )
            true_c21 = connected_m21(
                oracle_target_mean,
                oracle_target_second,
                true_raw_m21,
                np.asarray(oracle["m2"][args.target_layer], dtype=np.float64),
            )
            oracle_pre_mean = np.asarray(oracle["pre_mean"], dtype=np.float64)
            oracle_pre_second = np.asarray(oracle["pre_M11"], dtype=np.float64)
        if args.construction_state == "oracle":
            target_mean = oracle_target_mean
            target_covariance = oracle_target_covariance
            construction_pre_mean = oracle_pre_mean
            construction_pre_second = oracle_pre_second
        else:
            if len(args.checkpoints) != 1:
                raise ValueError(
                    "sample_tail_k2 currently requires exactly one checkpoint"
                )
            target_mean, target_covariance, tail_pre_mean, tail_pre_second = (
                sample_initialized_tail_k2(
                    weights,
                    snapshots[args.checkpoints[0]],
                    args.checkpoints[0],
                    args.target_layer,
                    radius,
                )
            )
            construction_pre_mean = oracle_pre_mean.copy()
            construction_pre_second = oracle_pre_second.copy()
            for layer, value in tail_pre_mean.items():
                construction_pre_mean[layer] = value
                construction_pre_second[layer] = tail_pre_second[layer]
        true_anchor = contraction(left, true_raw_m21, right)

        compact_state = None
        if args.skip_factorized_base:
            if args.defect_scales:
                raise ValueError(
                    "--skip-factorized-base requires an empty --defect-scales"
                )
            if args.transport_source != "oracle_gates":
                raise ValueError(
                    "--skip-factorized-base currently requires oracle_gates"
                )
            target_factorized_c21 = None
            target_factors = None
        elif args.compact_state_dir is not None:
            compact_path = args.compact_state_dir / f"mlp_{index:05d}.npz"
            with np.load(compact_path) as compact:
                compact_state = {
                    key: np.asarray(compact[key])
                    for key in compact.files
                }
            target_factorized_c21 = np.asarray(
                compact_state["target_c21"],
                dtype=np.float64,
            )
            target_factors = None
        else:
            target_path = (
                args.factor_root
                / f"layer{args.target_layer}"
                / f"mlp_{index:05d}.npz"
            )
            with np.load(target_path) as target_state:
                target_factorized_c21 = np.asarray(
                    target_state["c21"],
                    dtype=np.float64,
                )
                target_factors = (
                    load_factors(target_state)
                    if args.transport_source == "inferred_factors"
                    else None
                )

        corrected_c21 = {}
        if target_factorized_c21 is not None:
            corrected_c21["factorized_target"] = target_factorized_c21
        transport_diagnostics = {}
        for checkpoint in args.checkpoints:
            if compact_state is not None:
                column_transport = np.asarray(
                    compact_state[f"checkpoint{checkpoint}_tail_map"],
                    dtype=np.float64,
                )
                inherited_factorized_c21 = np.asarray(
                    compact_state[f"checkpoint{checkpoint}_inherited_c21"],
                    dtype=np.float64,
                )
                checkpoint_rank = int(
                    compact_state[f"checkpoint{checkpoint}_rank"]
                )
                diagnostics = {
                    "transport_norm": float(np.linalg.norm(column_transport)),
                    "transport_condition": float(
                        np.linalg.cond(column_transport)
                    ),
                    "source": "compact_factorized",
                }
            elif not args.skip_factorized_base:
                checkpoint_path = (
                    args.factor_root
                    / f"layer{checkpoint}"
                    / f"mlp_{index:05d}.npz"
                )
                with np.load(checkpoint_path) as checkpoint_state:
                    checkpoint_factors = load_factors(checkpoint_state)
                checkpoint_rank = checkpoint_factors[0].shape[1]
            else:
                checkpoint_rank = 0
            if compact_state is None and args.transport_source == "inferred_factors":
                if target_factors is None:
                    raise AssertionError("target factors not loaded")
                column_transport, diagnostics = infer_transport(
                    checkpoint_factors,
                    target_factors,
                    ridge_fraction=args.transport_ridge,
                )
                transported_checkpoint_factors = tuple(
                    factor[:, :checkpoint_rank]
                    for factor in target_factors
                )
            elif compact_state is None:
                row_transport = expected_gate_transport(
                    weights,
                    checkpoint,
                    args.target_layer,
                    construction_pre_mean,
                    construction_pre_second,
                )
                column_transport = row_transport.T
                if not args.skip_factorized_base:
                    transported_checkpoint_factors = tuple(
                        column_transport @ factor
                        for factor in checkpoint_factors
                    )
                diagnostics = {
                    "transport_norm": float(np.linalg.norm(column_transport)),
                    "transport_condition": float(
                        np.linalg.cond(column_transport)
                    ),
                }
            if compact_state is None and not args.skip_factorized_base:
                inherited_factorized_c21 = factor_c21(
                    transported_checkpoint_factors
                )
            transport_diagnostics[str(checkpoint)] = {
                **diagnostics,
                "factor_rank": checkpoint_rank,
            }
            for basis_label, basis_ids in (
                *cheap_basis_sets.items(),
                ("full", full_basis_ids),
            ):
                inherited_sample_c21 = empirical_transported_c21(
                    snapshots[checkpoint],
                    column_transport,
                    basis_ids,
                )
                for sample_scale in args.sample_only_scales:
                    corrected_c21[
                        f"checkpoint{checkpoint}_{basis_label}"
                        f"_sampleonly_scale{sample_scale:g}"
                    ] = sample_scale * inherited_sample_c21
                if args.defect_scales:
                    defect = inherited_sample_c21 - inherited_factorized_c21
                    for defect_scale in args.defect_scales:
                        corrected_c21[
                            f"checkpoint{checkpoint}_{basis_label}"
                            f"_scale{defect_scale:g}"
                        ] = target_factorized_c21 + defect_scale * defect

        anchors = {
            "oracle": true_anchor,
            "sample": sample_anchor,
        }
        for label, c21 in corrected_c21.items():
            raw_m21 = raw_m21_from_cumulants(
                target_mean,
                target_covariance,
                c21,
            )
            anchors[label] = contraction(left, raw_m21, right)

        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        for label, anchor in anchors.items():
            for control_rank in args.control_ranks:
                if control_rank <= 0 or control_rank > args.rank:
                    raise ValueError(control_rank)
                method_label = (
                    label
                    if args.control_ranks == [args.rank]
                    else f"{label}_r{control_rank}"
                )
                method_to_anchor[method_label] = label
                features = pointwise_features(
                    target_activation,
                    left[:, :control_rank],
                    right[:, :control_rank],
                    anchor[:control_rank],
                    radius,
                )
                prediction, _ = crossfit_grid(
                    features,
                    final,
                    args.folds,
                    [args.ridge],
                )
                method_mses[method_label] = float(
                    np.mean(np.square(prediction[args.ridge] - targets[-1]))
                )
        same_cloud_error = max(
            float(np.linalg.norm(sample_anchor - true_anchor)),
            1e-30,
        )
        anchor_diagnostics = {
            label: {
                "relative_to_same_cloud": float(
                    np.linalg.norm(anchor - true_anchor) / same_cloud_error
                ),
                "error": (anchor - true_anchor).tolist(),
                "shift_from_sample_z_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (anchor - sample_anchor)
                                / np.maximum(sample_anchor_se, 1e-30)
                            )
                        )
                    )
                ),
            }
            for label, anchor in anchors.items()
            if label != "oracle"
        }
        c21_diagnostics = {
            label: {
                "relative_error": float(
                    np.linalg.norm(c21 - true_c21)
                    / max(np.linalg.norm(true_c21), 1e-30)
                ),
                "norm_ratio": float(
                    np.linalg.norm(c21) / max(np.linalg.norm(true_c21), 1e-30)
                ),
            }
            for label, c21 in corrected_c21.items()
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "anchor_diagnostics": anchor_diagnostics,
            "c21_diagnostics": c21_diagnostics,
            "transport_diagnostics": transport_diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index}] "
            + " ".join(
                f"{label}={mse / baseline_mse:.3f}x"
                for label, mse in method_mses.items()
            )
            + f" ({record['seconds']:.1f}s)",
            flush=True,
        )

    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    summary = {}
    for label in labels:
        mse = np.asarray([r["method_mses"][label] for r in records])
        anchor_label = method_to_anchor[label]
        summary[label] = {
            "mse_ratio": float(np.mean(mse) / np.mean(baseline)),
            "wins": int(np.sum(mse < baseline)),
            "worst": float(np.max(mse / baseline)),
            "mean_anchor_error_relative_to_same_cloud": (
                0.0
                if anchor_label == "oracle"
                else float(
                    np.mean(
                        [
                            r["anchor_diagnostics"][anchor_label][
                                "relative_to_same_cloud"
                            ]
                            for r in records
                        ]
                    )
                )
            ),
        }
    output = {
        "protocol": {
            "indices": args.indices,
            "checkpoints": args.checkpoints,
            "target_layer": args.target_layer,
            "rotation_seed": args.rotation_seed,
            "rank": args.rank,
            "control_ranks": args.control_ranks,
            "folds": args.folds,
            "ridge": args.ridge,
            "transport_ridge": args.transport_ridge,
            "transport_source": args.transport_source,
            "cheap_basis_ids": {
                label: basis_ids.tolist()
                for label, basis_ids in cheap_basis_sets.items()
            },
            "defect_scales": args.defect_scales,
            "sample_only_scales": args.sample_only_scales,
            "skip_factorized_base": args.skip_factorized_base,
            "construction_state": args.construction_state,
            "factor_root": str(args.factor_root),
            "compact_state_dir": (
                None
                if args.compact_state_dir is None
                else str(args.compact_state_dir)
            ),
            "oracle_target_marginals": True,
            "uses_target_layer_state_for_correction": False,
            "uses_final_targets_for_construction": False,
        },
        "summary": summary,
        "method_to_anchor": method_to_anchor,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
