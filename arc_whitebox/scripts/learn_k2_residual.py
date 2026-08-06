"""Learn a seed-general K2 residual correction on official WhestBench MLPs.

This is deliberately a *white-box* model rather than an MLP-id lookup:

* the train/validation split is by whole MLP;
* every feature is computed from the supplied weights and a K2 trajectory;
* the official ``mini`` split is used only as a disjoint-seed external test.

The most useful feature family is a one-layer lookback.  K2 treats the final
pre-activation as Gaussian.  We instead draw a small fixed Sobol cloud from
K2's Gaussian approximation to the penultimate pre-activation, apply the last
two ReLU/linear operations, and estimate the final skew/kurtosis.  This costs
O(qmc_n * width^2) rather than sampling all 32 layers.

The script has two stages:

    extract          - cache final-layer features/targets in a compressed npz
    train            - fit final-layer ridge/MLP models
    extract-sequence - cache K2-local features at all 32 layers
    train-sequence   - fit a layerwise free-rollout residual chain

It is easiest to run with the kprop virtualenv.  The extractor adds the parent
research virtualenv's site-packages so that it can import pyarrow.
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
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
KPROP_SRC = ROOT / "vendor" / "mlp_cumulant_propagation" / "src"
ARROW_SITE = ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
sys.path.insert(0, str(KPROP_SRC))
sys.path.insert(0, str(ARROW_SITE))

import pyarrow.dataset as pads  # noqa: E402

from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402


WIDTH = 256
DEPTH = 32
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _tensor(x) -> np.ndarray:
    return x.to_tensor().detach().cpu().numpy()


def _stream_rows(dataset_dir: Path, columns: list[str], limit: int):
    """Yield parquet rows without materializing the multi-gigabyte weight column."""
    dataset = pads.dataset(dataset_dir / "data", format="parquet")
    total = dataset.count_rows()
    if limit:
        total = min(total, limit)
    emitted = 0
    for batch in dataset.scanner(columns=columns, batch_size=4).to_batches():
        for row in batch.to_pylist():
            if emitted >= total:
                return total
            emitted += 1
            yield total, row


def _make_mlp() -> MLP:
    # The extra identity readout leaves challenge layer 31 as a ReLU layer.
    mlp = MLP(
        input_dim=WIDTH,
        hidden_dim=WIDTH,
        output_dim=WIDTH,
        num_layers=DEPTH + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=[2.0] * DEPTH + [1.0],
        b_var=0.0,
        b_mean=0.0,
    )
    mlp.eval()
    with torch.no_grad():
        mlp.Ws[-1].weight.copy_(torch.eye(WIDTH))
    return mlp


def _load_weights(mlp: MLP, weights: np.ndarray) -> None:
    with torch.no_grad():
        for layer in range(DEPTH):
            # Dataset convention is row activations: a @ W.
            mlp.Ws[layer].weight.copy_(
                torch.from_numpy(np.ascontiguousarray(weights[layer].T))
            )


def _fixed_normal_cloud(n: int, seed: int) -> np.ndarray:
    """Scrambled Sobol normal cloud with exact antithetic pairing."""
    if n <= 0 or n % 2:
        raise ValueError("qmc_n must be a positive even integer")
    half = n // 2
    power = int(math.ceil(math.log2(half)))
    u = qmc.Sobol(WIDTH, scramble=True, seed=seed).random_base2(power)[:half]
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    z = ndtri(u).astype(np.float32)
    return np.concatenate([z, -z], axis=0)


def _sample_shape_features(
    pre_mean: np.ndarray,
    pre_cov: np.ndarray,
    final_weight: np.ndarray,
    normal_cloud: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """One-layer Gaussian-lookback features for the final pre-activation."""
    # K2 covariances can acquire tiny negative eigenvalues from float32.
    eigval, eigvec = np.linalg.eigh(
        0.5 * (pre_cov.astype(np.float64) + pre_cov.astype(np.float64).T)
    )
    eigval = np.maximum(eigval, 1e-9)
    root = (eigvec * np.sqrt(eigval)[None, :]).astype(np.float32)
    h_penult = normal_cloud @ root.T + pre_mean[None, :]
    a_penult = np.maximum(h_penult, 0.0)
    h_final = a_penult @ final_weight

    mean = h_final.mean(axis=0, dtype=np.float64)
    centered = h_final.astype(np.float64) - mean
    var = np.mean(centered * centered, axis=0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    skew = np.mean(centered**3, axis=0) / sd**3
    exkurt = np.mean(centered**4, axis=0) / sd**4 - 3.0
    direct = np.maximum(h_final, 0.0).mean(axis=0, dtype=np.float64)
    return (
        mean.astype(np.float32),
        sd.astype(np.float32),
        skew.astype(np.float32),
        exkurt.astype(np.float32),
        direct.astype(np.float32),
    )


def _safe_ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.abs(y), 1e-8)


def extract_features(
    dataset_dir: Path,
    output: Path,
    *,
    limit: int,
    qmc_n: int,
    qmc_seed: int,
    threads: int,
) -> None:
    torch.set_grad_enabled(False)
    if threads:
        torch.set_num_threads(threads)

    columns = ["mlp_id", "mlp_seed", "weights", "final_means"]
    rows = _stream_rows(dataset_dir, columns, limit)
    mlp = _make_mlp()
    z = _fixed_normal_cloud(qmc_n, qmc_seed)

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_base: list[np.ndarray] = []
    ids: list[int] = []
    seeds: list[int] = []
    names: list[str] | None = None
    start = time.perf_counter()

    total = None
    for row_index, (total, row) in enumerate(rows):
        weights = np.asarray(row["weights"], dtype=np.float32)
        target = np.asarray(row["final_means"], dtype=np.float32)
        _load_weights(mlp, weights)

        result = mlp_kprop(
            mlp,
            {1: torch.zeros(WIDTH), 2: torch.eye(WIDTH)},
            k_max=2,
            kind=SIMPLE,
            factor=False,
            use_avg_metric=True,
            output_all=True,
            output_d_max=2,
        )

        final_pre_mean = _tensor(result["pre31"][1]).astype(np.float64)
        final_pre_cov = _tensor(result["pre31"][2]).astype(np.float64)
        final_sd = np.sqrt(np.maximum(np.diag(final_pre_cov), 1e-12))
        t = final_pre_mean / final_sd
        phi = np.exp(-0.5 * t * t) / SQRT_2PI
        base = (final_pre_mean * ndtr(t) + final_sd * phi).astype(np.float32)

        penult_pre_mean = _tensor(result["pre30"][1]).astype(np.float32)
        penult_pre_cov = _tensor(result["pre30"][2]).astype(np.float32)
        prev_mean = _tensor(result["act30"][1]).astype(np.float64)
        prev_cov = _tensor(result["act30"][2]).astype(np.float64)
        w = weights[-1].astype(np.float64)

        q_mean, q_sd, q_skew, q_kurt, q_direct = _sample_shape_features(
            penult_pre_mean, penult_pre_cov, weights[-1], z
        )
        qt = q_mean.astype(np.float64) / np.maximum(q_sd, 1e-8)
        qphi = np.exp(-0.5 * qt * qt) / SQRT_2PI
        edge3 = q_sd * (-qt * qphi) * q_skew / 6.0
        edge4 = q_sd * ((qt * qt - 1.0) * qphi) * q_kurt / 24.0

        # Permutation-equivariant local contractions.  Each output coordinate
        # gets the same feature recipe; source-neuron order is only reduced by
        # sums or W contractions.
        w2 = w * w
        w3 = w2 * w
        w4 = w2 * w2
        cov_w = prev_cov @ w
        diag_cov = np.diag(prev_cov)
        diag_var = diag_cov @ w2
        exact_var = np.sum(w * cov_w, axis=0)

        feature_columns = [
            base,
            final_pre_mean,
            final_sd,
            t,
            phi,
            q_mean,
            q_sd,
            q_skew,
            q_kurt,
            q_direct,
            q_direct - base,
            edge3,
            edge4,
            edge3 + edge4,
            q_mean - final_pre_mean,
            q_sd - final_sd,
            np.sum(w, axis=0),
            np.sum(w2, axis=0),
            np.sum(w3, axis=0),
            np.sum(w4, axis=0),
            np.sum(np.abs(w), axis=0),
            prev_mean @ w,
            (prev_mean**2) @ w,
            (prev_mean**3) @ w,
            prev_mean @ w2,
            (prev_mean**2) @ w2,
            (prev_mean**3) @ w2,
            prev_mean @ w3,
            (prev_mean**2) @ w3,
            prev_mean @ w4,
            diag_var,
            exact_var,
            _safe_ratio(diag_var, exact_var),
            np.sum(cov_w * cov_w, axis=0),
            np.sum(np.abs(cov_w), axis=0),
            diag_cov @ w3,
            diag_cov @ w4,
            np.full(WIDTH, prev_mean.mean()),
            np.full(WIDTH, prev_mean.std()),
            np.full(WIDTH, diag_cov.mean()),
            np.full(WIDTH, np.mean(np.abs(prev_cov - np.diag(diag_cov)))),
        ]
        features = np.stack(feature_columns, axis=1).astype(np.float32)
        if names is None:
            names = [
                "base",
                "pre_mean",
                "pre_sd",
                "pre_t",
                "pre_phi",
                "qmc_mean",
                "qmc_sd",
                "qmc_skew",
                "qmc_exkurt",
                "qmc_direct",
                "qmc_direct_minus_base",
                "edge3",
                "edge4",
                "edge34",
                "qmc_mean_delta",
                "qmc_sd_delta",
                "w_sum",
                "w2_sum",
                "w3_sum",
                "w4_sum",
                "w_abs_sum",
                "wm",
                "wm2",
                "wm3",
                "w2m",
                "w2m2",
                "w2m3",
                "w3m",
                "w3m2",
                "w4m",
                "diag_var",
                "exact_var",
                "diag_exact_ratio",
                "covw_l2",
                "covw_l1",
                "w3_diagcov",
                "w4_diagcov",
                "global_mean_mean",
                "global_mean_sd",
                "global_var_mean",
                "global_offdiag_abs",
            ]

        all_features.append(features)
        all_targets.append(target)
        all_base.append(base)
        ids.append(int(row["mlp_id"]))
        seeds.append(int(row["mlp_seed"]))

        if (row_index + 1) % 10 == 0 or row_index + 1 == total:
            elapsed = time.perf_counter() - start
            mse = np.mean(
                (np.stack(all_base[-10:]) - np.stack(all_targets[-10:])) ** 2
            )
            print(
                f"{row_index + 1:4d}/{total}  "
                f"{elapsed / (row_index + 1):.3f}s/mlp  recent_k2={mse:.3e}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(all_features),
        targets=np.stack(all_targets),
        base=np.stack(all_base),
        mlp_id=np.asarray(ids),
        mlp_seed=np.asarray(seeds, dtype=np.uint64),
        feature_names=np.asarray(names),
        qmc_n=np.asarray(qmc_n),
        qmc_seed=np.asarray(qmc_seed),
    )
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def _local_sequence_features(
    result,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Static K2/local features for every layer of one MLP."""
    features_by_layer = []
    base_by_layer = []
    for layer in range(DEPTH):
        pre_mean = _tensor(result[f"pre{layer}"][1]).astype(np.float64)
        pre_cov = _tensor(result[f"pre{layer}"][2]).astype(np.float64)
        pre_sd = np.sqrt(np.maximum(np.diag(pre_cov), 1e-12))
        t = pre_mean / pre_sd
        phi = np.exp(-0.5 * t * t) / SQRT_2PI
        base = (pre_mean * ndtr(t) + pre_sd * phi).astype(np.float32)
        w = weights[layer].astype(np.float64)
        w2 = w * w
        w3 = w2 * w
        w4 = w2 * w2

        if layer == 0:
            prev_mean = np.zeros(WIDTH, dtype=np.float64)
            prev_cov = np.eye(WIDTH, dtype=np.float64)
        else:
            prev_mean = _tensor(result[f"act{layer - 1}"][1]).astype(np.float64)
            prev_cov = _tensor(result[f"act{layer - 1}"][2]).astype(np.float64)
        diag_cov = np.diag(prev_cov)
        cov_w = prev_cov @ w
        diag_var = diag_cov @ w2
        exact_var = np.sum(w * cov_w, axis=0)

        cols = [
            base,
            pre_mean,
            pre_sd,
            t,
            phi,
            np.sum(w, axis=0),
            np.sum(w2, axis=0),
            np.sum(w3, axis=0),
            np.sum(w4, axis=0),
            np.sum(np.abs(w), axis=0),
            prev_mean @ w,
            (prev_mean**2) @ w,
            (prev_mean**3) @ w,
            prev_mean @ w2,
            (prev_mean**2) @ w2,
            (prev_mean**3) @ w2,
            prev_mean @ w3,
            (prev_mean**2) @ w3,
            prev_mean @ w4,
            diag_var,
            exact_var,
            _safe_ratio(diag_var, exact_var),
            np.sum(cov_w * cov_w, axis=0),
            np.sum(np.abs(cov_w), axis=0),
            diag_cov @ w3,
            diag_cov @ w4,
            np.full(WIDTH, prev_mean.mean()),
            np.full(WIDTH, prev_mean.std()),
            np.full(WIDTH, diag_cov.mean()),
            np.full(WIDTH, np.mean(np.abs(prev_cov - np.diag(diag_cov)))),
            np.full(WIDTH, layer / (DEPTH - 1)),
        ]
        features_by_layer.append(np.stack(cols, axis=1).astype(np.float32))
        base_by_layer.append(base)
    return np.stack(features_by_layer), np.stack(base_by_layer)


def extract_sequence_features(
    dataset_dir: Path,
    output: Path,
    weights_output: Path,
    *,
    limit: int,
    threads: int,
) -> None:
    """Cache all-layer K2 features plus float16 weights for rollout fitting."""
    torch.set_grad_enabled(False)
    if threads:
        torch.set_num_threads(threads)
    columns = ["mlp_id", "mlp_seed", "weights", "all_layer_means"]
    dataset = pads.dataset(dataset_dir / "data", format="parquet")
    total = dataset.count_rows()
    if limit:
        total = min(total, limit)
    rows = _stream_rows(dataset_dir, columns, limit)
    mlp = _make_mlp()

    weights_output.parent.mkdir(parents=True, exist_ok=True)
    weight_map = np.lib.format.open_memmap(
        weights_output,
        mode="w+",
        dtype=np.float16,
        shape=(total, DEPTH, WIDTH, WIDTH),
    )
    all_features = []
    all_targets = []
    all_base = []
    ids = []
    seeds = []
    start = time.perf_counter()
    for row_index, (_, row) in enumerate(rows):
        weights = np.asarray(row["weights"], dtype=np.float32)
        target = np.asarray(row["all_layer_means"], dtype=np.float32)
        weight_map[row_index] = weights.astype(np.float16)
        _load_weights(mlp, weights)
        result = mlp_kprop(
            mlp,
            {1: torch.zeros(WIDTH), 2: torch.eye(WIDTH)},
            k_max=2,
            kind=SIMPLE,
            factor=False,
            use_avg_metric=True,
            output_all=True,
            output_d_max=2,
        )
        features, base = _local_sequence_features(result, weights)
        all_features.append(features)
        all_targets.append(target)
        all_base.append(base)
        ids.append(int(row["mlp_id"]))
        seeds.append(int(row["mlp_seed"]))
        if (row_index + 1) % 10 == 0 or row_index + 1 == total:
            mse = np.mean(
                (np.stack(all_base[-10:]) - np.stack(all_targets[-10:])) ** 2
            )
            elapsed = time.perf_counter() - start
            print(
                f"{row_index + 1:4d}/{total} "
                f"{elapsed / (row_index + 1):.3f}s/mlp all_layer_k2={mse:.3e}",
                flush=True,
            )
    weight_map.flush()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(all_features),
        targets=np.stack(all_targets),
        base=np.stack(all_base),
        mlp_id=np.asarray(ids),
        mlp_seed=np.asarray(seeds, dtype=np.uint64),
        weights_path=np.asarray(str(weights_output.resolve())),
    )
    print(
        f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB) "
        f"and {weights_output} ({weights_output.stat().st_size / 1e9:.2f} GB)"
    )


def _split_arrays(data, train_mlp_count: int):
    x = data["features"].astype(np.float32)
    y = data["targets"].astype(np.float32)
    b = data["base"].astype(np.float32)
    n = len(x)
    ntrain = min(train_mlp_count, n - 1)
    return (x[:ntrain], y[:ntrain], b[:ntrain]), (
        x[ntrain:],
        y[ntrain:],
        b[ntrain:],
    )


def _metrics(pred: np.ndarray, target: np.ndarray, base: np.ndarray) -> dict:
    per_mlp = np.mean((pred - target) ** 2, axis=1)
    base_per_mlp = np.mean((base - target) ** 2, axis=1)
    return {
        "mse": float(per_mlp.mean()),
        "median_mlp_mse": float(np.median(per_mlp)),
        "p90_mlp_mse": float(np.quantile(per_mlp, 0.9)),
        "base_mse": float(base_per_mlp.mean()),
        "gain": float(base_per_mlp.mean() / per_mlp.mean()),
    }


def _expand_features(x: np.ndarray) -> np.ndarray:
    """Small deterministic nonlinear basis for streaming ridge."""
    # The raw feature set already contains physical nonlinearities.  Add
    # bounded powers of the most important standardized shape features.
    core_index = [i for i in [0, 2, 3, 7, 8, 10, 11, 12, 14, 15, 32] if i < x.shape[-1]]
    core = x[..., core_index]
    return np.concatenate(
        [
            x,
            core * core,
            core * core * core,
            np.tanh(core),
            np.ones((*x.shape[:-1], 1), dtype=x.dtype),
        ],
        axis=-1,
    )


def fit_ridge(
    train,
    val,
    external,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    tx, ty, tb = train
    vx, vy, vb = val
    ex, ey, eb = external
    mean = tx.mean(axis=(0, 1), dtype=np.float64)
    std = tx.std(axis=(0, 1), dtype=np.float64)
    std = np.maximum(std, 1e-7)

    def design(x):
        return _expand_features(((x - mean) / std).astype(np.float32)).reshape(
            -1, _expand_features(x[:1, :1]).shape[-1]
        )

    X = design(tx).astype(np.float64)
    r = (ty - tb).reshape(-1).astype(np.float64)
    gram = X.T @ X
    penalty = alpha * np.trace(gram) / len(gram)
    coef = np.linalg.solve(gram + penalty * np.eye(len(gram)), X.T @ r)

    def predict(x, base):
        return base + (design(x) @ coef).reshape(base.shape)

    report = {
        "ridge_alpha": alpha,
        "train": _metrics(predict(tx, tb), ty, tb),
        "validation": _metrics(predict(vx, vb), vy, vb),
        "external": _metrics(predict(ex, eb), ey, eb),
    }
    return coef, report


class ResidualNet(torch.nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_neural(
    train,
    val,
    external,
    *,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    threads: int,
) -> tuple[dict, dict]:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    tx, ty, tb = train
    vx, vy, vb = val
    ex, ey, eb = external

    mean = tx.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = tx.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-7)

    def prep(x):
        return _expand_features(((x - mean) / std).astype(np.float32)).reshape(
            -1, _expand_features(x[:1, :1]).shape[-1]
        )

    train_x = torch.from_numpy(prep(tx))
    # Scale residuals into a numerically comfortable range.
    residual_scale = float(np.std(ty - tb))
    train_r = torch.from_numpy(((ty - tb) / residual_scale).reshape(-1))
    model = ResidualNet(train_x.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    rng = np.random.default_rng(seed)
    count = len(train_x)
    best_state = None
    best_val = float("inf")
    history = []

    def predict(x, base):
        xx = torch.from_numpy(prep(x))
        chunks = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(xx), 65536):
                chunks.append(model(xx[start : start + 65536]).numpy())
        residual = np.concatenate(chunks).reshape(base.shape) * residual_scale
        return base + residual

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(count)
        running = 0.0
        for start in range(0, count, batch_size):
            ix = torch.from_numpy(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(train_x[ix]) - train_r[ix]) ** 2)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(ix)

        val_pred = predict(vx, vb)
        val_mse = float(np.mean((val_pred - vy) ** 2))
        history.append({"epoch": epoch + 1, "loss": running / count, "val": val_mse})
        print(
            f"epoch {epoch + 1:3d} train_scaled={running / count:.5f} "
            f"val={val_mse:.4e}",
            flush=True,
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    report = {
        "hidden": hidden,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "residual_scale": residual_scale,
        "train": _metrics(predict(tx, tb), ty, tb),
        "validation": _metrics(predict(vx, vb), vy, vb),
        "external": _metrics(predict(ex, eb), ey, eb),
        "history": history,
    }
    state = {
        "model": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "residual_scale": residual_scale,
    }
    return state, report


def train_models(args) -> None:
    full = np.load(args.full_cache)
    mini = np.load(args.mini_cache)
    train, val = _split_arrays(full, args.train_mlp_count)
    external = (
        mini["features"].astype(np.float32),
        mini["targets"].astype(np.float32),
        mini["base"].astype(np.float32),
    )

    ridge_coef, ridge_report = fit_ridge(
        train, val, external, alpha=args.ridge_alpha
    )
    print("ridge", json.dumps(ridge_report, indent=2), flush=True)
    neural_state, neural_report = fit_neural(
        train,
        val,
        external,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        threads=args.threads,
    )
    print("neural", json.dumps(neural_report, indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "neural": neural_state,
            "ridge_coef": ridge_coef,
            "ridge_report": ridge_report,
            "neural_report": neural_report,
            "feature_names": full["feature_names"].tolist(),
            "qmc_n": int(full["qmc_n"]),
            "train_mlp_count": args.train_mlp_count,
        },
        args.output,
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps({"ridge": ridge_report, "neural": neural_report}, indent=2)
    )
    print(f"wrote {args.output} and {report_path}")


def _sequence_dynamic_features(
    weights: np.ndarray,
    previous_residual: np.ndarray,
    previous_base: np.ndarray,
) -> np.ndarray:
    """Features that make the learned correction a genuine free rollout."""
    w = weights.astype(np.float32)
    w2 = w * w
    r = previous_residual.astype(np.float32)
    rb = r * previous_base
    cols = [
        r @ w,
        r @ w2,
        rb @ w,
        rb @ w2,
        np.abs(r) @ w,
        np.abs(r) @ w2,
        (r * r) @ w,
        (r * r) @ w2,
        np.full(WIDTH, r.mean(), dtype=np.float32),
        np.full(WIDTH, r.std(), dtype=np.float32),
        np.full(WIDTH, np.mean(np.abs(r)), dtype=np.float32),
    ]
    return np.stack(cols, axis=1)


def _sequence_design_basis(x: np.ndarray) -> np.ndarray:
    """Moderate nonlinear basis; fitted separately at every depth."""
    # Free rollouts can visit a slightly wider feature range than training.
    # Clipping prevents polynomial extrapolation from turning that benign
    # covariate shift into an unstable positive-feedback loop.
    x = np.clip(x, -8.0, 8.0)
    # x is already standardized.  Interactions focus on propagated residual,
    # Gaussian gate and variance diagnostics.
    core_index = [0, 2, 3, 4, 19, 20, 21, 31, 32, 33, 34, 35, 36, 37, 38]
    core_index = [i for i in core_index if i < x.shape[-1]]
    core = x[..., core_index]
    interactions = [
        core,
        core * core,
        np.tanh(core),
    ]
    # Dynamic pre-mean correction times the local ReLU gate is the leading
    # analytic response.  Include it explicitly when the dynamic block exists.
    if x.shape[-1] >= 42:
        gate = x[..., 4:5]
        dyn = x[..., 31:39]
        interactions.extend([gate * dyn, x[..., 3:4] * dyn])
    return np.concatenate(
        [x, *interactions, np.ones((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1
    )


def _load_sequence_cache(path: Path):
    data = np.load(path)
    return {
        "features": data["features"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "base": data["base"].astype(np.float32),
        "weights": np.load(str(data["weights_path"])),
    }


def fit_sequence_ridge(
    full_path: Path,
    mini_path: Path,
    *,
    train_mlp_count: int,
    alpha: float,
) -> tuple[dict, dict]:
    full = _load_sequence_cache(full_path)
    external = _load_sequence_cache(mini_path)
    nfull = len(full["features"])
    ntrain = min(train_mlp_count, nfull - 1)
    groups = {
        "train": {k: v[:ntrain] for k, v in full.items()},
        "validation": {k: v[ntrain:] for k, v in full.items()},
        "external": external,
    }
    for group in groups.values():
        count = len(group["features"])
        group["previous_residual"] = np.zeros((count, WIDTH), dtype=np.float32)
        group["predictions"] = np.empty_like(group["base"])

    layer_models = []
    layer_report = []
    for layer in range(DEPTH):
        designs = {}
        for name, group in groups.items():
            if layer == 0:
                dynamic = np.zeros(
                    (len(group["features"]), WIDTH, 11), dtype=np.float32
                )
                previous_base = np.zeros(
                    (len(group["features"]), WIDTH), dtype=np.float32
                )
            else:
                previous_base = group["base"][:, layer - 1]
                dynamic = np.stack(
                    [
                        _sequence_dynamic_features(w[layer], r, b)
                        for w, r, b in zip(
                            group["weights"],
                            group["previous_residual"],
                            previous_base,
                        )
                    ]
                )
            raw = np.concatenate([group["features"][:, layer], dynamic], axis=-1)
            designs[name] = raw

        train_raw = designs["train"]
        mean = train_raw.mean(axis=(0, 1), dtype=np.float64)
        std = train_raw.std(axis=(0, 1), dtype=np.float64)
        std = np.maximum(std, 1e-7)

        def basis(raw):
            standardized = ((raw - mean) / std).astype(np.float32)
            return _sequence_design_basis(standardized).reshape(
                -1, _sequence_design_basis(standardized[:1, :1]).shape[-1]
            )

        X = basis(train_raw).astype(np.float64)
        target_residual = (
            groups["train"]["targets"][:, layer]
            - groups["train"]["base"][:, layer]
        ).reshape(-1).astype(np.float64)
        residual_bound = max(
            1e-4,
            1.5 * float(np.quantile(np.abs(target_residual), 0.9995)),
        )
        gram = X.T @ X
        penalty = alpha * np.trace(gram) / len(gram)
        coef = np.linalg.solve(gram + penalty * np.eye(len(gram)), X.T @ target_residual)
        layer_models.append(
            {
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "coef": coef.astype(np.float32),
            }
        )

        this_report = {"layer": layer}
        for name, group in groups.items():
            residual = (basis(designs[name]) @ coef).reshape(-1, WIDTH).astype(
                np.float32
            )
            residual = np.clip(residual, -residual_bound, residual_bound)
            prediction = group["base"][:, layer] + residual
            group["previous_residual"] = residual
            group["predictions"][:, layer] = prediction
            this_report[name] = float(
                np.mean((prediction - group["targets"][:, layer]) ** 2)
            )
        layer_report.append(this_report)
        print(
            f"layer {layer:2d} "
            + " ".join(
                f"{name}={this_report[name]:.3e}"
                for name in ("train", "validation", "external")
            ),
            flush=True,
        )

    report = {"alpha": alpha, "layers": layer_report}
    for name, group in groups.items():
        final = group["predictions"][:, -1]
        target = group["targets"][:, -1]
        base = group["base"][:, -1]
        report[name] = _metrics(final, target, base)
        report[name]["all_layer_mse"] = float(
            np.mean((group["predictions"] - group["targets"]) ** 2)
        )
    return {"layers": layer_models}, report


def train_sequence(args) -> None:
    state, report = fit_sequence_ridge(
        args.full_cache,
        args.mini_cache,
        train_mlp_count=args.train_mlp_count,
        alpha=args.ridge_alpha,
    )
    print(json.dumps(report, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "report": report}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("dataset_dir", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument("--qmc-n", type=int, default=2048)
    extract.add_argument("--qmc-seed", type=int, default=7349)
    extract.add_argument("--threads", type=int, default=8)

    extract_sequence = sub.add_parser("extract-sequence")
    extract_sequence.add_argument("dataset_dir", type=Path)
    extract_sequence.add_argument("output", type=Path)
    extract_sequence.add_argument("weights_output", type=Path)
    extract_sequence.add_argument("--limit", type=int, default=0)
    extract_sequence.add_argument("--threads", type=int, default=8)

    train = sub.add_parser("train")
    train.add_argument("full_cache", type=Path)
    train.add_argument("mini_cache", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--train-mlp-count", type=int, default=800)
    train.add_argument("--ridge-alpha", type=float, default=1e-6)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--seed", type=int, default=20260727)
    train.add_argument("--threads", type=int, default=8)

    train_sequence_parser = sub.add_parser("train-sequence")
    train_sequence_parser.add_argument("full_cache", type=Path)
    train_sequence_parser.add_argument("mini_cache", type=Path)
    train_sequence_parser.add_argument("output", type=Path)
    train_sequence_parser.add_argument("--train-mlp-count", type=int, default=800)
    train_sequence_parser.add_argument("--ridge-alpha", type=float, default=1e-5)

    args = parser.parse_args()
    if args.command == "extract":
        extract_features(
            args.dataset_dir,
            args.output,
            limit=args.limit,
            qmc_n=args.qmc_n,
            qmc_seed=args.qmc_seed,
            threads=args.threads,
        )
    elif args.command == "extract-sequence":
        extract_sequence_features(
            args.dataset_dir,
            args.output,
            args.weights_output,
            limit=args.limit,
            threads=args.threads,
        )
    elif args.command == "train":
        train_models(args)
    else:
        train_sequence(args)


if __name__ == "__main__":
    main()
