from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baselines import fit_anchor_shrink, invariant_features, ridge_fit, ridge_predict
from .contracts import load_bundle
from .cost import estimate_dws_flops, effective_compute_b
from .edge_dws import EdgeStateDWS
from .metrics import evaluate


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_or_none(x: np.ndarray | None, idx: np.ndarray, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    return torch.from_numpy(np.asarray(x[idx], dtype=np.float32)).to(device)


def predict_model(model: EdgeStateDWS, arrays: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    corr, scale, conf = [], [], []
    with torch.no_grad():
        for i in idx:
            w = torch.from_numpy(arrays["weights"][i:i+1].astype(np.float32, copy=False)).to(device)
            no = torch.from_numpy(arrays["node_observables"][i:i+1].astype(np.float32, copy=False)).to(device) if "node_observables" in arrays else None
            lo = torch.from_numpy(arrays["layer_observables"][i:i+1].astype(np.float32, copy=False)).to(device) if "layer_observables" in arrays else None
            out = model(w, no, lo)
            corr.append(out.correction.cpu().numpy()[0])
            scale.append(float(out.scale.cpu()[0]))
            conf.append(float(out.confidence.cpu()[0]))
    return np.asarray(corr), np.asarray(scale), np.asarray(conf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    seed_all(int(cfg["seed"]))
    bundle = load_bundle(args.data, args.manifest, args.splits)
    a = bundle.arrays
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    node_obs_dim = a["node_observables"].shape[-1] if "node_observables" in a else 0
    layer_obs_dim = a["layer_observables"].shape[-1] if "layer_observables" in a else 0
    model = EdgeStateDWS(
        depth=32,
        label_dim=bundle.label_dim,
        node_obs_dim=node_obs_dim,
        layer_obs_dim=layer_obs_dim,
        edge_channels=cfg["model"]["edge_channels"],
        node_channels=cfg["model"]["node_channels"],
        token_channels=cfg["model"]["token_channels"],
        passes=cfg["model"]["passes"],
        transformer_heads=cfg["model"]["transformer_heads"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    train_idx = bundle.splits["train"]
    calib_idx = bundle.splits["calibration"]
    rng = np.random.default_rng(cfg["seed"])
    best_state, best_cal = None, float("inf")
    history = []
    t0 = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        opt.zero_grad(set_to_none=True)
        accum = int(cfg["training"].get("gradient_accumulation", 1))
        for step, i in enumerate(order):
            w = torch.from_numpy(a["weights"][i:i+1].astype(np.float32, copy=False)).to(device)
            no = torch.from_numpy(a["node_observables"][i:i+1].astype(np.float32, copy=False)).to(device) if "node_observables" in a else None
            lo = torch.from_numpy(a["layer_observables"][i:i+1].astype(np.float32, copy=False)).to(device) if "layer_observables" in a else None
            e0 = torch.from_numpy(a["baseline_error"][i:i+1].astype(np.float32, copy=False)).to(device)
            j = torch.from_numpy(a["replay_jacobian"][i:i+1].astype(np.float32, copy=False)).to(device)
            anchor = torch.from_numpy(a["anchor_coeffs"][i:i+1].astype(np.float32, copy=False)).to(device)
            target = torch.from_numpy(a["target_coeffs"][i:i+1].astype(np.float32, copy=False)).to(device)
            out = model(w, no, lo)
            total = anchor + out.correction
            err = e0 + torch.einsum("bod,bd->bo", j, total)
            replay_loss = err.square().mean()
            target_resid = target - anchor
            aux = (out.correction - target_resid).square().mean()
            target_benefit = ((e0 + torch.einsum("bod,bd->bo", j, target)).square().mean(dim=1) < e0.square().mean(dim=1)).float()
            conf_loss = nn.functional.binary_cross_entropy(out.confidence, target_benefit)
            loss = replay_loss + cfg["training"]["aux_coefficient_weight"] * aux + cfg["training"]["confidence_weight"] * conf_loss
            (loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        pred, _, conf = predict_model(model, a, calib_idx, device)
        coeff = a["anchor_coeffs"][calib_idx] + pred
        cal = evaluate(a, calib_idx, coeff, conf, cfg["cost"]["baseline_effective_compute_B"], cfg["cost"]["baseline_effective_compute_B"])
        cal_loss = cal["candidate_raw_mse"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "calibration_raw_mse": cal_loss})
        if cal_loss < best_cal:
            best_cal = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    # Global residual shrink selected only on calibration.
    pred_cal, _, conf_cal = predict_model(model, a, calib_idx, device)
    grid = np.linspace(0.0, 1.5, 301)
    best_alpha, best = 0.0, float("inf")
    for alpha in grid:
        coeff = a["anchor_coeffs"][calib_idx] + alpha * pred_cal
        err = a["baseline_error"][calib_idx] + np.einsum("nod,nd->no", a["replay_jacobian"][calib_idx], coeff)
        loss = float(np.mean(err * err))
        if loss < best:
            best, best_alpha = loss, float(alpha)

    inference_flops = estimate_dws_flops(
        32, 256, cfg["model"]["edge_channels"], cfg["model"]["node_channels"],
        cfg["model"]["token_channels"], cfg["model"]["passes"], bundle.label_dim,
    )
    candidate_compute = effective_compute_b(
        cfg["cost"]["baseline_effective_compute_B"], cfg["cost"]["anchor_extra_compute_B"],
        inference_flops, cfg["cost"].get("replay_extra_compute_B", 0.0),
    )
    baseline_compute = cfg["cost"]["baseline_effective_compute_B"]

    # Controls fit without test access.
    alpha_anchor = fit_anchor_shrink(a["anchor_coeffs"][calib_idx], a["baseline_error"][calib_idx], a["replay_jacobian"][calib_idx])
    x_all = invariant_features(a["weights"], a.get("layer_observables"))
    y_resid = a["target_coeffs"] - a["anchor_coeffs"]
    ridge_models = []
    for ridge in cfg["ridge_grid"]:
        rm = ridge_fit(x_all[train_idx], y_resid[train_idx], ridge)
        pc = ridge_predict(rm, x_all[calib_idx])
        coeff = a["anchor_coeffs"][calib_idx] + pc
        err = a["baseline_error"][calib_idx] + np.einsum("nod,nd->no", a["replay_jacobian"][calib_idx], coeff)
        ridge_models.append((float(np.mean(err * err)), rm))
    ridge_model = min(ridge_models, key=lambda x: x[0])[1]

    report = {
        "status": "completed",
        "seed": cfg["seed"],
        "device": str(device),
        "split_examples": {k: int(len(v)) for k, v in bundle.splits.items()},
        "split_base_networks": {k: int(len({str(x) for x in a["base_network_id"][v]})) for k, v in bundle.splits.items()},
        "label_dim": bundle.label_dim,
        "params": int(sum(p.numel() for p in model.parameters())),
        "inference_flops": inference_flops,
        "inference_effective_compute_B": inference_flops / 1e9,
        "candidate_effective_compute_B": candidate_compute,
        "calibrated_model_residual_alpha": best_alpha,
        "constant_anchor_shrink_alpha": alpha_anchor,
        "history": history,
        "runtime_seconds": time.time() - t0,
        "splits": {},
    }
    for split_name in ("validation", "test"):
        idx = bundle.splits[split_name]
        pred, scale, conf = predict_model(model, a, idx, device)
        coeff_model = a["anchor_coeffs"][idx] + best_alpha * pred
        coeff_anchor = a["anchor_coeffs"][idx]
        coeff_const = alpha_anchor * a["anchor_coeffs"][idx]
        coeff_ridge = a["anchor_coeffs"][idx] + ridge_predict(ridge_model, x_all[idx])
        report["splits"][split_name] = {
            "anchor_only": evaluate(a, idx, coeff_anchor, np.ones(len(idx)), baseline_compute, baseline_compute + cfg["cost"]["anchor_extra_compute_B"]),
            "constant_shrinkage": evaluate(a, idx, coeff_const, np.ones(len(idx)), baseline_compute, baseline_compute + cfg["cost"]["anchor_extra_compute_B"]),
            "invariant_ridge": evaluate(a, idx, coeff_ridge, np.ones(len(idx)), baseline_compute, baseline_compute + cfg["cost"]["anchor_extra_compute_B"]),
            "edge_dws": evaluate(a, idx, coeff_model, conf, baseline_compute, candidate_compute),
            "prediction_scale_mean": float(scale.mean()),
            "prediction_confidence_mean": float(conf.mean()),
        }
    test = report["splits"]["test"]["edge_dws"]
    report["gate"] = {
        "raw_gain_ge_1_15": test["raw_gain_baseline_over_candidate"] >= 1.15,
        "adjusted_ci_excludes_no_gain": test["adjusted_gain_group_bootstrap_ci95"][0] > 1.0,
        "worst_le_1_10": test["worst_candidate_over_baseline"] <= 1.10,
        "inference_cost_repaid": test["adjusted_gain_baseline_over_candidate"] > 1.0,
    }
    report["gate"]["pass"] = all(report["gate"].values())
    torch.save({"state_dict": best_state, "config": cfg, "alpha": best_alpha}, args.out / "model.pt")
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
