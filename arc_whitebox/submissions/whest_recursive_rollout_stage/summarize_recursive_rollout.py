#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("result", type=Path)
args = ap.parse_args()
r = json.loads(args.result.read_text())
s = r["test_corrected"]
print(f"best alpha: {r['best_alpha']:.4g}")
print(f"final mean MSE gain: {s['mean_mse_gain']:.3f}x")
print(f"final relative-variance MSE gain: {s['relative_variance_mse_gain']:.3f}x")
print(f"final sigma MSE gain: {s['sigma_mse_gain']:.3f}x")
print(f"MLPs with improved final mean: {100*s['fraction_mlps_mean_improved']:.1f}%")
print(f"MLPs with improved final variance: {100*s['fraction_mlps_variance_improved']:.1f}%")
print(f"final sigma RMS: {100*s['final_sigma_relative_rms']:.3f}%")
print(f"PSD repairs: {s['psd_repairs']}")
print(f"minimum pre-repair eigenvalue: {s['minimum_pre_repair_eigenvalue']:.4g}")

if (s['mean_mse_gain'] >= 1.25
    and s['fraction_mlps_mean_improved'] >= 0.75
    and s['relative_variance_mse_gain'] >= 1.25):
    print("VERDICT: PROCEED TO FACTORIZED-K3 ROLLOUT")
elif s['mean_mse_gain'] >= 1.05:
    print("VERDICT: BORDERLINE — inspect per-layer drift and tune stabilization")
else:
    print("VERDICT: STOP — one-step covariance gains do not survive rollout")
