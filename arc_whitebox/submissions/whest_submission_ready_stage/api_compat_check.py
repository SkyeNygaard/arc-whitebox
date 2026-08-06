#!/usr/bin/env python3
"""Fail-fast audit for the installed ARC mlp_kprop revision.

This intentionally checks only the public symbols and behaviors used by the
hybrid evaluator. It gives actionable errors before a multi-hour validation run.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    report: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
    }

    kp = importlib.import_module("mlp_kprop")
    harmonic = importlib.import_module("mlp_kprop.kprop_harmonic")
    wick = importlib.import_module("mlp_kprop.wick")
    factor = importlib.import_module("mlp_kprop.factor_k3")

    # mlp_kprop may be installed as a namespace package, in which case
    # __file__ is None and only __path__ is available.
    if getattr(kp, "__file__", None) is not None:
        report["mlp_kprop_file"] = str(Path(kp.__file__).resolve())
    else:
        report["mlp_kprop_file"] = [str(Path(p).resolve()) for p in kp.__path__]
    required_harmonic = ["SIMPLE", "coerce_input", "linear_kprop", "nonlin_kprop"]
    for name in required_harmonic:
        require(hasattr(harmonic, name), f"mlp_kprop.kprop_harmonic missing {name}")
    require(hasattr(wick, "relu_wick_coef"), "mlp_kprop.wick missing relu_wick_coef")

    report["linear_kprop_signature"] = str(inspect.signature(harmonic.linear_kprop))
    report["nonlin_kprop_signature"] = str(inspect.signature(harmonic.nonlin_kprop))
    report["factor_k3_file"] = str(Path(factor.__file__).resolve())

    dtype = torch.float64
    n = 8
    K = harmonic.coerce_input(
        {1: torch.zeros(n, dtype=dtype), 2: torch.eye(n, dtype=dtype)},
        k_max=3,
        kind=harmonic.SIMPLE,
    )
    W = torch.eye(n, dtype=dtype)
    # linear_kprop only propagates the cumulant orders present in its input, so a
    # Gaussian input (K3 == 0) yields no K3 at the first pre-activation. K3 first
    # appears after the nonlinearity; the second linear step then propagates it.
    # The evaluator reads get_dslice off K_pre[3], i.e. the state checked below.
    Kpre = harmonic.linear_kprop(K, W, k_max=3)
    require(1 in Kpre and 2 in Kpre, "linear_kprop lost the mean/covariance")

    Kpost = harmonic.nonlin_kprop(
        Kpre,
        nonlin_wick_coef=wick.relu_wick_coef,
        k_max=3,
        kind=harmonic.SIMPLE,
        use_pK=True,
        factor=True,
    )
    for order in (1, 2, 3):
        require(order in Kpost, f"nonlin_kprop missing cumulant order {order}")

    Kpre2 = harmonic.linear_kprop(Kpost, W, k_max=3)
    require(3 in Kpre2, "linear_kprop did not propagate K3 to the next pre-activation")
    require(hasattr(Kpre2[3], "get_dslice"), "factorized K3 object has no get_dslice")
    ds = Kpre2[3].get_dslice((2, 1))
    require(tuple(ds.shape) == (n, n), f"get_dslice((2,1)) shape is {tuple(ds.shape)}, expected {(n,n)}")
    require(torch.isfinite(ds).all().item(), "get_dslice returned nonfinite values")
    for order in (1, 2):
        obj = Kpost[order]
        require(hasattr(obj, "core"), f"Kpost[{order}] has no mutable core")
        require(getattr(obj, "r", None) == 0, f"Kpost[{order}] expected r=0, got {getattr(obj,'r',None)}")

    report.update({
        "status": "ok",
        "dslice_shape": list(ds.shape),
        "kpost_types": {str(k): type(v).__name__ for k, v in Kpost.items()},
    })
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
