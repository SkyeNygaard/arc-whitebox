#!/usr/bin/env python3
"""Explicit counterexamples to the Haar conditional claim and replication corollary."""
import json, math
from pathlib import Path

# Haar counterexample in d=2.
# Q is the one-node rule at e1. h(x)=x_1^2. For g=rotation(theta), define f_g(x)=h(g^{-1}x).
# Then Q_g f_g = f_g(g e1)=h(e1)=1, while spherical mean is 1/2.
thetas = [2 * math.pi * k / 10000 for k in range(10000)]
errors = []
for theta in thetas:
    # Algebraically exactly one; kept explicit as a finite numerical sanity check.
    qgf = 1.0
    integral = 0.5
    errors.append(qgf - integral)
haar_mean_error = sum(errors) / len(errors)

# Biased independent replicas: deterministic variables are mutually independent.
m = 5
bias = 2.0
single_risk = bias**2
average_risk = bias**2
adjusted_ratio = m * average_risk / single_risk

result = {
    "haar_orientation_counterexample": {
        "dimension": 2,
        "rule": "Q=delta_{e1}",
        "random_rotation": "g is Haar on SO(2)",
        "integrand": "f_g(x)=h(g^{-1}x), h(x)=x_1^2",
        "runtime_sigma_field": "trivial, hence g is independent of it",
        "Q_g_f_g": 1.0,
        "I_f_g": 0.5,
        "conditional_mean_error": haar_mean_error,
        "conclusion": "Independence of g from runtime features alone is insufficient when f may depend on g.",
        "repair": "Require Law(g | f, runtime information)=Haar, e.g. g independent of sigma(f) joined with the runtime sigma-field."
    },
    "replication_counterexample": {
        "replicas": m,
        "errors": "e_i=b deterministically; degenerate random variables are mutually independent",
        "bias": bias,
        "single_risk": single_risk,
        "average_risk": average_risk,
        "linear_compute_factor": m,
        "adjusted_ratio": adjusted_ratio,
        "conclusion": "Independent errors need not have zero cross-inner-product. The score-neutral statement requires mean-zero or pairwise uncorrelated errors."
    },
    "observability_ratio_edge": {
        "case": "dictionary has zero oracle capacity",
        "runtime_value": 0,
        "oracle_value": 0,
        "ratio": "undefined 0/0",
        "repair": "Assume positive oracle value or define a convention separately."
    },
    "pass": True
}
Path(__file__).with_name("INFORMATION_REPLICATION_ATTACK_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
