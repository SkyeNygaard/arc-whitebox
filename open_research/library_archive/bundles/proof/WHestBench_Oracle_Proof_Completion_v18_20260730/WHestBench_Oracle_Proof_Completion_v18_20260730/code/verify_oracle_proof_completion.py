#!/usr/bin/env python3
"""Independent checks for the Oracle proof-completion package.

Trust boundary:
- exact rational harmonic algebra: SymPy
- directed real enclosures: mpmath.iv
- finite-dimensional theorem identities: NumPy
- source empirical summaries: authenticated JSON copied into source_snapshots

This is a computer-assisted audit, not a proof-assistant formalization.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

mp.mp.dps = 90
iv = mp.iv
iv.dps = 90

DIM = 256
NODES = 66_048
DEPTH = 32
ORDER = 47
T = sp.symbols("t")


def interval_point(x: int | float | str):
    return iv.mpf([x, x])


def interval_zero():
    return interval_point(0)


def series_add(a, b):
    return [x + y for x, y in zip(a, b)]


def series_scale(a, c):
    return [x * c for x in a]


def series_mul(a, b):
    out = [interval_zero() for _ in range(ORDER + 1)]
    for n in range(ORDER + 1):
        total = interval_zero()
        for i in range(n + 1):
            total += a[i] * b[n - i]
        out[n] = total
    return out


def series_inv(a):
    out = [interval_zero() for _ in range(ORDER + 1)]
    out[0] = 1 / a[0]
    for n in range(1, ORDER + 1):
        total = interval_zero()
        for i in range(1, n + 1):
            total += a[i] * out[n - i]
        out[n] = -total / a[0]
    return out


def series_sqrt(a):
    out = [interval_zero() for _ in range(ORDER + 1)]
    out[0] = iv.sqrt(a[0])
    for n in range(1, ORDER + 1):
        total = interval_zero()
        for i in range(1, n):
            total += out[i] * out[n - i]
        out[n] = (a[n] - total) / (2 * out[0])
    return out


def series_derivative(a):
    return [(n + 1) * a[n + 1] for n in range(ORDER)] + [interval_zero()]


def series_integral(a, constant):
    out = [interval_zero() for _ in range(ORDER + 1)]
    out[0] = constant
    for n in range(1, ORDER + 1):
        out[n] = a[n - 1] / n
    return out


def compose_relu_dual(p):
    one = [interval_zero() for _ in range(ORDER + 1)]
    one[0] = interval_point(1)
    p2 = series_mul(p, p)
    one_minus_p2 = [one[i] - p2[i] for i in range(ORDER + 1)]
    root = series_sqrt(one_minus_p2)

    # d/dt acos(p(t)) = -p'(t)/sqrt(1-p(t)^2).
    acos_derivative = series_scale(
        series_mul(series_derivative(p), series_inv(root)), -1
    )
    acos_constant = iv.atan2(iv.sqrt(1 - p[0] * p[0]), p[0])
    acos_series = series_integral(acos_derivative, acos_constant)

    pi_minus_acos = [-x for x in acos_series]
    pi_minus_acos[0] += iv.pi
    return series_scale(
        series_add(root, series_mul(pi_minus_acos, p)), 1 / iv.pi
    )


def interval_endpoints(x) -> tuple[mp.mpf, mp.mpf]:
    text = str(x)
    comma = text.index(",")
    return mp.mpf(text[1:comma]), mp.mpf(text[comma + 1 : -1])


def kernel_maclaurin_jet():
    jet = [interval_zero() for _ in range(ORDER + 1)]
    jet[1] = interval_point(1)
    for _ in range(DEPTH):
        jet = compose_relu_dual(jet)
    return jet


def spherical_moment(power: int) -> sp.Rational:
    if power % 2:
        return sp.Rational(0)
    m = power // 2
    numerator = sp.Integer(1) if m == 0 else sp.factorial2(2 * m - 1)
    denominator = sp.Integer(1)
    for j in range(m):
        denominator *= DIM + 2 * j
    return sp.Rational(numerator, denominator)


def harmonic_dimension(degree: int) -> int:
    if degree == 0:
        return 1
    if degree == 1:
        return DIM
    return math.comb(DIM + degree - 1, degree) - math.comb(
        DIM + degree - 3, degree - 2
    )


def normalized_gegenbauer(degree: int):
    poly = sp.gegenbauer(degree, sp.Rational(DIM - 2, 2), T)
    return sp.expand(poly / poly.subs(T, 1))


def spherical_expectation(poly) -> sp.Rational:
    return sum(
        coefficient * spherical_moment(monomial[0])
        for monomial, coefficient in sp.Poly(sp.expand(poly), T).terms()
    )


def projection_of_monomial(power: int, degree: int) -> sp.Rational:
    g = normalized_gegenbauer(degree)
    return sp.factor(
        spherical_expectation(T**power * g) / spherical_expectation(g**2)
    )


def projection_of_polynomial(poly, degree: int) -> sp.Rational:
    g = normalized_gegenbauer(degree)
    return sp.factor(
        spherical_expectation(poly * g) / spherical_expectation(g**2)
    )


# Certified T16 intervals copied from the completed proof package.
H_INTERVALS = [
    ("0.97472997513094444136665930858028707859238690682343487472283238278005349338757190", "0.97472997513094444136665930858028707859238690682343487472283238278005350661242810"),
    ("0.0027964730615411841661658602352601821301693853633433268680467387544975333875719043", "0.0027964730615411841661658602352601821301693853633433268680467387544975466124280957"),
    ("0.0024362952737152224244706806097631082956725787352544274932020326217268633875719043", "0.0024362952737152224244706806097631082956725787352544274932020326217268766124280957"),
    ("0.0018037348551971006089123342400015767220307118987410296501926650616942633875719043", "0.0018037348551971006089123342400015767220307118987410296501926650616942766124280957"),
    ("0.0010317284867674261481582137477767852671420383283799842341609475791693633875719043", "0.0010317284867674261481582137477767852671420383283799842341609475791693766124280957"),
    ("0.00017989892346364458549448698909864663853047158683039399322157885175165338757190436", "0.00017989892346364458549448698909864663853047158683039399322157885175166661242809564"),
]


def signed_floor_certificate() -> dict[str, Any]:
    jet = kernel_maclaurin_jet()
    k_lower: list[mp.mpf] = []
    coefficient_rows: list[dict[str, Any]] = []
    for degree in range(7):
        lower = mp.mpf("0")
        terms = []
        for power in range(degree, ORDER + 1):
            projection = projection_of_monomial(power, degree)
            if projection == 0:
                continue
            lo, _ = interval_endpoints(jet[power])
            contribution = lo * mp.mpf(int(projection.p)) / int(projection.q)
            lower += contribution
            terms.append(
                {
                    "power": power,
                    "projection": str(projection),
                    "lower_contribution": mp.nstr(contribution, 45),
                }
            )
        k_lower.append(lower)
        row: dict[str, Any] = {
            "degree": degree,
            "K_coefficient_lower": mp.nstr(lower, 70),
            "terms": terms,
        }
        if degree <= 5:
            h_upper = mp.mpf(H_INTERVALS[degree][1])
            margin = lower - h_upper
            row.update(
                {
                    "h_coefficient_upper": str(h_upper),
                    "residual_margin_lower": mp.nstr(margin, 70),
                    "residual_margin_positive": bool(margin > 0),
                }
            )
        coefficient_rows.append(row)

    degrees = range(4)
    reproducing_kernel = sum(
        sp.Integer(harmonic_dimension(j)) * normalized_gegenbauer(j) for j in degrees
    )
    squared_coefficients = [
        projection_of_polynomial(reproducing_kernel**2, degree)
        for degree in range(7)
    ]

    # The original exploratory verifier displayed only degrees 2,4,6.
    # The proof obligation is all positive nonconstant coefficients, 1..6.
    ratios = []
    for degree in range(1, 7):
        b = squared_coefficients[degree]
        if b > 0:
            ratio = k_lower[degree] / (mp.mpf(int(b.p)) / int(b.q))
            ratios.append((ratio, degree))
    gamma, binding_degree = min(ratios)

    dimension = sum(harmonic_dimension(j) for j in degrees)
    rank_defect = mp.mpf(dimension) * dimension / NODES - dimension
    lower_bound = gamma * rank_defect
    kerdock_mse = mp.mpf("2.433660357543006e-7")

    # Exhaust all nonempty degree subsets of {0,1,2,3}. Full <=3 is best.
    subset_scan = []
    for size in range(1, 5):
        for subset in itertools.combinations(range(4), size):
            kernel = sum(
                sp.Integer(harmonic_dimension(j)) * normalized_gegenbauer(j)
                for j in subset
            )
            bvals = [projection_of_polynomial(kernel**2, degree) for degree in range(7)]
            local_ratios = []
            for degree in range(1, 7):
                if bvals[degree] > 0:
                    local_ratios.append(
                        (
                            k_lower[degree]
                            / (mp.mpf(int(bvals[degree].p)) / int(bvals[degree].q)),
                            degree,
                        )
                    )
            if not local_ratios:
                continue
            local_gamma, local_binding = min(local_ratios)
            local_dimension = sum(harmonic_dimension(j) for j in subset)
            local_defect = max(
                mp.mpf("0"),
                mp.mpf(local_dimension) * local_dimension / NODES - local_dimension,
            )
            subset_scan.append(
                {
                    "degrees": list(subset),
                    "dimension": local_dimension,
                    "binding_degree": local_binding,
                    "lower_bound": mp.nstr(local_gamma * local_defect, 55),
                }
            )

    return {
        "settings": {
            "dimension": DIM,
            "node_budget": NODES,
            "depth": DEPTH,
            "jet_order": ORDER,
            "interval_digits": 90,
        },
        "coefficient_rows": coefficient_rows,
        "residual_positive_definite": all(
            row.get("residual_margin_positive", True) for row in coefficient_rows
        ),
        "rank_feature_space": {
            "degrees": list(degrees),
            "dimension": dimension,
            "squared_kernel_coefficients": {
                str(i): str(value) for i, value in enumerate(squared_coefficients)
            },
            "all_active_degree_ratios": {
                str(degree): mp.nstr(ratio, 70) for ratio, degree in ratios
            },
            "binding_degree": binding_degree,
            "gamma_lower": mp.nstr(gamma, 70),
            "rank_defect": mp.nstr(rank_defect, 70),
            "signed_rule_mse_lower_bound": mp.nstr(lower_bound, 70),
            "fraction_of_kerdock_mse": mp.nstr(lower_bound / kerdock_mse, 60),
            "maximum_improvement_factor": mp.nstr(kerdock_mse / lower_bound, 60),
        },
        "subset_scan": sorted(
            subset_scan, key=lambda row: mp.mpf(row["lower_bound"]), reverse=True
        ),
    }


def information_checks() -> dict[str, Any]:
    grid = np.linspace(-0.999999, 0.999999, 200_001)
    phi = 0.5 * (1 + grid) * np.log1p(grid) + 0.5 * (1 - grid) * np.log1p(-grid)
    gap = phi - 0.5 * grid**2

    rng = np.random.default_rng(20260730)
    # Random finite channels represented by posterior means m(X).
    posterior_means = rng.uniform(-1, 1, size=10_000)
    info = np.mean(
        0.5 * (1 + posterior_means) * np.log1p(posterior_means)
        + 0.5 * (1 - posterior_means) * np.log1p(-posterior_means)
    )
    value = np.mean(posterior_means**2)
    return {
        "minimum_phi_minus_m2_over_2_on_grid": float(np.min(gap)),
        "random_channel_value": float(value),
        "random_channel_twice_information": float(2 * info),
        "inequality_pass": bool(value <= 2 * info + 1e-14),
    }


def symmetry_defect_checks() -> dict[str, Any]:
    rng = np.random.default_rng(45)
    n = 40
    h = 7
    # Involutive index permutation and involutive unitary representation.
    tau = np.arange(n).reshape(-1, 2)[:, ::-1].reshape(-1)
    signs = rng.choice([-1.0, 1.0], size=h)
    u = np.diag(signs)
    e = rng.normal(size=(n, h))
    c = rng.normal(size=(n, h))
    e_tilde = e[tau] @ u
    c_tilde = c[tau] @ u
    lhs = 2 * np.mean(np.sum(e * c, axis=1))
    rhs = np.mean(np.sum((e + e_tilde) * c, axis=1)) + np.mean(
        np.sum(e_tilde * (c_tilde - c), axis=1)
    )

    # Exact anti-invariant error and invariant correction.
    exact_e = np.zeros((n, h))
    exact_c = np.zeros((n, h))
    for i in range(0, n, 2):
        exact_e[i] = rng.normal(size=h)
        exact_e[i + 1] = -u @ exact_e[i]
        exact_c[i] = rng.normal(size=h)
        exact_c[i + 1] = u @ exact_c[i]
    exact_alignment = np.mean(np.sum(exact_e * exact_c, axis=1))

    e_norm = math.sqrt(np.mean(np.sum(e * e, axis=1)))
    c_norm = math.sqrt(np.mean(np.sum(c * c, axis=1)))
    delta_e = math.sqrt(np.mean(np.sum((e + e_tilde) ** 2, axis=1))) / e_norm
    delta_c = math.sqrt(np.mean(np.sum((c_tilde - c) ** 2, axis=1))) / c_norm
    correlation = abs(np.mean(np.sum(e * c, axis=1))) / (e_norm * c_norm)
    bound = 0.5 * (delta_e + delta_c)

    return {
        "identity_absolute_error": float(abs(lhs - rhs)),
        "exact_anti_invariant_alignment": float(exact_alignment),
        "normalized_alignment": float(correlation),
        "defect_bound": float(bound),
        "bound_pass": bool(correlation <= bound + 1e-12),
        "maximum_fractional_gain_bound": float(min(1.0, bound**2)),
    }


def gauge_checks() -> dict[str, Any]:
    rng = np.random.default_rng(46)
    m = 5
    vector = rng.normal(size=m)
    group = np.array(list(itertools.product([-1.0, 1.0], repeat=m)))
    orbit = group * vector
    projection = orbit.mean(axis=0)

    target = 1.7
    predictions = np.linspace(-3, 3, 20_001)
    paired_worst = np.maximum((predictions - target) ** 2, (predictions + target) ** 2)
    return {
        "sign_group_size": int(len(group)),
        "invariant_projection_norm": float(np.linalg.norm(projection)),
        "two_point_minimax_numeric": float(np.min(paired_worst)),
        "two_point_minimax_exact": target**2,
        "checks_pass": bool(
            np.linalg.norm(projection) < 1e-14
            and abs(np.min(paired_worst) - target**2) < 1e-3
        ),
    }


def coherence_checks(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text())
    result: dict[str, Any] = {}
    for split in ("development", "validation", "confirmation"):
        cosine = np.asarray(source[split]["increment_cosine"], dtype=float)
        fractions = np.asarray(source[split]["increment_energy_fraction"], dtype=float)
        gram = cosine * np.sqrt(fractions[:, None] * fractions[None, :])
        eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0)
        effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        cumulative_ratio = float(np.ones(len(fractions)) @ gram @ np.ones(len(fractions)))
        result[split] = {
            "depths": source[split]["depths"],
            "eigenvalues": eigenvalues.tolist(),
            "effective_rank_of_pooled_increment_gram": effective_rank,
            "cumulative_energy_over_sum_increment_energy": cumulative_ratio,
            "cross_term_fraction": cumulative_ratio - 1.0,
            "last_two_checkpoint_energy_fraction": float(fractions[-2:].sum()),
            "max_abs_off_diagonal_cosine": float(source[split]["max_abs_offdiag"]),
        }

    # Counterexample: pooled orthogonality does not imply within-case multidimensionality.
    # Each case is scalar/one-dimensional. Across cases the two increments have sign patterns
    # [1,1] and [1,-1], so their concatenated cosine is zero.
    first = np.array([1.0, 1.0])
    second = np.array([1.0, -1.0])
    pooled_cosine = float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))
    result["counterexample"] = {
        "pooled_increment_1": first.tolist(),
        "pooled_increment_2": second.tolist(),
        "pooled_cosine": pooled_cosine,
        "within_each_case_output_dimension": 1,
        "implication": "Low pooled cosine can be caused entirely by cross-case sign heterogeneity.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    signed = signed_floor_certificate()
    information = information_checks()
    symmetry = symmetry_defect_checks()
    gauge = gauge_checks()
    coherence = coherence_checks(root / "source_snapshots" / "COHERENCE_RESULTS.json")

    (results / "signed_floor_order47.json").write_text(json.dumps(signed, indent=2) + "\n")
    (results / "coherence_diagnostics.json").write_text(json.dumps(coherence, indent=2) + "\n")
    overall = {
        "status": "PASS",
        "signed_floor": {
            "residual_positive_definite": signed["residual_positive_definite"],
            **signed["rank_feature_space"],
        },
        "information": information,
        "symmetry_defect": symmetry,
        "gauge": gauge,
        "coherence": coherence,
        "scope": [
            "Static/network-independent mass-one signed rules only for the rank floor.",
            "Abstract phase models only for information bounds.",
            "No actual WHestBench measure-preserving phase symmetry is asserted.",
            "Pooled coherence does not establish within-network component rank.",
        ],
    }
    assert signed["residual_positive_definite"]
    assert signed["rank_feature_space"]["binding_degree"] == 6
    assert information["inequality_pass"]
    assert symmetry["identity_absolute_error"] < 1e-12
    assert abs(symmetry["exact_anti_invariant_alignment"]) < 1e-12
    assert symmetry["bound_pass"]
    assert gauge["checks_pass"]
    (results / "verification.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "signed_floor": signed["rank_feature_space"]["signed_rule_mse_lower_bound"],
        "improvement_cap": signed["rank_feature_space"]["maximum_improvement_factor"],
        "coherence_effective_ranks": {
            key: coherence[key]["effective_rank_of_pooled_increment_gram"]
            for key in ("development", "validation", "confirmation")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
