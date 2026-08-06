#!/usr/bin/env python3
"""Full computer-assisted verification of strengthened frozen degree-23 certificate."""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp
from verify_oracle_proof_completion import (
    NODES, ORDER, harmonic_dimension, interval_endpoints,
    kernel_maclaurin_jet, mp, normalized_gegenbauer,
    projection_of_monomial, projection_of_polynomial,
)

L = 23
WEIGHT_STRINGS = [
    "0.580027924665828198",
    "1",
    "0.777074267674444097",
    "0.746755561505182408",
    "0.0065394127780017278",
    "0.000076223762757812727",
    "0.00000125400888991954673",
    "0.0000000275983300854912068",
    "0.000000000678764140775296306",
    "0.0000000000186171885693140567",
    "0.000000000000481912464439409887",
    "0.0000000000000257438763325270784",
    "0.000000000000000712501140985209107",
    "0.0000000000000000416613358324680716",
    "0.0000000000000000017927097612805915",
    "0.0000000000000000000909958121383644366",
    "0.00000000000000000000469839764896782971",
    "0.000000000000000000000299076202725570297",
    "0.0000000000000000000000181206660636651611",
    "0.000000000000000000000000454647083680545048",
    "0.0000000000000000000000000281050536570803334",
    "0.00000000000000000000000000639303109407317976",
    "0.000000000000000000000000000516840208522216336",
    "0.0000000000000000000000000000341082132220033759",
]



def main() -> None:
    weights = [sp.Rational(x) for x in WEIGHT_STRINGS]
    dimensions = [harmonic_dimension(l) for l in range(L + 1)]
    weighted_kernel = sum(
        weights[l] * sp.Integer(dimensions[l]) * normalized_gegenbauer(l)
        for l in range(L + 1)
    )
    b = [projection_of_polynomial(weighted_kernel**2, r) for r in range(2 * L + 1)]
    assert all(x > 0 for x in b[1:])

    jet = kernel_maclaurin_jet()
    k_lower = []
    for degree in range(2 * L + 1):
        lower = mp.mpf("0")
        for power in range(degree, ORDER + 1):
            projection = projection_of_monomial(power, degree)
            if projection == 0:
                continue
            lo, _ = interval_endpoints(jet[power])
            lower += lo * mp.mpf(int(projection.p)) / int(projection.q)
        k_lower.append(lower)

    ratios = []
    for r in range(1, 2 * L + 1):
        br = mp.mpf(int(b[r].p)) / int(b[r].q)
        ratios.append((k_lower[r] / br, r))
    gamma, binding_degree = min(ratios)

    items = sorted(
        [(weights[l], dimensions[l], l) for l in range(L + 1)],
        reverse=True,
        key=lambda item: item[0],
    )
    remaining = NODES
    tail_sum = sp.Rational(0)
    tail_square_sum = sp.Rational(0)
    selection = []
    for value, count, degree in items:
        take = min(remaining, count)
        remaining -= take
        tail = count - take
        tail_sum += tail * value
        tail_square_sum += tail * value * value
        selection.append({"degree": degree, "selected": take, "dimension": count})
    assert remaining == 0

    rank_defect = tail_square_sum + tail_sum * tail_sum / sp.Integer(NODES)
    rank_defect_mp = mp.mpf(int(rank_defect.p)) / int(rank_defect.q)
    floor = gamma * rank_defect_mp
    kerdock_mse = mp.mpf("2.433660357543006e-7")

    result = {
        "status": "PASS",
        "weights": WEIGHT_STRINGS,
        "dimensions": dimensions,
        "selection": selection,
        "all_active_coefficients_positive": True,
        "active_coefficients": {str(r): str(b[r]) for r in range(1, 2 * L + 1)},
        "ratios": {str(r): mp.nstr(ratio, 70) for ratio, r in ratios},
        "binding_degree": binding_degree,
        "gamma_lower": mp.nstr(gamma, 70),
        "rank_defect_exact": str(rank_defect),
        "rank_defect": mp.nstr(rank_defect_mp, 70),
        "floor_lower": mp.nstr(floor, 70),
        "fraction_of_kerdock_mse": mp.nstr(floor / kerdock_mse, 70),
        "maximum_improvement_factor": mp.nstr(kerdock_mse / floor, 70),
        "jet_order": ORDER,
    }
    root = Path(__file__).resolve().parents[1]
    output = root / 'results' / 'SYMPY_WEIGHTED_RANK_L23_RECOMPUTED.json'
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "binding_degree": binding_degree,
        "floor_lower": result["floor_lower"],
        "fraction_of_kerdock_mse": result["fraction_of_kerdock_mse"],
        "maximum_improvement_factor": result["maximum_improvement_factor"],
    }, indent=2))


if __name__ == "__main__":
    main()
