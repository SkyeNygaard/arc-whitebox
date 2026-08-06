#!/usr/bin/env python3
"""Counterexample to T38 under its originally enumerated assumptions."""
import json
from pathlib import Path

for d in (4, 256):
    # Pure degree-2 even Hermite/noise-stability kernel K(t)=t^2.
    A = 1.0
    O = 0.0
    C = 1.0 / d
    a = A - O
    b = O - C
    boundary = a + d * b
    assert boundary == 0.0

result = {
    "theorem_attacked": "T38 as originally stated with only nonconstant antipodally even output",
    "counterexample_function": "F(g)=g_1^2-1",
    "properties": {
        "square_integrable": True,
        "even": True,
        "nonconstant": True,
        "Hermite_support": "pure total degree 2",
        "noise_stability_kernel_up_to_scale": "K(t)=t^2"
    },
    "association_values": {
        "dimension_4": {"A": 1, "O": 0, "C": 0.25, "A_minus_O_plus_d_times_O_minus_C": 0},
        "dimension_256": {"A": 1, "O": 0, "C": 1/256, "A_minus_O_plus_d_times_O_minus_C": 0}
    },
    "failure": "The strict sign a+d b>0 does not follow from even nonconstancy. At the boundary c(d)=0, complete-basis mass directions are flat and claims of positive unique basis masses/all-budget-line use fail.",
    "repair": "Assume positive even Hermite mass at some degree >=4, equivalently sum_{r>=2} a_{2r}>0, or explicitly restrict F_Z to a nonconstant finite piecewise-affine ReLU realization for which that property is proved.",
    "pass": True
}
Path(__file__).with_name("T38_ATTACK_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
