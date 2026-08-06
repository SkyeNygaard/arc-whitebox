# Static Cubature Diagnostics

This directory contains the compact numerical diagnostics behind the public
discussion of Kerdock/MUB static cubature for the width-256, depth-32 normalized
ReLU limiting kernel.

It is intentionally not a claimed replay of the full computer-assisted theorem.
The proof-critical interval artifacts, all-degree witness inputs, and signed
certificate replay referenced by the external-review materials are not present
in this repository. Do not cite these scripts as independent proof verification.

## Scope

The scripts concern static, network-independent spherical cubature rules at the
fixed 66,048-node budget. They do not establish claims about finite-width,
adaptive, nonlinear, network-dependent, or equal-wall-time estimators.

## Run

Use Python 3.12 with the declared numerical dependencies:

```bash
python -m pip install -r requirements-public.txt
python validate_ceiling.py
```

`spectrum.py` derives a finite-dimension Gegenbauer spectrum,
`design_potentials.py` evaluates the Kerdock/MUB frame potentials, and
`validate_ceiling.py` compares those diagnostics with the documented exposed
Mini-100 baseline. The final comparison is numerical evidence only.

See [`../whestbench/`](../whestbench/) for the release policy, evidence labels,
and the distinction between checked, reported, and external-review claims.
