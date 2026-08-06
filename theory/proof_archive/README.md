# Ancillary verification archive

This directory contains the minimal proof objects for **Limits of Static
Cubature for Deep ReLU Gaussian Expectations**.

## Run

Python 3.11 or later is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/check_package.py
python scripts/run_verification.py
python scripts/check_package.py
```

The final hash check should still pass because all generated verification
records are deterministic.

## Evidence map

- `proof/nonnegative/prove_t16_all_degree.py`: exact integer/rational proof that
  every omitted Gegenbauer degree has negative reduced cost.
- `proof/nonnegative/v5_2/verify_bundle.py`: checks the recovered directed
  degree-five primal-dual records, endpoint separation, one-sided theorem
  metadata, and record hash. This is a consistency replay, not an independent
  interval reconstruction.
- `proof/kerdock/sanity_check_kerdock_risk.py`: independent high-precision,
  non-directed calculation from the exact complete-MUB pair spectrum.
- `proof/signed/verify_signed_near_optimality_certificate_blocktrace_order320.py`:
  exact rational replay of the frozen 146-profile signed witness downstream of
  archived one-sided depth-32 coefficient intervals.
- `proof/signed/verify_inertia_strengthened_frozen_witness.py`: exact rational
  positive-index strengthening, universal nonnegative/signed case split, and
  selected negative-support thresholds.
- `independent_checks/`: separate high-precision formal-series and Hermite
  calculations, with their saved outputs, used as non-rigorous cross-checks.
  They are not part of the default replay because they are substantially slower.

## Trust boundary

The archive exactly checks all integer and rational steps downstream of the
stored one-sided interval inputs. It includes records of the degree-five
interval checks and the clean regeneration report. It does **not** independently
reconstruct the complete depth-32 Gegenbauer coefficient and curvature interval
stack in a second directed-arithmetic implementation. Such a reconstruction,
and human review of the analytic reduction, remain external verification tasks.

Some machine-readable records retain an earlier internal project label because
their hashes are proof inputs. The manuscript and submission filenames use the
paper title.
