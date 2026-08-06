# Phase 1 submission #320802 — Kerdock/MUB 5-design with tracked Winograd propagation

This directory holds the estimator behind AIcrowd Phase 1 submission
**#320802** (`skye_nygaard`, ARC White-Box Estimation Challenge 2026),
together with the frozen archive and the two local result bundles used as
development evidence.

## Graded result

Read from the AIcrowd submission page for #320802 on 2026-08-03.

| field | value |
|---|---|
| submission id | 320802 |
| status | GRADED — "Graded successfully" |
| round | Phase 1 |
| submitted | 2026-07-29 13:03 |
| public MLPs scored | 50 / 50 |
| failures | 0 |
| adjusted score (public split) | `1.55e-7` |
| final-layer MSE (public split) | `2.416e-7` |
| all-layers MSE (mean) | `0.7437` |
| best public MLP | `3.62e-8` (dylan-meyer) |
| worst public MLP | `5.11e-7` (angela-walker) |
| IQR p25–p75 | `9.81e-8` – `1.94e-7` |
| mean effective compute | `1.75e11` of a `2.72e11` budget (64.19%) |
| vs. Monte Carlo reference `6.47e-7` | 4.2× |

The 50-MLP private split is sealed until Phase 2 close, so the final rank is
not yet determined. These are public-split numbers only.

## Archive

`submission.tar.gz` — SHA-256
`77be0e8865b2aeee6c6c16314cac4d38496efefed6b2b758f75bc3033bb6b7bc`

It passes the official CLI:

```bash
whest validate-package submission.tar.gz
```

Member hashes:

| file | sha256 |
|---|---|
| `estimator.py` | `f1e32ce44fe43b53eba3f70f9cf6383da588ec1bbb3d82c047edbc916a98d8df` |
| `fast_matmul.py` | `fb1b93cb625b66ce5f26220ea3b6b685dbb9887d50f8756cafa9426577d45085` |
| `kerdock_mub5_seed3.npz` | `58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad` |

### Provenance boundary — read this before citing the archive

This is the **frozen re-package** of the production estimator, cut on
2026-07-30 as `production_baseline.tar.gz` inside the T0 grader-instrumentation
bundle and described there as "T0.3 frozen production partial-tree baseline
source". AIcrowd holds the authoritative tarball that was actually uploaded on
2026-07-29; that upload is the object of record for the prize.

The identification rests on the recorded operating point rather than on a
byte comparison with the uploaded file. `T0.../RESULTS.json` records the
then-current position as adjusted `1.55e-07`, raw MSE `2.416e-07`, effective
compute `1.745e11`, compute multiplier `0.64154` — matching the graded
#320802 page field for field, and matching no other local package.

One packaging defect is on the record and is not hidden here: the earlier
`kerdock_mub5_winograd_tree` archive shipped a `manifest.json` whose declared
`estimator.py` hash disagreed with the file actually archived
(`7c3fcc2ac542bda41ab568e62428ec75b7edec6f146d61a036c0710d9ee49694`). The
`manifest_actual.json` in the T0 bundle records that mismatch. The archive in
*this* directory is internally consistent: its manifest hashes match its
members, and `whest validate-package` passes.

## What the estimator does

Width-256, depth-32 networks only; anything else returns zeros.

1. **Layer 0** is closed-form, not estimated: under the isotropic Gaussian
   input the post-ReLU mean of unit *j* is the half-normal mean
   `||w_j|| / sqrt(2*pi)`.
2. **The design.** A static, network-independent spherical 5-design of 66,048
   points on the sphere in dimension 256 — 128 Kerdock/maximal-real-MUB bases
   plus the coordinate basis, every direction carried with its antipode:
   `128 x 256 x 2 = 65,536` plus `1 x 256 x 2 = 512`. The design is frozen in
   `kerdock_mub5_seed3.npz` (rotation seed 3) and reused for every network.
3. **First propagation** applies the design to `W_0` through a Walsh–Hadamard
   transform over the chirp structure (8 butterfly stages) rather than an
   explicit 66,048 x 256 by 256 x 256 product.
4. **Hidden layers 1–30** propagate the full 66,048-row activation through each
   weight with a tracked depth-5 Strassen–Winograd kernel
   (`winograd_hybrid_p3_d5_partial_tree`): `7^5 = 16,807` products where the
   conventional recursion charges `32^3 = 32,768`, a 1.9497× reduction in
   charged multiplies. The first three levels are carried as tensor axes; the
   deepest two levels keep sixteen decoded quadrants as a small Python tree and
   assemble once.
5. **Layer 31** is evaluated in 2,048-row chunks, each chunk reduced
   immediately into a float64 accumulator ("Path 6"), avoiding a full
   66,048 x 256 float64 materialization.
6. **Layers 1–30 are returned as zeros.** Only layer 0 and layer 31 carry
   estimates. This is a deliberate choice against the scored objective
   (`adjusted_final_layer_score`) and it is why all-layers MSE sits near 0.74
   while final-layer MSE is ~2.4e-7.

The estimator is deterministic: no randomness, no adaptivity, no dependence on
the grader seed or on the `budget` argument.

## Cost

| quantity | value |
|---|---|
| tracked FLOPs per network | `170,906,815,488` |
| FLOP budget | `272,000,000,000` |
| mean effective compute (graded) | `~1.745e11` |
| official row-0 wall time | 24.28 s against the 30 s predict guard |
| charged residual time, row 0 | 0.0547 s |

Effective compute is `tracked FLOPs + 1e11 * residual wall seconds`, so the
charged term is hardware-dependent and does not transfer between machines
unchanged.

## Local result bundles

- `official_129basis_mini100_20260731.json` — a full 100-network run on the
  **exposed Mini split** (`official_phase1_mini`, dataset SHA-256
  `5b00938b6bd809fe80acef08772c5654edf467863225ca9e304b76c779ecf433`) of the
  *later* fused-reduction build, not of this archive. Adjusted `1.4641716e-7`,
  0/100 failures. Its tracked FLOPs are `170,872,998,912`, exactly
  `33,816,576` below this package — the fused-reduction saving. Use it as
  development evidence for the design, never as the graded result.
- `kerdock_mub5_official_full100.json` — the same 66,048-point design with
  ordinary dense propagation, on the same exposed Mini split: adjusted
  `2.2565646e-7` at `2.689e11` effective compute (98.86% of budget).

Those two files bracket the arithmetic contribution on one fixed cohort: same
statistical design, same output rows, dense vs. Winograd propagation, adjusted
score `2.2565646e-7` → `1.4641716e-7`, a 1.5412× improvement bought entirely
by charged-cost reduction.

The exposed Mini cohort and the graded AIcrowd cohort are **different network
sets** (compare `daniel-harrison` against `patricia-hawkins`). Do not chain a
number from one onto the other.
