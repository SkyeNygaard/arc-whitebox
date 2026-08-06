# Theory: limits of static cubature for deep ReLU Gaussian expectations

This is the paper and the replayable proof archive behind the "how much room
is left?" half of the project. The competition half asks how well a static
cubature rule can do in practice; this half asks how well *any* static rule
could do, and answers it with two class-wide lower bounds.

## The two results, in plain terms

The benchmark asks for Gaussian expectations of a bias-free ReLU network.
Positive homogeneity turns that into integration on the sphere in dimension
256, and because the rule is fixed before the network is drawn, the ensemble
mean-squared error is exactly a spherical kernel discrepancy for the
normalized depth-32 ReLU kernel. That reduction is what makes the question
answerable at all.

The complete Kerdock system of 129 real mutually unbiased bases gives an
equal-weight rule on 66,048 antipodal nodes — the same design the competition
estimator ships.

| result | statement | scope |
|---|---|---|
| **Theorem 1** (nonnegative) | Kerdock risk is at most **0.0233242%** above the infimum over *all* nonnegative mass-one rules on arbitrary spherical nodes with at most 66,048 nodes | dimension 256, depth-32 limiting kernel, static and network-independent |
| **Theorem 2** (signed) | Even with arbitrary *signed* mass-one weights, no static rule can reduce Kerdock risk by more than **6.2940%** | same kernel and node budget |
| **Theorem 3** (auxiliary) | The optimal Delsarte minorant is the unique degree-five Hermite interpolant at the roots of `22102t³ + 21930t² − 87t − 85` | the polynomial auxiliary program |

The practical reading: for this kernel and this budget, moving nodes around or
reweighting them is nearly exhausted. Beating Kerdock by a wide margin
requires leaving the class — using the realized network, adaptive queries,
nonlinear reconstruction, or a different cost model. That is exactly why the
competition work stopped hunting for better static designs and spent its
remaining budget on [arithmetic instead](../arc_whitebox/submissions/production_baseline_320802/README.md).

## Layout

- [`paper/`](paper/) — LaTeX source of *Limits of Static Cubature for Deep ReLU
  Gaussian Expectations* (`main.tex`, `main.bbl`, `references.bib`).
- [`proof_archive/`](proof_archive/) — the arXiv ancillary verification
  archive, shipped **byte-identical** to the submitted version. Its file
  hashes are proof inputs, so nothing inside the hashed set is edited here.

## Replaying the proof

```bash
python -m pip install -r theory/proof_archive/requirements.txt
cd theory/proof_archive
python scripts/check_package.py
python scripts/run_verification_portable.py
python scripts/check_package.py
```

Five checks, about 15 seconds total on a laptop:

| check | what it does | time |
|---|---|---:|
| all-degree exact reduced costs | exact integer/rational proof that every omitted Gegenbauer degree has strictly negative reduced cost | ~3.3 s |
| nonnegative recovered-record consistency | replays the directed degree-five primal–dual records, endpoint separation, and record hash | ~0.02 s |
| Kerdock risk high-precision sanity check | independent non-directed calculation from the exact complete-MUB pair spectrum | ~1.3 s |
| original frozen signed rational witness | exact rational replay of the 146-profile signed certificate | ~9.4 s |
| positive-index and sign-count strengthening | exact rational inertia strengthening and the negative-support hierarchy | ~0.2 s |

### One portability note

`scripts/run_verification.py` is the runner as submitted to arXiv, and it
invokes the checks with the literal command `python`. On systems that only
provide `python3` — current macOS, most Debian-family installs — it dies with
`FileNotFoundError: 'python'` before running any mathematics. Because its
SHA-256 is listed in `SHA256SUMS` and is therefore a proof input, it is **not**
edited here. [`scripts/run_verification_portable.py`](proof_archive/scripts/run_verification_portable.py)
runs the same five checks in the same order using `sys.executable`. Adding it
does not disturb `check_package.py`, which verifies the listed files and
tolerates extra ones.

## Trust boundary — read before citing

The archive **exactly** checks every integer and rational step downstream of
the stored one-sided interval inputs. That is a real and checkable thing, and
it is what the five green checks above mean.

It does **not** independently reconstruct the complete depth-32 Gegenbauer
coefficient and curvature interval stack in a second directed-arithmetic
implementation. Those interval endpoints are inputs to the replay, not outputs
of it. Human review of the analytic reduction is also outstanding. So:

- ✅ "The rational and integer arithmetic of the certificate replays exactly."
- ✅ "An independent high-precision calculation of Kerdock risk lands inside the certified interval."
- ❌ "The theorem has been independently verified."
- ❌ "The interval inputs have been reconstructed by a second implementation."

The paper states the same boundary in its Scope and reproducibility section,
and the archive states it in `proof_archive/README.md`. This repository does
not soften it.

Some machine-readable records inside the archive retain an earlier internal
project label (`T16`, `T22`, `v5_2`) because their hashes are proof inputs and
renaming them would invalidate the certificate. The manuscript and the
directory names use the paper title.

## Certified constants

Cross-checked against the certificate JSONs on 2026-08-04.

| quantity | certified value |
|---|---|
| Kerdock risk interval | `[2.4336603575430029389, 2.4336603575430052277] × 10⁻⁷` |
| nonnegative auxiliary risk lower bound | `2.4330928587565937917 × 10⁻⁷` |
| Kerdock / nonnegative-infimum cap | `1.0002332417295004` |
| relative excess over the infimum | `0.023324172950039%` |
| signed frozen-witness risk lower bound | `2.2804861843861462175 × 10⁻⁷` |
| signed floor / Kerdock-upper fraction | `0.9370601683665084` |
| Kerdock / signed-infimum cap | `1.0671673322143325` |
| maximum signed reduction vs. Kerdock | `6.29398316334916%` |

The independent non-directed Kerdock calculation returns
`2.43366035754300522725…× 10⁻⁷`, which lies inside the certified interval.

## What the theorems do not say

They are one-sided lower bounds. Neither constructs an optimizer, and neither
proves its bound is attained — Kerdock may well *be* optimal in the
nonnegative class. They concern the infinite-width limiting kernel, not a
realized finite-width network. The budget counts full function evaluations; it
does not equalize preprocessing cost or wall time, which is precisely the term
the competition score charges most heavily. And they say nothing about
adaptive, nonlinear, or network-dependent estimators — the classes where the
current leaderboard leaders are presumably operating.

## Generative AI

Language models assisted with brainstorming, code, adversarial review,
drafting, and editing throughout, including in this directory. See the
paper's acknowledgments and
[`../whestbench/README.md`](../whestbench/README.md) for the project-wide
disclosure.
