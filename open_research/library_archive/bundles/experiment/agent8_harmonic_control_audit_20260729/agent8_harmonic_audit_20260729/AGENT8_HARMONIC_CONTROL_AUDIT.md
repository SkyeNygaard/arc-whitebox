# Agent 8 — Harmonic-control theorem audit

**Project:** ARC White-Box Estimation Challenge 2026 / WHestBench  
**Date:** 2026-07-29  
**Verdict:** **VERIFIED AFTER SCOPE CORRECTIONS**

## Executive verdict

The harmonic-control story is defensible only as a **taxonomy of distinct results**, not as one universal obstruction.

1. Several named low-degree controls are **exact algebraic no-ops** on the complete Kerdock spherical 5-design.
2. A wider, nonpolynomial class—bias-free one-hidden-layer ReLU Stein fields—also has an **exact blockwise annihilation theorem**, but only under explicit homogeneity and architecture assumptions.
3. Degree-6/8/10 zonal harmonic controls are **not annihilated**. A small frozen dictionary failed empirically; the limiting infinite-width kernel supplies degree-specific oracle ceilings, not a finite-width or universal impossibility theorem.
4. “Analytically integrable implies low harmonic degree” is false. The symmetrized spherical Poisson kernel is an explicit counterexample with exact expectation and nonzero harmonic content at every even degree.

The manuscript should say:

> Complete Kerdock exactly removes several broad but explicitly named low-degree and homogeneous one-layer control families. Small tested high-degree zonal dictionaries did not improve frozen validation. These results do not exclude general analytically integrable, adaptive, biased, deep, or otherwise rich high-degree controls.

## 1. Exact taxonomy

| Control family | Exact class audited | Primary status | Evidence status | Narrow conclusion |
|---|---|---|---|---|
| Degree-1/2 exact-mean controls | Exactly radialized angular polynomials of degree at most 2; fixed linear output combinations | **Algebraically annihilated** | PROVED under the 5-design assumption | The correction `E[g]-Q_K[g]` is exactly zero. |
| Learned Hermite controls, degrees 1–4 | For each fixed network, coefficients may be arbitrary functions of weights but not quadrature-node-dependent; resulting angular polynomial has degree at most 4 | **Algebraically annihilated** | PROVED under the 5-design assumption | Learning coefficients cannot create a nonzero correction inside this class. |
| Polynomial Stein controls | Vector field components are polynomials of total degree at most 4 | **Algebraically annihilated** | PROVED under the 5-design assumption | The Gaussian Stein image has degree at most 5 and is integrated exactly. Polynomial fields of degree 5 or more are **not covered**. |
| One-hidden-layer bias-free ReLU Stein controls | `phi(x)=sum_j a_j ReLU(v_j^T x)`, fixed parameters, exact Gaussian radialization, antipodal orthonormal-basis blocks | **Algebraically annihilated blockwise** | PROVED under explicit model; numerically reproduced | Each individual basis-block average is exactly zero. Biases, depth at least 2, products, and nonhomogeneous radial terms are **not covered**. |
| Degree-6 zonal harmonics | Pure degree-6 component | **Bounded by an oracle ceiling in the limiting kernel; not annihilated** | ORACLE DIAGNOSTIC under the depth-32 infinite-width kernel | Degree 6 accounts for 13.93% of limiting-kernel Kerdock MSE; perfect removal gives at most about `1.162x` raw-MSE gain. This is not a width-256 theorem. |
| Small degree-6/8/10 zonal dictionaries | Tested fixed/adaptive direction families, 2/4/8 directions, ridge fitting/cross-fitting | **Empirically failed in tested forms** | One frozen failure plus exploratory variants | The frozen degree-6+8 four-direction rule scored `1.004439`; later shrinkage scored `0.999876`. This closes that campaign, not the full high-degree class. |
| Nonpolynomial analytically integrable controls | General class | **Untested as a class; not algebraically annihilated** | OPEN | Exact expectation says nothing about harmonic bandwidth. Explicit analytic controls can contain arbitrarily high degrees. |

## 2. Paper-ready lemmas

### Lemma 1 — radialized polynomial annihilation

Let `U` be uniform on `S^(d-1)` and let `Q_K` be a spherical `t`-design. Let `p(x)` be a polynomial of total degree at most `t`, and let

`p_bar(u) = E_R[p(Ru)]`

for any independent radial variable `R` whose required moments exist. Then `p_bar` is an angular polynomial of degree at most `t`, and

`Q_K[p_bar] = E_U[p_bar(U)] = E[p(RU)]`.

Therefore the exact-mean control correction `E[p]-Q_K[p]` is zero. The statement remains true after any fixed linear map or fixed linear combination.

**Proof.** Radial expectation changes coefficients but cannot increase polynomial degree. A spherical `t`-design integrates every spherical polynomial of degree at most `t` exactly. Linearity gives the final statement. `square`

**Corollary 1.1.** On the complete Kerdock 5-design, exactly radialized degree-1/2 controls and fixed-network learned Hermite controls through degree 4 are no-ops.

**Scope guard.** Passing a low-degree feature through a nonlinear suffix can generate higher degrees; Lemma 1 applies to the resulting control only when its final angular dependence remains degree at most 5.

### Lemma 2 — bounded-degree polynomial Stein annihilation

For the Gaussian Stein operator

`T phi(x) = div phi(x) - x^T phi(x)`,

suppose every component of the vector field `phi` is a polynomial of degree at most `r`. Then `T phi` is a polynomial of degree at most `r+1`. Hence a spherical 5-design with exact radialization integrates `T phi` exactly whenever `r <= 4`.

**Proof.** `div phi` has degree at most `r-1`, while `x^T phi` has degree at most `r+1`. Apply Lemma 1. `square`

**Scope guard.** “Polynomial Stein controls vanish” must be replaced by “polynomial Stein fields of component degree at most 4 vanish.” A degree-5 field can produce a live degree-6 term.

### Lemma 3 — blockwise annihilation of bias-free one-hidden-layer ReLU Stein fields

Let `B={b_1,...,b_d}` be an orthonormal basis, and use the antipodal block `D_B={+b_i,-b_i : i=1,...,d}`. Let

`phi(x) = sum_(j=1)^m a_j ReLU(v_j^T x)`

with fixed `a_j,v_j in R^d`, no input biases, and exact Gaussian radialization. Then the average of the Gaussian Stein control `T phi=div phi-x^T phi` over `D_B` is exactly zero for every basis `B`.

**Proof.** By linearity it suffices to consider `phi(x)=a ReLU(v^T x)`. Almost everywhere,

`div phi(x) = (a^T v) 1_{v^T x>0}`.

For a nonzero projection, exactly one of `v^T b_i` and `v^T(-b_i)` is positive. At a zero projection, use the symmetric convention `ReLU'(0)=1/2`, so the two antipodal derivative values still sum to one. Thus every basis pair contributes one copy of `a^T v` to the divergence sum, and the block-average divergence is `(a^T v)/2`.

For exact Gaussian radialization, `E[R^2]=d`, and

`(a^T b_i) ReLU(v^T b_i) + (a^T(-b_i)) ReLU(v^T(-b_i)) = (a^T b_i)(v^T b_i)`.

Summing over the basis and dividing by `2d`, with the radial factor `d`, gives

`(1/2) sum_i (a^T b_i)(v^T b_i) = (a^T v)/2`.

The two terms cancel. Summing over hidden units proves the claim. `square`

**Covered:** arbitrary signed linear combinations, including separately parameterized positive and negative components.  
**Not covered:** input biases, depth-2 or deeper fields, multiplicative interactions, node-dependent fitting, incorrect radialization, or a nonsymmetric derivative convention at exact zero projections.

### Proposition 4 — analytically integrable does not imply low harmonic degree

Fix `0<r<1`, a unit direction `v`, and define the spherical Poisson kernel

`P_r(t) = (1-r^2)/(1-2rt+r^2)^(d/2)`.

Define its antipodally symmetrized version

`A_r(u) = (P_r(v^T u)+P_r(-v^T u))/2`.

Then:

1. `A_r` is nonpolynomial and real analytic on the sphere.
2. Its spherical expectation is exactly 1.
3. Its harmonic expansion contains every even degree `0,2,4,6,...` with a nonzero multiplier proportional to `r^ell`.

Therefore `g_r(x)=A_r(x/||x||)` for `x!=0` (arbitrarily defined at zero) has exactly known Gaussian expectation 1 and nontrivial arbitrarily high harmonic content. In particular it contains degrees 6, 8, and 10.

**Justification.** `P_r` is the Poisson kernel for the unit ball, so its normalized boundary integral is 1 and its spherical-harmonic multiplier at degree `ell` is `r^ell`. Antipodal symmetrization removes odd degrees and preserves every even degree. `square`

This directly refutes the sentence “anything analytically integrable is low degree.”

## 3. Degree-6/8/10 empirical and oracle audit

### What is genuinely frozen

The campaign tested 36 configurations during discovery. The post-selected development winner was a fixed-random, degree-6+8, four-direction correction with raw-MSE ratio `0.988123` and 10/16 wins. It was then frozen on seeds 10016–10031 and reversed:

- aggregate raw-MSE ratio: `1.004439`;
- wins: `6/16`;
- paired bootstrap interval: `[0.996366, 1.013471]`;
- additional feature-output contraction: roughly `0.27B` FLOPs.

A later shrinkage sweep reused these networks as development and found a best ratio of `0.999876`, effectively turning the correction off.

**Correct classification:** the fixed degree-6+8 four-direction rule is **FROZEN EMPIRICAL — FAILED**. The other direction counts, direction families, and the degree-6+8+10 variants are **EXPLORATORY EMPIRICAL** because they were screened during post-selection but were not independently frozen one by one.

### What the limiting-kernel oracle says

The exact depth-32 infinite-width ReLU-kernel decomposition attributes:

- degree 6: `13.93%`;
- degree 8: `10.25%`;
- degree 10: `8.14%`;
- degrees 6+8+10: `32.32%`.

The associated ideal single-degree gains are approximately:

- degree 6 only: `1/(1-0.1393) = 1.162x`;
- degree 8 only: `1.114x`;
- degree 10 only: `1.089x`;
- degrees 6+8+10 jointly: `1.478x`.

These are **orthogonal-component oracle ceilings inside the specified limiting kernel**, not guarantees for width-256 networks and not bounds on controls that also affect other degrees.

The finite-width harmonic probe was reported as unbiased but variance-intractable; no reliable width-256 degree decomposition was established. The paper must not call the 13.93% figure a finite-width measurement.

## 4. Discrepancies and required wording changes

### D1 — “Three faces are one theorem”

**Verdict:** reject. Harmonic annihilation, high-degree empirical failure, and learning failure have different logical statuses.

**Replacement:** “Three complementary obstructions in tested information classes.”

### D2 — “Anything analytically integrable is low degree”

**Verdict:** false. Proposition 4 is an explicit counterexample.

**Replacement:** “Several named analytically tractable controls reduce, after radialization, to degree-at-most-5 angular functions and are therefore annihilated.”

### D3 — “Whole Stein family is zero”

**Verdict:** overbroad.

**Replacement:** “Polynomial Stein fields of component degree at most 4 and bias-free one-hidden-layer ReLU Stein fields are annihilated under the stated radialized complete-block construction.”

### D4 — “Degree-6/8/10 controls do not help”

**Verdict:** overbroad.

**Replacement:** “A preregistered small degree-6+8 zonal dictionary failed frozen validation; the broader small-dictionary discovery sweep, including degree 10, produced no validated gain.”

### D5 — “Only live error is degree 6”

**Verdict:** false if read literally.

**Replacement:** “All degrees at most 5 are integrated exactly; the first potentially nonzero harmonic degree is 6, with substantial error also present at higher even degrees.”

### D6 — V67’s status

The archive currently mixes an analytic identity with language saying “every tested” one-hidden-layer component vanished numerically. The blockwise identity is provable exactly. Split it into:

- a theorem row for the class in Lemma 3;
- a computational-reproduction row recording the `3.12e-17` maximum residual.

## 5. Proposed canonical-ledger changes

### Add theorem T29

- **ID:** T29
- **Evidence level:** Exact analytic identity with independent numerical check
- **Family:** Blockwise ReLU-Stein annihilation
- **Experiment/claim:** Bias-free one-hidden-layer fields `phi(x)=sum_j a_j ReLU(v_j^T x)` under exact radialization on antipodal orthonormal-basis blocks
- **Result:** `Q_B[div phi - x^T phi]=0` for every basis block `B`
- **Verdict:** PROVED UNDER EXPLICIT MODEL; do not extend to biased or depth-2+ fields
- **Primary source:** Agent-8 audit, Lemma 3; `verify_harmonic_claims.py`

### Amend V67

Replace “Every tested one-hidden-layer bias-free ReLU Stein component averaged to zero” with:

> The named bias-free one-hidden-layer ReLU Stein class is exactly blockwise annihilated by T29. The archived numerical reproduction had maximum block residual `3.12e-17`. Polynomial fields are covered only when their Stein image has degree at most 5.

### Amend Paper Claims Matrix

Replace the claim row “Analytic controls are annihilated” with:

- **Claim:** Named low-degree and homogeneous one-layer controls are annihilated
- **Status:** Proved, class-specific
- **Scope:** Exactly radialized angular polynomials through degree 5; polynomial Stein fields through component degree 4; bias-free one-hidden-layer ReLU Stein fields on antipodal orthonormal-basis blocks
- **Recommended wording:** “Several named low-degree and homogeneous one-layer control classes vanish exactly under complete Kerdock.”
- **Main attack:** Biased/deep/nonhomogeneous and general analytically integrable controls remain outside scope

Replace “Degree6+ controls do not help” with:

- **Claim:** Small tested degree-6+ zonal dictionaries did not validate
- **Status:** Frozen empirical for the selected degree-6+8 rule; exploratory for the rest
- **Recommended wording:** “A frozen four-direction degree-6+8 correction failed; no tested small degree-6/8/10 dictionary produced a validated gain.”

### Add counterexample to paper appendix

Include Proposition 4 to make the non-implication explicit rather than merely warning about it in prose.

## 6. Reproducibility checks

The included script independently checks:

1. A random 256-dimensional antipodal orthonormal block and a 17-unit bias-free ReLU Stein field: absolute block mean `1.15e-14`.
2. The symmetrized Poisson kernel in dimension 256: normalized spherical mean error `3.11e-15` and nonzero numerical Gegenbauer coefficients at degrees 6, 8, 10, and 12.

These numerical checks support the exact algebra but are not substitutes for the proofs.

## 7. Final decision

**Accept with scope corrections.** The paper has a strong harmonic-control section if it presents:

- exact low-degree annihilation;
- exact blockwise annihilation for a narrowly defined one-hidden-layer bias-free ReLU Stein family;
- limiting-kernel oracle attribution as model-specific diagnosis;
- the degree-6+ dictionary result as a frozen negative experiment;
- an explicit statement that general nonpolynomial analytically integrable and richer adaptive high-degree controls remain open.

**Claims that must not appear:**

- “Analytically integrable controls are low degree.”
- “The whole Stein family is zero.”
- “Degree-6+ controls cannot help.”
- “Only degree 6 remains.”
- “The 13.93% degree-6 share is a measured width-256 fact.”
- “The harmonic evidence proves no adaptive statistical correction exists.”
