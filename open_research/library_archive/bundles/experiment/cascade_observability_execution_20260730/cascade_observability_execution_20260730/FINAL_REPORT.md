# Cascade / Observability Blueprint — Completed Execution and Hostile Audit

**Executor:** Agent 3  
**Date:** 2026-07-30  
**Scope:** Execute the supplied `Cascade / Observability Proof Blueprint`, pursue every supported branch, and decide whether it yields (a) an observability-gap impossibility theorem or (b) a constructive improvement route.

## Executive verdict

# THE PROPOSED OBSERVABILITY-GAP THEOREM IS FALSE / UNPROVED AS FORMULATED

The execution does **not** certify that every legal estimator can improve the complete Kerdock baseline by at most a small finite-width quantity `gamma(256)`.

The failure is structural, not a missing experiment:

1. **The actual network output in the retained implementation is post-ReLU.** The implementation applies ReLU after every matrix multiplication, including the final layer.
2. **A ReLU transform of a nondegenerate Gaussian process is not a Gaussian random element.** At a single input, `ReLU(G)` has an atom at zero. Thus the Gaussian-measure average-case theorems invoked in C3/C4 do not apply to the actual output process.
3. **The best linear kernel rule is not automatically the Bayes rule among all nonlinear algorithms.** The completed orthonormal-basis counterexample demonstrates the gap explicitly: nonlinear processing of antipodal ReLU observations can exactly integrate a nontrivial ReLU family while the equal-weight linear estimator cannot.
4. **TEST-2 cannot upper-bound a supremum over all S2-measurable corrections.** Failure of ridge or boosted trees on one fixed dictionary is evidence against that dictionary, not a bound on `gamma(n) = sup_d rho(e,d)^2`.
5. **The claimed `O(L/n)` finite-width sector is not an error decomposition.** A finite-width covariance correction size does not imply that all nonlinear exploitability vanishes at infinite width. Non-Gaussianity already exists after applying ReLU to a Gaussian limiting field.

Accordingly, C3, C4, C8, and the central thesis must be removed or replaced. The valid theorem paper is narrower but still strong:

> **A certified boundary for static linear neural cubature, exact all-width on-support symmetry, and an empirical falsification map for cascade and nonlinear correction families.**

## What was completed

- TEST-0 architecture and prior audit.
- TEST-1a exhaustive Kerdock row-profile verification without constructing the full Gram matrix.
- TEST-1b exact infinite-width global-scale calculation.
- A constructive nonlinear counterexample to the C3/C4 inference.
- TEST-2 actual-width grouped observability probe on the complete retained T4 development panel.
- TEST-3 mode-resolved perturbation experiment on a fresh width-256/depth-32 network.
- TEST-4 reconstruction of the archived full layer oracle ladder.
- TEST-5 basis-economics and control-hurdle reconstruction.
- TEST-6 stratified correction-alignment audit.
- TEST-7 complete signed-weight interval-certificate reproduction and numerical exclusion curve.
- Corrected claim registry, revised blueprint, reproducible code, outputs, figures, hashes, and proposed ledger updates.

No protected T4 calibration or validation cohorts were opened.

---

## 1. TEST-0 — architecture and prior

### Result: PASS for the retained architecture-matched generator

The inspected implementation and archived architecture-matched reports agree on:

- input dimension 256;
- width 256;
- depth 32;
- no biases;
- ReLU after every layer, including the final layer;
- independent Gaussian matrices scaled by `sqrt(2/256)`.

This verifies the positive-homogeneity assumptions used by the exact scale-mode result. It also reveals the fatal C3/C4 issue: the final observable is post-ReLU rather than a linear Gaussian-process output.

### Evidence

- `sources/CASCADE_OBSERVABILITY_BLUEPRINT.txt`
- retained implementation excerpts identified in the Library (`shared_arithmetic_external_phase.py`)
- archived architecture report (`EXPERIMENT_REPORT(2).md`)

---

## 2. C2 / TEST-1a — fixed linear weights on the complete support

### Correct theorem

Let `F=(F(x_1),...,F(x_N))` be any square-integrable random field whose second-moment kernel is zonal, and let the complete Kerdock support have constant kernel Gram row sum. Let

- `G_ij = E[F(x_i)F(x_j)]`;
- `z_i = E[F(x_i) theta]`, where `theta` is the rotationally invariant integral;
- `w0 = 1/N * 1`.

Rotation invariance makes `z=z0*1`. Constant Kerdock row sums give `G1=s1`. For every mass-one rule `w=w0+v`, `1^T v=0`,

```
R(w)-R(w0)
 = v^T G v + 2 v^T(Gw0-z)
 = v^T G v
 >= 0.
```

Therefore uniform weights minimize ensemble MSE among **fixed linear mass-one weights on the complete support at every width**.

### Corrections to the blueprint

- The prior mean need not be zero; it is not used in the proof.
- “Exactly minimize” should not imply uniqueness unless `G` is positive definite on `1^perp`.
- This theorem does not close data-dependent weights, nonlinear aggregation, a changed support, or compute-adjusted partial designs.
- If total mass is not constrained, the optimum is an alpha-scaled uniform vector, not exactly the mass-one rule.

### TEST-1a result: exhaustive PASS

The retained 128 chirps were checked pairwise using an exact integer Walsh transform:

- chirp pairs checked: **8,128**;
- violations of Walsh magnitude 16: **0**;
- maximum integer deviation from 16: **0**.

This certifies, without storing a `66,048 x 66,048` Gram matrix, the pointwise profile:

- self: 1;
- antipode: 1;
- orthogonal neighbors: 510;
- cross-basis neighbors with absolute inner product `1/16`: 65,536.

Numerical checks of the retained rotation and explicit bases agree to about `1.7e-8` in float32-derived data.

See `results/MATH_DESIGN_AUDIT.json`.

---

## 3. TEST-1b — global output scale

### Result: FAILS the blueprint's `|alpha-1| <= 1e-6` gate only in the trivial direction; no useful win

For the infinite-width depth-32 kernel:

- spherical kernel mean `A0 = 0.9747299895417149`;
- complete-support kernel row average `s/N = 0.9747302329077503`;
- mass-one baseline linear risk `R0 = 2.433660354350664e-7`;
- unconstrained optimum `alpha* = A0/(s/N) = 0.9999997503247287`.

The optimal shrinkage differs from one by `-2.49675e-7` and reduces risk by only:

- absolute: `6.07625e-14`;
- relative: `2.49675e-7`.

So the scalar correction is mathematically real but operationally irrelevant. The finite-width alpha remains an empirical quantity; no fresh protected cohort was opened merely to estimate a sub-ppm effect.

See `results/MATH_DESIGN_AUDIT.json`.

---

## 4. C3 and C4 — decisive refutation of the proof route

### C3 status: FALSE / NOT ESTABLISHED FOR THE ACTUAL OUTPUT

The inference

```
infinite-width preactivation is a GP
=> final post-ReLU output process is Gaussian
=> kernel quadrature is the all-algorithm posterior mean
```

is invalid. If `G(x)` is a nondegenerate Gaussian field, then `ReLU(G(x))` has a point mass at zero and is not Gaussian. Kernel quadrature minimizes linear mean-square risk from second moments; it is not generally the conditional expectation under a non-Gaussian prior.

Even for a genuinely Gaussian random element, the exact linear Bayes rule would be the alpha-scaled uniform rule from TEST-1b, not the mass-one baseline.

### C4 status: NOT APPLICABLE

The classical average-case no-adaptation results cited in the blueprint concern linear problems under Gaussian measures. The actual induced post-ReLU field law is non-Gaussian, so those theorems cannot rule out adaptive or nonlinear evaluation algorithms here.

A further technical caveat is required before importing a broad information-based-complexity theorem: allowing arbitrary linear information is not automatically equivalent to restricting information to point evaluations.

### Explicit nonlinear counterexample

For dimension `d=16`, let

```
f_a(u) = ReLU(a^T u),  u on S^(d-1).
```

Observe `f_a(e_i)` and `f_a(-e_i)` on one antipodal orthonormal basis. Then

```
f_a(e_i)+f_a(-e_i) = |a_i|.
```

The nonlinear estimator

```
c_d * sqrt(sum_i |a_i|^2)
```

recovers the exact spherical integral `c_d ||a||` for every `a`. In 10,000 random trials its maximum absolute error was zero to floating-point precision. The equal-weight linear estimator depends on `||a||_1` and had MSE `6.35538e-4` in the same experiment.

This is not claimed to be a deployable depth-32 challenge estimator. It is a constructive counterexample to the logical step “Gaussian preactivation plus linear estimand implies all useful nonlinear processing is impossible after ReLU.”

See `results/MATH_DESIGN_AUDIT.json`.

---

## 5. C1 and C6 / TEST-3 — cascade and transfer

### C1 status: VALID AS A LOCAL DECOMPOSITION, NOT AN IMPOSSIBILITY FLOOR

The linearized recursion

```
e_(l+1) = A_l e_l + eta_l + r_l
```

and its telescoping expansion are valid bookkeeping when the ReLU crossing remainder `r_l` is controlled. This classifies where correction can enter. It does **not** prove that the injection terms are irreducible, unobservable, mutually noncancellable, or already minimized in the downstream-sensitive metric.

### Exact scale theorem: VERIFIED

For a bias-free positively homogeneous suffix, multiplying an intermediate activation cloud by `c>0` multiplies every downstream activation and final mean by exactly `c`.

A fresh 12-network float32 diagnostic at width 256/depth 32 found maximum relative discrepancy `2.25e-7` across tested layers, consistent with roundoff.

### Mode probe: diagnostic PASS for “no large amplification” on the tested panel

On 12 fresh architecture-matched networks with 192 Gaussian rows each:

- maximum mean-direction transfer gain: `1.0112`;
- maximum random-direction transfer gain: `1.1956`;
- maximum tested rank-one shape gain: `0.1432`;
- no tested transfer gain exceeded the preregistered `1.5` amplification gate.

The median mean-shift gain rises from about `0.31` at layer 1 toward `1` late in the network, while random-direction effects are more variable. No compounding mode was seen in this diagnostic. This is still not a uniform theorem over networks or perturbation directions.

See `results/TEST3_TRANSFER_PROBE.json`.

---

## 6. TEST-4 — oracle depth ladder

### Result: mechanism replicated from archived artifacts; blueprint monotonicity gate fails at early layers

The archived screen ladder gives MSE removed:

| Layer | MSE removed |
|---:|---:|
| 1 | 13.83% |
| 4 | 13.22% |
| 8 | 22.54% |
| 12 | 40.20% |
| 16 | 48.96% |
| 20 | 53.55% |
| 24 | 62.69% |
| 28 | 75.45% |
| 29 | 77.83% |
| 30 | 79.97% |
| 31 | 82.69% |

The complete per-layer table is in `results/TEST4_ARCHIVED_ORACLE_LADDER.csv` and the figure in `figures/TEST4_ORACLE_LADDER.png`.

The curve is not strictly monotone at early layers, so the blueprint's proposed monotonic sanity gate is too strong for this operational oracle swap. More importantly, the reviewed artifact does not contain the requested cross-layer coherence matrix of increments. Therefore C7 remains conditional; signed source-share numbers cannot be converted into a universal prefix-improvement ceiling without the missing covariance/cross-term information.

The robust conclusion remains empirical: late means are highly repairable, with layer 31 carrying the dominant oracle channel. It remains an unavailable-reference mechanism, not an observability theorem.

---

## 7. TEST-2 — exploitability probe

### The proposed gate is mathematically invalid as an upper bound

The blueprint defines

```
gamma(n) = max over all S2-measurable d of rho(e,d)^2.
```

A finite ridge/tree experiment on a chosen feature dictionary can establish only a lower bound if it succeeds. If it fails, it says nothing about the supremum over all measurable functions of full trajectories. Thus the prescribed PASS condition cannot pin `gamma(256) <= 1%`.

### Completed actual-width feature-class test

Using the retained T4 exact-rotation development panel:

- width 256, depth 32;
- 16 independent base networks;
- rotations 3, 11, 97 grouped by base network;
- 48 rows;
- no protected calibration or validation rows opened;
- nine pre-existing target-free geometry features;
- grouped leave-one-network-out nested ridge and ExtraTrees.

Every tested regression target had negative grouped OOF `R^2`:

- c17 ideal cosine, ridge: `-0.239`;
- c17 ideal cosine, trees: `-0.115`;
- p2 ideal cosine, ridge: `-0.106`;
- p4 ideal cosine, ridge: `-0.108`;
- oracle ratio, ridge: `-0.222`.

The learned arm selector achieved 56.25% accuracy and mean selected ideal cosine `0.1585`, below simply using the best fixed arm (`0.1968`) and far below the target-labeled oracle (`0.3625`).

### Decision

**Close this exact T4 feature dictionary for absolute-phase prediction. Do not convert the result into an upper bound on gamma.**

The width-scaling power law was not run as a theorem-deciding experiment because:

1. its PASS conclusion is logically invalid;
2. the actual retained challenge support is dimension-specific, so changing width and input/design dimension together would confound geometry with width;
3. varying hidden width while fixing input/output dimension requires a new rectangular architecture and a carefully frozen scaled baseline, not the square-width generator assumed in the blueprint.

See `results/TEST2_T4_OBSERVABILITY_PROBE.json`.

---

## 8. TEST-5 — marginal value and compute economics

### Result: PASS for the tested basis-count and companion regimes, not a universal information theorem

The retained full-width basis-count audit gives:

| Bases | Raw candidate/base | Wins | Row-scaled adjusted ratio |
|---:|---:|---:|---:|
| 96 | 1.3847 | 0/16 | 1.0275 |
| 64 | 2.2475 | 0/16 | 1.1123 |
| 32 | 4.2476 | 0/16 | 1.0522 |

With a `+5B` control, 96 bases would require about a `1.067x` raw gain merely to break even. The complete 129-basis rule remains the standalone package; 96 bases is at most a potential control host.

These measurements close the tested basis-removal economics. They do not establish that every possible shared-arithmetic evaluation scheme has negative value.

See:

- `results/TEST5_BASIS_ECONOMICS.csv`
- `results/TEST5_CONTROL_HURDLES_ROW_SCALED.csv`
- retained `sources/CONTROL_HURDLES.csv` and `sources/REPORT.md`.

---

## 9. TEST-6 — archived alignment meta-audit

### Result: the proposed pooled zero-alignment gate is invalid and empirically false for finite-width S2 methods

A meaningful meta-analysis must stratify by:

- S1 versus S2 information;
- infinite-width versus finite-width evidence;
- fixed versus learned/adaptive method;
- oracle versus legal correction;
- raw statistical gain versus cost and tail safety.

Pooling all legal correction families and testing a common zero cosine would mix fundamentally different estimands and regimes.

Several finite-width S2 families have clearly nonzero average alignment:

- compact companion: cosine about `0.400`, but tail/cost problems;
- full companion: cosine about `0.490`, but heavy cost and bad worst case;
- frozen radial-Hermite correction: mean cosine about `0.612`, raw ratio `0.729`, but worst ratio `1.583` and incomplete oracle capture;
- T4 frozen policy: cosine `0.0969`, but raw ratio `1.127854`.

The empirical pattern is therefore not “all legal corrections have zero alignment.” It is:

> **Some legal finite-width trajectory corrections carry real average signed signal, but existing methods fail complete deployment because of variance, unstable phase, tails, incomplete oracle capture, or compute.**

That is a strong falsification-map result but not C3.

See `results/TEST6_STRATIFIED_META_AUDIT.csv` and `figures/TEST6_ALIGNMENT_MAP.png`.

---

## 10. TEST-7 — signed-weight certificate

### Result: certificate reproduced; quantitative bound is practically vacuous

The full signed-weight certificate package was rerun successfully.

Certified value:

```
M = sup_(t in [-1,1]) [K(t)-h(t)]
  = 156999263604490023 / 9223372036854775808
  = 0.017021894267861247...
```

The proof identifies the supremum at `t=1`.

The resulting minimum negative-mass lower bounds are extremely small:

| Desired Kerdock-relative improvement | Required beta lower bound |
|---:|---:|
| 0.1% | ~5.59e-9 |
| 1% | ~6.99e-8 |
| 5% | ~3.56e-7 |
| 10% | ~7.13e-7 |

Thus the proposition is rigorous but does not quantitatively exclude competition-relevant off-support signed gains. The signed loophole on the **Kerdock line universe** is separately closed by T27; the off-support signed class remains open.

See:

- `sources/M_CERTIFICATE.json`
- `sources/NEGATIVE_MASS_EXCLUSION_CURVE.csv`
- `sources/AGENT5_SIGNED_WEIGHT_REPORT.md`
- `figures/TEST7_SIGNED_WEIGHT_CURVE.png`.

---

## 11. Final claim registry

| Claim | Final status | Correct interpretation |
|---|---|---|
| C1 cascade normal form | **PROVED-ROUTE / scoped** | Local linearized decomposition with crossing remainder; no irreducible-floor conclusion. |
| C2 uniform weights all widths | **VERIFIED AFTER CORRECTIONS** | Fixed linear mass-one optimum on complete support; alpha-scaled uniform if mass is free. |
| C3 Bayes optimality / zero alignment | **FALSE FOR ACTUAL OUTPUT** | Post-ReLU field is non-Gaussian; kernel linear optimum is not all-algorithm Bayes optimum. |
| C4 no free adaptation | **NOT APPLICABLE** | Gaussian-measure theorem assumptions fail for the post-ReLU field. |
| C5 static design floor | **DONE / scoped** | T22 infinite-width, fixed network-independent, nonnegative linear rules only. |
| C6 scale mode | **PROVED** | Exact positive-homogeneity result for bias-free suffixes. |
| C6 center/shape contraction | **EMPIRICAL / OPEN** | One-network probe finds no large amplification; no uniform theorem. |
| C7 early-node ceiling | **OPEN / CONDITIONAL** | Oracle ladder exists; coherence/cross-terms required for a rigorous ceiling. |
| C8 self-anchoring synthesis | **INVALID** | Depends on failed C3/C4 and unproved finite-width per-layer floors/noncancellation. |
| C9 gamma sector | **DEFINITION VALID; BOUND INVALID** | A model screen cannot upper-bound a supremum over S2. Nonlinear infinite-width sector omitted. |
| C10 S3 exclusion | **DOCUMENTED EMPIRICAL EXCLUSION** | No universal theorem claimed. |
| C11 signed-weight slack | **CERTIFIED BUT WEAK** | Rigorous `M`; practical off-support exclusion curve is vacuous. |

Machine-readable version: `CLAIM_STATUS_MATRIX.csv`.

---

## 12. Strongest defensible paper

### Recommended thesis

> We certify a near-optimal boundary for fixed network-independent nonnegative linear cubature at the infinite-width deep-ReLU kernel, prove exact all-width optimality of uniform mass-one weights on the complete Kerdock support by symmetry, and map why a large collection of finite-width cascade and nonlinear corrections fail deployment despite occasionally carrying real average alignment.

### Claims that can appear

1. T22's scoped computer-assisted near-optimality theorem.
2. T27's exact real-weight optimization inside the Kerdock line universe.
3. Exact all-width complete-support mass-one symmetry theorem.
4. Exact positive-homogeneity scale transfer.
5. Exact correction-risk and explicitly modeled anchor-replacement identities.
6. Frozen empirical oracle-depth ladder.
7. Stratified negative-result map showing the distinction between signal, observability, safety, and economics.

### Claims that must not appear

- “The baseline is the exact Bayes rule for the challenge output at infinite width.”
- “Adaptive or nonlinear evaluation cannot help at infinite width.”
- “All remaining headroom is finite-width non-Gaussianity of order depth/width.”
- “TEST-2 bounds gamma(256) by 1%.”
- “The 78% oracle headroom minus gamma is a proved observability gap.”
- “No legal cascade can bootstrap.”
- “All legal corrections have zero alignment.”

---

## 13. Corrected next program

The broad proof program is closed in its current form. A valid continuation has two separate tracks.

### Track A — theorem

Restrict the theorem to **fixed linear estimators**:

- complete-support all-width symmetry;
- T22 arbitrary-node nonnegative static infinite-width bound;
- T27 arbitrary real weights inside the Kerdock line universe;
- signed off-support stability bound, honestly labeled weak.

Do not claim nonlinear/adaptive Bayes optimality without a theorem for the actual post-ReLU process.

### Track B — constructive finite-width/nonlinear methods

Treat exploitability as empirical and class-indexed:

```
gamma_F = max over d in a preregistered function class F of rho(e,d)^2.
```

A failed experiment closes only `F`, not all S2. The existing evidence prioritizes:

- tail-safe use of the real radial-Hermite/companion alignment;
- genuinely external absolute-phase observables;
- targeted models trained on downstream-weighted correction loss;
- abstention calibrated by base network, not rotation rows;
- no reopening based only on in-sample cosine or oracle coefficients.

The current T4 geometry dictionary is closed. Existing radial-Hermite and companion families have real mean signal but fail tails/cost/completeness, so a continuation must solve those exact failure modes rather than repeat broad feature search.

---

## 14. Reproduction

Run:

```bash
./REPRODUCE.sh
```

This reruns the local mathematical/design audit, T4 grouped observability probe, transfer probe, figure generation, signed-weight certificate, and manifest verification.

The package contains source copies sufficient for all computations reported here except external Library documents used only for theorem-scope auditing. No protected dataset was opened.
