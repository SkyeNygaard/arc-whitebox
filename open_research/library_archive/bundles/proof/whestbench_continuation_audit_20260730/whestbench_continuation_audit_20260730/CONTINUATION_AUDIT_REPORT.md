# WHestBench continuation audit

**Date:** 2026-07-30  
**Role:** independent continuation of the cascade/observability review  
**Disposition:** **major scoped theorem upgrades verified; broad adaptive impossibility remains invalid; no reopened empirical branch is a submission candidate**

## Executive decision

The newer Markdown reports materially improve the paper program, but they do not reverse the earlier central correction.

The project now has a stronger mathematical core:

1. **T22/T23 is fully dual-engine numerically reproduced.** The complete theorem-critical path was regenerated with both CPython Decimal/libmpdec and direct-C GMP/MPFR. The MPFR engine rebuilt 1,421 certified curvature leaves from 1,079 source intervals, made 342 exact dyadic splits, verified complete coverage of `[-1,1]`, reproduced the global minorant and spherical mean, and produced byte-identical GCC and Clang outputs.
2. **T16 is a full all-degree auxiliary-LP theorem, not merely a reduced-cost theorem.** The directed-Decimal primal certificate reruns byte-identically; all nonconstant degree-5 coefficients are positive; `K32^(6)>0` is certified; exact primal-dual equality and uniqueness follow. Combining T16 with the Kerdock upper certificate tightens the relative-excess upper bound to **`0.023324172950039%`**.
3. **T27 extends exactly to finite width on the fixed Kerdock/MUB line universe.** For a standard isotropic Gaussian first layer, conditioning on later network randomness gives a Hilbert-valued Gaussian-noise-stability expansion with nonnegative power-series coefficients. The symmetrized line kernel therefore satisfies the association-value signs needed for the complete-bases-plus-one-partial-basis theorem at every finite width and depth, under a nondegeneracy condition.
4. **The information-theoretic framing is improved but remains class-specific.** Conditional expectation gives the exact value of a runtime information sigma-field, and group-invariant observations can recover only group-invariant error components. The actual WHestBench feature map has not been proved to possess the required sign symmetry.
5. **The reopened empirical branches remain closed.** Saved row-level arrays reproduce the reported failures exactly. No tested Poisson, projected-ReLU, signed-near-collision, high-degree Stein, or T4 shrink/abstention branch produces a robust deployable gain.

The correct paper thesis is therefore:

> A certified static boundary, an exact finite-width fixed-support theorem, and a scoped information/falsification map for adaptive correction.

It is still not a universal theorem that nonlinear, adaptive, network-dependent, or arbitrary signed-node estimation cannot improve.

---

## 1. T22/T23: verification upgraded to full dual-engine reproduction

### Local execution

I ran the canonical v5.1 fast verification:

- `verify_theorem_package.py`: pass;
- 1,421 formal pointwise subintervals;
- global pointwise upper margin `-1.0045862406584556e-13`;
- spherical kernel-mean interval width `2.288700332700929e-22`;
- one-sided theorem logic verified;
- `verify_manifest.py`: `59 files` verified.

I then ran the independent direct-C GMP/MPFR replay:

- reconstructed 1,421 leaves from 1,079 source intervals;
- independently made 342 splits;
- maximum tree depth 4;
- exact dyadic ancestry and no gaps/overlaps verified;
- independently reproduced the global minorant;
- independently reproduced a spherical-mean interval nested inside the Decimal interval;
- independently reproduced the Delsarte energy and final ratio;
- GCC and Clang JSON outputs were byte-identical.

The MPFR ratio upper bound is

`1.00023365501029481377066020018598171905...`,

consistent with and slightly inside the conservative standalone T22 v5.1 published bound.

### Status

**T22: COMPUTER-ASSISTED CERTIFIED.**  
**T23: FULL THEOREM-CRITICAL NUMERICAL REPRODUCTION WITH TWO ARITHMETIC ENGINES.**

This remains one mathematical derivation implemented independently, not a proof-assistant formalization or independent proof of the underlying analytic argument.

### Scope

The theorem covers:

- dimension 256;
- depth-32 infinite-width normalized ReLU kernel;
- at most 66,048 nodes;
- network-independent deterministic rules, or randomized rules independent of the realized field;
- nonnegative mass-one linear cubature.

It does not cover finite-width arbitrary nodes, adaptive support or weights, nonlinear processing, or arbitrary signed nodes.

---

## 2. T16: full all-degree auxiliary optimum verified

The newest coordinator package closes the former primal-attainment gap.

### Reproduction

- `prove_t16_all_degree.py` reran in 3.69 seconds and reproduced the stored output byte-for-byte.
- `prove_t16_primal_dual.py` reran in 2.98 seconds and reproduced the stored output byte-for-byte.
- The independent C++ exact-integer audit reproduced all finite reduced-cost signs and the analytic-tail cutoff.
- A separate 400,001-point double-precision grid found no negative primal gap beyond rounding (`-1.11e-16` at contact-scale resolution).
- High-precision sampled sixth derivatives were positive at eleven points from `-0.999` to `0.999`, including the difficult negative region. This is a sanity check, not the proof.

### Certified proof data

- exact contact polynomial:
  
  `22102 t^3 + 21930 t^2 - 87 t - 85`;
- exact reduced-cost sweep: degrees `6..14,658`;
- analytic tail begins at degree `14,659`;
- worst reduced cost occurs at degree 7:
  
  `-2327215 / 9290262647272`;
- directed outer derivative-ratio bound:
  
  `F''/F' < 2.398586389549085 < 3`;
- directed transformed lower margin for `kappa^(6)+3 B_(6,2)`:
  
  `8.14928622573927... > 0`;
- all five nonconstant normalized-Gegenbauer coefficients are strictly positive;
- the degree-5 Hermite minorant is globally feasible and uniquely optimal.

### Tightened theorem number

The completed auxiliary optimum gives

- Kerdock/auxiliary-optimum ratio upper endpoint:
  
  `1.0002332417295003899...`;
- relative excess upper bound:
  
  **`0.023324172950039%`**.

This supersedes `0.02336550102949%` when the T16 completion is incorporated. The old number remains correct for the standalone T22 v5.1 release.

### Remaining proof-engineering gap

The full primal interval proof has been independently rerun, and the reduced-cost half has an independent C++ implementation. It has **not** yet been completely reimplemented in a second interval stack. Before publication, the highest-value proof task is an Arb/MPFI/MPFR or proof-assistant reproduction of the T16 primal-feasibility and derivative-ratio certificate.

---

## 3. Exact finite-width extension of T27

### The theorem

Let the first-layer Gaussian matrix be independent of all later network randomness, and write the finite-network output as

`Y(x) = F_Z(W1 x)`.

Conditioning on `Z`, expand the Hilbert-valued function `F_Z` in multivariate Hermites. Gaussian noise stability gives

`K_m(t) = sum_n a_n t^n`, with `a_n >= 0`.

After antipodal line symmetrization,

`Kbar_m(t) = sum_r a_(2r) t^(2r)`.

For mutually unbiased bases, define

- `A = Kbar_m(1)`;
- `O = Kbar_m(0)`;
- `C = Kbar_m(1/sqrt(d))`.

Then

- `A-O > 0` for a nonconstant even component;
- `O-C < 0`;
- `(A-O)+d(O-C) > 0` for a nondegenerate finite ReLU network.

The association-scheme optimization therefore applies exactly: at every support budget, arbitrary real mass-one line weights are optimized by complete bases plus at most one partial basis, with equal positive weights within each active basis and positive optimal basis masses.

### Independent checks

The bundled verifier passed its exact algebra and orientation-symmetry checks. Its 5,000-simulation width-256 association sanity had all required signs.

I also ran a direct, independent finite-network experiment:

- input dimension 4;
- width 8;
- depth 3;
- post-ReLU vector output;
- 120,000 independently sampled networks.

Estimated association values:

- `A = 0.17091584`;
- `O = 0.13078943`;
- `C = 0.13915568`;
- `A-O = 0.04012641 > 0`;
- `O-C = -0.00836625 < 0`;
- `(A-O)+4(O-C) = 0.00666141 > 0`.

Brute enumeration for every line budget from 1 through 8 selected exactly the theorem-predicted complete-basis/one-partial-basis allocation.

### Revised scope statement

Statements that “T27 is infinite-width only” are now stale. The correct split is:

- **T22 arbitrary-node near-optimality:** infinite width only;
- **T27 fixed MUB/Kerdock line-universe optimality:** exact at every finite width under the explicit Gaussian-first-layer ensemble model;
- **arbitrary-node finite-width near-optimality:** still open.

---

## 4. Group-invariant observability: theorem valid, WHestBench hypothesis unproved

The general projection theorem is sound:

- unrestricted value of runtime information is
  
  `E || E[e | G] ||^2`;
- if observations are invariant under a group action, only the invariant component of the error can be recovered;
- if the relevant representation has no invariant component, an invariant-observation correction has zero value.

This is a useful organizing theorem, but it does not automatically prove that the actual T4 feature map discards the needed orientation.

### New T4 empirical audit

Using the 48 frozen T4 development rows and the nine legal sign-invariant geometry features:

- all splits were grouped by the 16 base networks;
- targets were signed correction cosines for c17, p2, and p4;
- ridge, quadratic ridge, and ExtraTrees predictors were evaluated out of sample;
- network-level permutation tests preserved the three rotations of each base network.

Results:

- c17: all model `R^2 <= -0.085`; no significant signed correlation;
- p2: best `R^2 = 0.0009`, but correlation permutation `p = 0.23`;
- p4: all model `R^2 < -0.065`; no significant signed correlation;
- 11/16 networks had both c17 signs across rotations;
- 13/16 had both p2 signs;
- 13/16 had both p4 signs;
- within-network rotation variation accounted for roughly 65%–75% of total target variance;
- the best c17 model’s sign accuracy merely equaled the positive-sign constant baseline.

### Conclusion

The actual data are consistent with a small invariant signed component for this feature class. They do **not** prove exact conditional sign symmetry or upper-bound all legal observability.

A formal impossibility theorem now has a precise missing obligation: define the exact observation map and exhibit a measure-preserving symmetry that leaves it invariant while removing the target correction component.

---

## 5. Reopened empirical branches: exact recomputation and closure

### Reopened-path package

All 154 listed SHA-256 checks passed. I independently recomputed the headline metrics from the 68 saved per-network NPZ packages without importing the original analysis library.

Reproduced results:

#### Projected-ReLU 48-network extension

- pooled raw ratio `1.012682140618133`;
- pooled cross ratio `1.0162461364986783`;
- mean ratio `1.0423247210440894`;
- 18/48 wins;
- worst `1.3123910186960719`.

#### Frozen terminal descendants

Poisson:

- raw `1.0379391539423775`;
- cross `1.0445741546970215`;
- 7/16 wins;
- worst `1.1859479628207268`.

Projected ReLU, initial terminal block:

- raw `0.9272394367704979`;
- cross `0.9145146830063725`;
- 9/16 wins;
- worst `1.1950417993978952`;

but the larger 48-network extension reversed the apparent gain.

#### Outside-Kerdock signed probes

Network-derived pairs:

- frozen global raw `1.5573136434384325`;
- cross `1.7160851461130726`;
- 8/64 wins;
- worst `5.403558317893894`.

Random pairs:

- frozen global raw `1.441833108745329`;
- cross `1.567705689531434`;
- 10/64 wins;
- worst `2.8515338024613337`.

The large per-network oracle ceiling remains a phase-identification diagnostic, not a deployable result.

### Packaging defect

The archive’s hashes and saved outputs are coherent, but the included analysis scripts import an omitted upstream path:

`agent9_10_oracle_bundle/.../arc_experiments.py`.

Therefore the package supports **row-level metric reproduction**, not clean generation from source. Add the exact upstream bundle or vendor the minimal dependency.

---

## 6. Agent 5 competition-opportunity bundle

All 117 SHA-256 entries passed.

I independently recomputed from saved row-level outputs:

### High-degree Stein validation

Primary preregistered tanh control:

- candidate/base `1.0107204113104977`;
- 9/16 wins;
- median `0.9957583`;
- worst `1.1301987`;
- no gate pass.

### Fresh T4 shrink/abstention validation

- candidate/base `1.0000971160040377`;
- 3/8 network wins;
- worst `1.06802090327476`;
- same-direction target-labeled oracle `0.8564692816838451`.

The frozen estimator is an exact tie; the oracle gap again demonstrates direction capacity without deployable signed-phase recovery.

### Signed calibration

The tested weight calibrations used negligible negative mass and inherit the failed control result. They do not materially test the unrestricted signed-node loophole.

### Packaging qualification

The compact bundle contains row-level Stein JSONs and the anchor aggregate CSV, but omits the fresh-anchor vector NPZs and upstream T4/ARC dependencies. Its headline aggregates are reproducible from saved rows; fresh generation and vector-level replay are not self-contained.

---

## 7. Markdown and manuscript corrections

### Mandatory numerical corrections

1. Wherever the completed T16 certificate is used, replace
   
   `0.02336550102949%`
   
   with
   
   **`0.023324172950039%`**.

2. In the T16 proof sketch, replace the unsupported claims
   
   `F''/F' < 9/4`
   
   and “the negative Bell term is bounded by one quarter of the leading term” with the actual certified route:
   
   - `F''/F' < 2.398586389549085 < 3`;
   - `kappa^(6) + 3 B_(6,2) > 0` on `[-1,0]`;
   - therefore `kappa^(6) + (F''/F') B_(6,2) > 0` when `B_(6,2)<0`.

3. Update every theorem-scope table that excludes finite width from T27. T27 now covers finite width within the fixed symmetrized MUB line universe under the explicit Gaussian-first-layer model.

### Superseded reports

The Agent 10 architecture report and Agent 11 hostile-referee report remain valuable historical audits. Their broad critique of universal impossibility remains correct. The following objections are now superseded:

- T16 lacked primal attainment;
- T22 release had no complete implementation-diverse regeneration;
- all T27 claims were infinite-width only.

They should be labeled **PRE-CLOSURE HISTORICAL REVIEW**, not used as the current claim register.

### Minor corrections

- Fix malformed TeX in `STRONGER_CLAIMS_PROOF_ATTEMPTS_20260730.md`:
  
  `rac1{sum_b h(r_b)}` → `1 / sum_b h(r_b)`.
- Replace literal proof placeholders such as `` `square` `` with a proper end-of-proof symbol.
- Distinguish the standalone T22 v5.1 bound from the combined T22+T16 tightened bound.

---

## 8. Updated canonical claim hierarchy

### Headline theorems

1. **T22 — computer-assisted certified:** Kerdock is within `0.023324172950039%` of the infimum in the static, network-independent, nonnegative class after incorporating T16.
2. **T16 — computer-assisted certified:** the named degree-5 Hermite auxiliary is the unique optimizer of the unrestricted all-degree auxiliary LP.
3. **T27-FW — proved under explicit model:** complete bases plus at most one partial basis optimize arbitrary real mass-one line weights on the fixed MUB line universe at every finite width and depth for the standard Gaussian-first-layer ensemble.

### Exact general theory

- conditional-expectation information value;
- data processing for runtime information;
- association-scheme support extremality under explicit signs;
- common-bias and group-invariant non-identifiability under explicit models;
- correction-risk/replacement/crossing results;
- residual-kernel identity and equivariant spectral recertification;
- exact control-nullspace results for uniformly annihilated families.

### Frozen empirical conclusions

- T4 tested phase features do not transfer;
- reopened Poisson/projected-ReLU/signed-near-collision paths fail;
- high-degree Stein validation fails;
- T4 shrink/abstention ties baseline;
- no current candidate changes the executable.

### Still open

- arbitrary-node finite-width analogue of T22;
- arbitrary outside-universe signed nodes;
- nonlinear/network-adaptive cubature;
- a real independent absolute-phase observable;
- a formal symmetry theorem for an actual legal observation map;
- transformed residual kernels outside the tractable equivariant-linear class.

---

## 9. Highest-value continuation program

1. **Build T22/T16 v5.2.** Merge the tightened T16 optimum into one canonical one-sided theorem JSON and extend the independent C theorem assembly to the new lower bound.
2. **Second-stack T16 primal audit.** Reimplement the derivative-ratio, sixth-derivative, Hermite coefficient, and primal-dual certificates in Arb/MPFI/MPFR or a proof assistant.
3. **Human proof review of T27-FW.** Check Hilbert-valued Hermite measurability, continuity/a.e.-polynomial reasoning, and the exact nondegeneracy condition.
4. **Arbitrary-node finite-width certificate.** Derive or bound the finite-width ensemble kernel strongly enough to construct a finite-width Delsarte minorant. Uniform sup-norm perturbation is much too crude at the current `~5.7e-11` additive gap.
5. **Observation-symmetry theorem.** Freeze a precise T4-like feature map and either prove a measure-preserving sign action or construct a feature that breaks it.
6. **Repair empirical releases.** Vendor all upstream dependencies, raw vectors, split manifests, and clean commands so each negative result is generation-self-contained.

---

## Final conclusion

The continuation does not revive the universal observability-gap theorem. It does produce a substantially stronger and cleaner paper:

> Static arbitrary-node positive cubature is certified near-optimal for the limiting kernel; the auxiliary certificate is exactly all-degree optimal; fixed Kerdock/MUB line allocation is exactly solved even at finite width; and every tested cheap correction family still fails its deployment gate despite large oracle phase capacity.

That is a strong result. The unresolved frontier is now sharply localized to new geometry, new runtime information, transformed residual kernels, or arbitrary-node finite-width structure.
