# Hostile audit of the WHestBench two-paper open release

**Audit date:** 2026-08-02  
**Audited object:** two-paper manuscript package, open experiment ledger, theory artifacts, release scripts, reviewer packet, and public-facing claims  
**Protected benchmark data opened:** no  
**Overall verdict:** **the headline nonnegative theorem survived the audit; the original release assembly did not.** The revised repository is suitable for public external review, but not yet for an unqualified formal-publication claim of independent proof verification.

## 1. Executive assessment

The most important conclusion is positive: I did not find a mathematical contradiction in the central one-sided nonnegative theorem. The recovered exact recurrence, primal certificate, second interval implementation, endpoint-separation certificate, canonical theorem record, and independent direct-risk sanity calculation are mutually consistent. The theorem remains narrowly scoped to the dimension-256, depth-32 infinite-width normalized-ReLU kernel, static network-independent linear cubature, at most 66,048 arbitrary spherical nodes, nonnegative mass-one weights.

The release did, however, contain material defects that would have undermined reviewer confidence:

1. the stronger signed headline constant was not reproducible from the files that were actually shipped;
2. the signed factor ratio had been translated into the wrong percentage reduction;
3. a fixed-node-budget theorem was sometimes described as an equal-cost or same-cost theorem;
4. the public paper misidentified the replayed signed witness as the unrecovered reoptimized allocation;
5. strict nonattainment of an older abstract floor was allowed to read as if it applied to the newer strengthened floor;
6. a sign-count theorem was described in language that could be confused with a bound on total negative mass;
7. proof-critical corrected T16 and T22 artifacts existed in the Library but were omitted from the repository;
8. the repository lacked machine-readable ledger exports and a clear warning that historical ledger rows are not current claims;
9. bibliography metadata contained a wrong DOI and omitted the closest known Kerdock spherical-code optimality literature;
10. the release lacked a strict clean-checkout verification contract.

All ten issues are corrected or explicitly quarantined in this audited revision.

## 2. Final audited claims

### 2.1 Nonnegative mass-one static rules

For the declared limiting kernel and node budget, the canonical one-sided certificate gives:

- auxiliary lower bound on risk:
  `2.433092858756593791746720517735782461690689819511...e-7`;
- complete Kerdock risk upper endpoint:
  `2.433660357543005227609466502669764591481120637006...e-7`;
- certified ratio:
  `Kerdock / infimum <= 1.0002332417295003899...`;
- relative excess:
  `<= 0.02332417295003899...%`.

This is one-sided. It proves neither that Kerdock is strictly suboptimal nor that another cubature rule attains the auxiliary lower bound.

### 2.2 Signed mass-one static rules

The audited public signed theorem uses the frozen released coefficient allocation, not the unrecovered later reoptimization. Exact replay gives:

- signed-rule risk at least `0.93706016836650839349...` of the Kerdock upper risk;
- Kerdock-to-signed-optimum ratio at most `1.0671673322143324904...`;
- maximum reduction relative to Kerdock risk at most `6.29398316334916065...%`.

The distinction matters. `1.067167... - 1` is about `6.7167%`, but that is not the fractional reduction in Kerdock risk. The correct reduction is `1 - 1/1.067167...`, equivalently `1 - 0.937060168...`.

This is a fixed-node-budget theorem. It does not prove equal FLOPs, equal wall time, or equal implementation overhead.

### 2.3 Negative-weight support-count hierarchy

After duplicate support locations are consolidated and zero weights removed, the frozen witness implies:

- at least 1,072 negative-weight support entries rule out a Kerdock-to-rule factor of 1.05;
- at least 4,160 make the signed rule's certified floor exceed the Kerdock upper risk.

This is a support-count statement. It does not lower-bound or upper-bound total negative mass.

### 2.4 Nonattainment scope

The exact shared-Gram and zero-code arguments prove strict nonattainment for the older abstract rank/block-trace floor. They do not, without an additional argument, prove strict nonattainment of the later inertia-strengthened frozen-witness floor. Moreover, because signed total variation is unbounded, strict nonattainment alone does not yield a uniform positive numerical separation.

## 3. Mathematical audit

### 3.1 Kernel and cubature-risk formulation

The normalized ReLU correlation map is

`kappa(t) = [sqrt(1-t^2) + (pi-acos(t))t]/pi`,

with `K_0(t)=t` and `K_{l+1}=kappa(K_l)`. For a rule independent of the realized Gaussian field, ensemble MSE equals the zonal-kernel discrepancy. This independence condition is essential: the identity does not justify applying the same theorem to network-adaptive supports or weights.

An independent high-precision sanity script now evaluates the exact complete-MUB pair spectrum:

- one pair at `t=1` per node;
- one at `t=-1`;
- 510 at `t=0`;
- 32,768 each at `t=+1/16` and `t=-1/16`.

It independently integrates the spherical kernel mean and obtains risk

`2.43366035754300522725364710853...e-7`,

inside the certified interval. This is reassuring but is explicitly labeled non-directed numerical evidence, not a proof.

### 3.2 All-degree auxiliary theorem

The exact reduced-cost replay establishes strict negativity for every omitted normalized Gegenbauer degree. The finite exact sweep and analytic tail agree with the archived certificate.

The primal side is no longer supported by the superseded quarter-bound argument. The canonical route uses:

- a Krawczyk enclosure of the Hermite coefficients;
- positivity of all nonconstant Gegenbauer coefficients;
- an independently implemented `mpmath.iv` interval stack;
- the direct inequalities `F''/F' < 3` and `kappa^(6)+3B_{6,2}>0`;
- explicit positive endpoint residuals at both `t=1` and `t=-1`.

The endpoint certificate is important: continuity by itself proves endpoint nonnegativity, not equality only at the three interior contacts.

### 3.3 T22/global interval certificate

The recovered report states that a clean tree regenerated all 1,421 certified subintervals and verified a 59-file manifest. The release now includes the report and canonical record. What remains missing is the complete independently reconstructed directed coefficient/curvature stack in a separate implementation. Local replay against inherited intervals is not the same as independent verification of those intervals.

### 3.4 Signed witness

The originally released signed verifier replays the older degree-280/order-320 rational allocation. The audited strengthening holds that allocation fixed and substitutes the positive-index floor componentwise without changing coefficient consumption. This produces a valid exact-rational certificate and is slightly weaker than the unrecovered reoptimized T70 report.

The stronger historical number remains in provenance files with an explicit warning. It is excluded from abstracts, README headlines, reviewer summaries, and public theorem tables.

## 4. Literature and novelty audit

The paper now distinguishes five adjacent literatures:

1. classical spherical designs and Delsarte linear programming;
2. RKHS/kernel quadrature and worst-case discrepancy;
3. Kerdock codes, orthogonal spreads, real MUBs, and unitary/projective designs;
4. linear-programming energy bounds for weighted spherical codes and designs;
5. recent exact semidefinite-programming optimality results for certain Kerdock/MUB spherical-code arrangements.

The closest methodological neighboring work derives linear-programming energy bounds for weighted spherical codes, while the closest Kerdock-specific result proves packing optimality for the same 66,048-point dimension-256 arrangement under a separation/cardinality objective. The present theorem optimizes a specific deep-ReLU zonal-kernel energy/discrepancy at a fixed node budget. Neither objective automatically implies the other. The novelty claim must therefore remain kernel-, dimension-, depth-, budget-, and estimator-class-specific.

The Can et al. DOI was corrected to `10.1109/TIT.2020.3015683`; volume, issue, and pages were added. The Abdukhalikov-Bannai-Suda association-scheme paper and the exact-SDP Kerdock spherical-code preprint were added to Paper A.

Priority and novelty still require a human literature review before journal submission.

## 5. Empirical-paper audit

Paper B is strongest as a transparent technical report and open research record, not as a theorem of universal impossibility. Its most durable contributions are:

- separation of oracle capacity from legal estimator construction;
- the complete-block variance identity;
- explicit observability, initialization, recurrence, compute, and tail gates;
- a ledger that preserves negative results, supersessions, and missingness.

The empirical release remains incomplete. The exact final 129-basis package, official per-network Mini-100 JSON, mixture-ladder scripts and arrays, pooled-within Taylor scripts, and rank-sweep bundle are absent. Claims derived from them remain labeled reported rather than independently reproduced.

No public sentence should infer that every adaptive, nonlinear, finite-width, or analytic-residual estimator is impossible.

## 6. Repository and reproducibility audit

The revised repository now includes:

- recovered v5.2 canonical artifacts;
- corrected T16 interval and endpoint records;
- exact all-degree replay;
- original and audited signed-witness replay;
- independent direct Kerdock-risk sanity check;
- machine-readable CSV exports of major ledger sheets;
- an explicit historical-artifact status note;
- a strict manifest/hash/link/citation/placeholder check;
- a single command for the core mathematical replay;
- CI configuration for the public repository once published.

The CSV exports deliberately preserve superseded claims. `ledger/csv/README.md` warns that status and supersession fields must be used and that the exports are not a flat current-claim list.

## 7. Clean-checkout contract

A fresh extraction must pass:

```bash
python scripts/check_release_strict.py
python scripts/run_core_verification.py
```

The second command performs:

1. exact all-degree reduced-cost replay;
2. recovered v5.2 bundle consistency and endpoint checks;
3. independent high-precision Kerdock-risk sanity calculation;
4. original signed comparison-witness replay;
5. audited inertia/sign-count strengthening.

A pass establishes internal release consistency and exact replay against the archived inputs. It does not replace an external interval reconstruction or named human proof review.

## 8. Readiness judgment

### Ready now

- public GitHub release as an **external-review repository**;
- circulation of the two-page overview and Paper A to mathematical reviewers;
- publication of the experiment ledger with conspicuous evidence labels;
- forum posts that use the audited scope and constants.

### Not ready yet

- describing the proof as independently verified;
- a journal submission without human review of the analytic bridge and scope;
- claiming the finite-width benchmark is globally solved;
- presenting Paper B's missing-artifact experiments as reproduced;
- using the unrecovered stronger T70 constant as a theorem.

## 9. Remaining external gates

1. independently regenerate the full depth-32 kernel coefficient/curvature intervals using Arb, FLINT, MPFR, or another directed stack;
2. have a named mathematical reviewer check the Delsarte reduction, Hermite remainder, all-degree argument, signed inertia step, and scope language;
3. run the frozen public archive in multi-platform CI and publish its SHA-256 outside the archive;
4. complete a human related-work and novelty review;
5. recover or permanently demote the missing empirical baseline and experiment bundles.

## 10. Bottom line

The headline nonnegative result appears coherent and unusually strong within its declared class. The signed result is also meaningful, but it is a roughly 6.294% risk-reduction ceiling, not near-exact signed optimality. The original release overstated the reproducibility of that signed constant and blurred several scope and metric distinctions. The audited revision fixes those flaws, makes the proof trail substantially more inspectable, and provides a credible object for external review.
