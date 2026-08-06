# WHestBench competition-opportunity experiments

**Agent:** 5 follow-on synthesis and execution  
**Date:** 2026-07-30  
**Decision:** **STOP every newly reopened competition branch. No holdout or official submission package was opened.**

## Executive result

I executed the two opportunities identified by the cross-agent review and followed their natural descendants:

1. network-adaptive, exact-mean, high-degree sphere-Stein and harmonic controls;
2. downstream-weighted salvage of the T4 layer-31 anchor hedge;
3. signed/calibrated Kerdock weighting as a descendant of the control branch;
4. downstream-singular-vector-aligned coefficient fitting as a descendant of the Stein oracle ceiling.

The experiments used actual width 256, depth 32, the complete 66,048-row seed-3 Kerdock design, deterministic He-Gaussian networks, and independent scrambled-Sobol references. Across all stages I propagated approximately **21,889,024 Gaussian reference trajectories** and **3,170,304 main Kerdock rows**, excluding the additional companion and rotation work inside T4.

The conclusions are decisive:

- The exact-mean controls contain substantial **oracle** correction signal, but no tested legal coefficient estimator extracts it robustly.
- The preregistered Stein candidate lost **1.072%** unbiased MSE on 16 fresh networks and had a **13.02%** worst-network regression.
- A downstream-aligned fitting descendant achieved only a post-development **1.32%** gain with an interval touching no gain; it did not clear a validation gate.
- The signed-calibration implementation mostly retained nonnegative weights and did not improve validation performance.
- The anchor shrink/abstention model looked mildly positive under grouped development cross-validation, but a fully frozen fresh eight-network × three-rotation validation landed at **1.000097 candidate/base**, essentially exact tie, with only **3/8** network wins.
- The same anchor correction span still has a **1.1676× oracle gain**, proving that absolute phase identification—not direction capacity—is the remaining obstruction.

No result justifies changing the current competition executable.

---

## 1. Exact-mean high-degree controls

### 1.1 Construction

For a unit direction `u`, let

` t = <x / ||x||, u> `.

For smooth `psi`, the uniform-sphere Stein control is

` s(t) = (1 - t^2) psi'(t) - (d - 1) t psi(t). `

Its exact spherical expectation is zero. Since a Gaussian direction is uniform and independent of radius, the corresponding Gaussian expectation is also exactly zero. This gives a legal external absolute reference without pilot truth.

I implemented:

- `softplus`, `sigmoid`, and `tanh` fields;
- thresholds `-0.1, 0, 0.1`;
- top downstream-Jacobian singular directions;
- first-layer and downstream-path direction comparators in the broad screen;
- projection away from Gegenbauer degrees 1–5, leaving broadband degree-6+ content while preserving exact zero mean;
- degree-6/8 and degree-6/8/10 zonal harmonic comparators;
- row-level ridge control fitting, complete-basis four-fold cross-fitting, output-rank truncation, and frozen scalar shrinkage.

### 1.2 Initial oracle screen

On eight fresh networks with two 65,536-sample references per network:

- degree-6/8 harmonic oracle scalar gain: **2.1388×**, 8/8 wins;
- tanh-Jacobian Stein oracle gain: **1.8961×**, 8/8 wins;
- sigmoid-Jacobian Stein oracle gain: **1.8794×**, 8/8 wins;
- combined broadband Stein oracle gain: **1.7704×**, 8/8 wins.

The legal four-fold fits were much smaller:

- harmonic: **1.0325×** gain, 5/8 wins, CI crossing no gain;
- best single Stein family: approximately **1.003×**, also inconclusive.

This passed the oracle-ceiling gate and justified coefficient-estimation descendants.

### 1.3 Aggressive ridge/rank/direction grid

I tested:

- 2, 4, and 8 directions;
- harmonics 6/8 and 6/8/10;
- sigmoid, tanh, and combined sigmoid+tanh Stein controls;
- ridge multipliers from `1e-8` through `100`;
- coefficient ranks 2, 4, 8, 16, and full;
- same-design and four-fold complete-basis fitting.

A low-reference development grid initially suggested a 7%+ candidate, but independent-half noise sometimes exceeded the baseline MSE. I therefore rejected that selection and reran the decisive screen with two independent **262,144-sample** references per network.

### 1.4 High-precision screen and preregistration

At high precision, the safest survivor was:

- tanh sphere-Stein controls;
- eight downstream-Jacobian directions;
- biases `-0.1, 0, 0.1`;
- degrees 1–5 projected out;
- same-design ridge multiplier `1`;
- full coefficient rank;
- frozen scalar multiplier `2`.

On eight development networks:

- unbiased candidate/base: **0.978149**;
- gain: **1.02234×**;
- wins: **6/8**;
- worst ratio: **1.01397**;
- pooled reference-noise fraction: **17.0%** of observed baseline MSE.

The validation protocol and thresholds were frozen in `PREREGISTERED_VALIDATION.json` before running IDs 7200–7215.

### 1.5 Fresh validation

On 16 fresh networks with two independent 262,144-sample references each, the preregistered primary produced:

| Metric | Result |
|---|---:|
| Unbiased candidate/base | **1.010720** |
| Unbiased gain | **0.989393×** |
| Network-bootstrap gain CI | **[0.97113, 1.00528]** |
| Wins | **9/16** |
| Median ratio | **0.99576** |
| Worst ratio | **1.13020** |
| Observed candidate/base | **1.008964** |
| Pooled reference-noise fraction | **16.4%** |

The unscaled secondary also lost:

- candidate/base **1.005053**;
- gain CI **[0.98586, 1.00281]**;
- worst **1.06237**.

The degree-6/8 cross-fitted comparator lost **1.3876%** aggregate unbiased MSE and had a **1.2426** worst ratio. Its same-design version gained **1.68%** in aggregate but had a **1.3072** worst ratio and a wide interval crossing no gain.

**Decision:** STOP exact-mean Stein/harmonic control deployment. The protected holdout IDs 7300–7315 were not opened.

---

## 2. Downstream-aligned coefficient descendant

The broad regression estimates an unrestricted feature-by-output coefficient matrix. The next natural descendant constrained each input control direction to its matched output right-singular direction of the linearized depth-32 Jacobian.

This reduces each direction to a tiny scalar regression and directly instantiates the proposed adjoint/downstream structure.

I screened:

- 2/4/8/16 matched singular modes;
- harmonics 6/8 and 6/8/10;
- sigmoid, tanh, and combined Stein controls;
- ridge `1e-8` through `100`;
- same-design and four-fold fitting.

The strongest reasonably safe fixed result was degree-6/8/10 harmonic alignment, 16 modes, ridge 1, frozen scale 2:

- gain **1.01318×**;
- 6/8 wins;
- worst **1.01020**;
- bootstrap CI **[0.99974, 1.02677]**.

The largest point estimate was approximately **1.0154×**, but its interval crossed no gain and its worst ratio was **1.0436**.

This is below the 2%–3% raw effect needed to pay for downstream Jacobian propagation, SVD, feature projections, and control fitting, and it is materially smaller than the parent oracle ceiling.

**Decision:** STOP without fresh validation.

---

## 3. Signed/calibrated Kerdock descendant

The same-design control estimator can be written as a network-specific calibrated weighting of the Kerdock rows. I audited the implied weights on all 16 Stein validation networks.

For the preregistered tanh candidate:

| Weighting | Networks with negative weights | Median negative mass | Maximum negative mass | Median effective support |
|---|---:|---:|---:|---:|
| scale 1 | 0/16 | 0 | 0 | 66,042.0 |
| scale 2 | 1/16 | 0 | `5.22e-6` | 66,024.0 |

The degree-6/8 harmonic calibration used negative weights on 6/16 networks, with maximum total negative mass `1.13e-5`.

Thus the tested descendant did **not** rely materially on the unrestricted signed-weight loophole. It remained almost a positive, near-uniform reweighting, and its statistical validation failed anyway.

**Decision:** no separate signed-node or weight-optimization branch is justified.

---

## 4. Downstream-weighted anchor salvage

### 4.1 Archived grouped reanalysis

I reconstructed candidate MSE exactly from the stored downstream geometry:

` MSE(alpha) = baseline + 2 alpha inner + alpha^2 norm_sq. `

Only the nine legal runtime features from T4 were used. All rotations remained grouped by base network.

The original unit-scale T4 correction had:

- candidate/base **1.14517**;
- only 6/16 network wins;
- worst network ratio **1.95386**.

A nested leave-one-network-out random-forest shrink model improved the archived development rows:

- candidate/base **0.97843**;
- gain **1.02205×**;
- gain CI **[0.99706, 1.04340]**;
- 10/16 network wins;
- worst **1.08213**.

A simpler OOB constant-scale probability gate gave:

- candidate/base **0.98434**;
- gain **1.01591×**;
- gain CI **[1.00026, 1.03257]**;
- 9/16 network wins;
- worst **1.04708**.

This was exploratory development evidence only. It uses the same c17/p2/p4/nested information class that T4 had already declared closed, so I did not open sealed IDs 6016–6031.

### 4.2 Frozen fresh-cohort test

To test whether the weak development signal transferred, I froze:

- the original T4 action policy;
- a 2,000-tree random-forest beneficial-direction classifier;
- active correction scale `0.2420163611`;
- probability threshold `0.5`;
- validation IDs 7401–7408 and literal rotations 3/11/97.

Network 7400 was generated before the final freeze and was explicitly excluded.

The eight fresh networks used six independent 65,536-sample Sobol streams each, aggregated into two 196,608-sample reference halves. Results across 24 rotation records:

| Metric | Frozen shrink policy | Original unscaled policy | Same-direction oracle scalar |
|---|---:|---:|---:|
| Candidate/base | **1.000097** | 1.186800 | 0.856469 |
| Gain | **0.999903×** | 0.842602× | 1.167584× |
| Network wins | **3/8** | 4/8 | 7/8 |
| Row wins | **5/24** | 9/24 | 15/24 |
| Worst network | **1.06802** | 2.88591 | 1.00000 |
| Gain CI | **[0.97821, 1.02652]** | [0.63664, 1.00967] | [1.08804, 1.26646] |

Pooled reference noise was **26.1%** of observed baseline MSE.

The frozen classifier activated on 10/24 rows. It successfully avoided most catastrophic full corrections, but it could not predict the remaining signed phase well enough to improve aggregate MSE.

**Decision:** STOP. The fresh result is effectively baseline, and the oracle/frozen gap directly confirms the absolute-phase observability barrier.

---

## 5. Competition decision

No branch should alter the current executable:

1. **Stein/high-degree controls:** failed preregistered fresh validation; raw and adjusted score are both worse.
2. **Downstream-aligned controls:** effect too small and uncertain to pay for implementation compute.
3. **Signed calibration:** does not materially exploit negative weights and inherits the failed control result.
4. **Anchor shrink/abstention:** exact tie on a genuinely fresh cohort, despite large target-labeled oracle headroom.

The most defensible current conclusion is:

> The remaining large oracle gain is real, but none of the tested legal observables estimates its network- and rotation-specific signed phase. More fitting, thresholds, shrinkage, control dictionaries, or signed reweighting within these information classes is not a credible competition path.

### Reopening condition

Reopen only for a materially new runtime observable that independently supplies absolute phase and satisfies one of:

- at least **1.3×** raw MSE improvement on a fresh grouped full-width screen with safe tails; or
- a complete measured adjusted-score win after actual subprocess packaging.

Do not reopen from another learner on the existing T4 features, another small zonal dictionary, or a post-hoc scalar chosen on exposed validation.

---

## 6. Reproducibility and governance

- Protected Stein holdout IDs 7300–7315 were not opened.
- Sealed T4 IDs 6016–6031 were not generated or opened.
- The Stein validation policy was hashed before IDs 7200–7215.
- The fresh anchor classifier and policy were serialized and hashed before IDs 7401–7408.
- All fresh tests used deterministic network IDs, fixed Kerdock construction, and recorded reference seeds.
- Raw per-network and per-rotation results, scripts, frozen specifications, model hash, and aggregate tables are included in the companion bundle.

