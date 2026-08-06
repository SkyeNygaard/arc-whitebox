# ARC White-Box Estimation Challenge 2026
## Reopened Paths Experimental Campaign — Final Report

**Date:** 2026-07-30  
**Research role:** post-Agent-11 competition-path audit  
**Verdict:** **No submission candidate. All four reopened paths fail the continuation gate in their tested forms.**

Ratios below are candidate MSE divided by baseline MSE; lower is better.

---

## 1. Executive result

The hostile referee review correctly identified logical gaps in the broad impossibility narrative, but those gaps did **not** convert into a competition-winning mechanism in this campaign.

| Branch | Strongest honest result | Final decision |
|---|---:|---|
| 1. Downstream-weighted re-score of archived anchors | Best fixed arm across 28 external archived cases: `0.852×` pooled raw, `0.844×` cross-MSE, 21/28 wins; worst `1.924×` | Useful diagnostic, **not deployable**: large rotation/tail failures and far below the required ~`4.34×` competition gain |
| 2. Network-derived Poisson controls | Exploratory 16-network result `0.929×`; preregistered terminal result `1.038×`, 7/16 wins | **Closed in tested form** |
| 3. Exact-mean nonlinear projected-ReLU controls | Initial 16-network terminal result `0.927×`, but frozen 48-network extension `1.013×`; optimized adjusted ratio `1.067×` | **Closed in tested form** |
| 4. Outside-Kerdock signed near-collision probes | Per-network oracle `0.131×` on 64 validation networks, but frozen global weights `1.557×`, 8/64 wins, worst `5.404×` | Strong oracle/phase gap; **closed as a transferable estimator** |

The competition position requires approximately `4.34×` adjusted improvement. The only broad external raw improvement, the archived-anchor `0.852×`, is about `1.17×` and has unacceptable tails. The newly constructed controls either regress or disappear under larger frozen validation.

**Bottom line:** these logical loopholes remain mathematically open in broader classes, but none is now an evidence-backed route to winning the competition.

---

## 2. Work completed and compute used

The campaign recovered the canonical ledger, research tree v2, production code snapshot, Agent 1/3/4/5/6/8/9 bundles, and raw archived anchor records.

New full-width work used:

- width `256`, depth `32`;
- the exact `66,048`-row Kerdock design;
- 68 independently generated full-width networks, seeds `3000–3067`;
- four independent Kerdock rotations (`20–23`) for exposed synthetic reference halves;
- one baseline rotation (`3`);
- 340 complete 66,048-row depth-32 propagations: five per network;
- nominally about `59.7T` baseline-equivalent effective FLOPs at `175.62B` per propagation, before control overhead;
- 52 archived analytic-companion cases, 2 designs and 21 arms per case, yielding 1,092 arm-record evaluations.

No official protected cohort was opened. All new validation used generated synthetic networks or already-exposed archived records.

---

## 3. Frozen split protocol

### Full-width synthetic campaign

| Role | Seeds | Use |
|---|---|---|
| Fit | `3000–3003` | coefficients, ridge choice and initial scalar shrink only |
| Exploratory validation | `3004–3019` | identify whether any descendant deserved one terminal test |
| Preregistered terminal | `3020–3035` | fixed Poisson and projected-ReLU candidates |
| Preregistered extension | `3036–3067` | exact projected-ReLU candidate only; no refit |

The descendant preregistration SHA-256 was:

`a760d61708c917cbc2d0346168bed3e034be232744f9d4bbd7bacbb8a0156190`

The 32-network extension preregistration SHA-256 was:

`e5e093f74069425f943233c44fed27664b34224c91e7f47101efb8d90f4c8885`

### Archived anchor campaign

- fit: development cases `0–15`;
- calibration-only reporting: cases `16–23`;
- external: validation `100–107`, expansion `300–307`, rotation `200–203`, rotation expansion `400–407`;
- higher-reference truth used for validation and expansion where present.

---

## 4. Branch 1 — downstream-weighted re-scoring

### Question

Were useful archived anchors incorrectly killed by an unweighted layer-31 error threshold, even though their errors might lie in downstream-insensitive directions?

### Method

For every archived record and design:

1. recover baseline output, truth halves, probe map `beta`, stored anchor and stored output correction;
2. solve for the minimum-norm oracle anchor in the span of `beta`;
3. compare coefficient-space error with downstream output-space error;
4. fit one global scalar on development cases `0–15` only;
5. evaluate raw MSE, cross-MSE, cosine, inner product, wins and tails on all external groups.

### Findings

The corrected objective matters: several anchors have meaningful output correction even though their coefficient estimates are poor. The oracle span itself is enormous; for the strongest reduced-112 records, the oracle span leaves only about `0.0054×` pooled raw MSE.

The best fixed external arm was:

- `reduced112/companion_8`;
- frozen alpha `0.0907954`;
- 28 external cases;
- pooled raw `0.851978×`;
- pooled cross-MSE `0.843621×`;
- mean per-case ratio `0.955287×`;
- 21/28 wins;
- p90 `1.3407×`;
- worst `1.9242×`;
- mean correction cosine `0.2931`.

Performance by group:

| Group | Pooled raw | Wins | Worst |
|---|---:|---:|---:|
| validation | `0.6917×` | 7/8 | `1.356×` |
| expansion | `0.8848×` | 7/8 | `1.094×` |
| rotation | `0.8172×` | 3/4 | `1.201×` |
| rotation expansion | `1.0778×` | 4/8 | `1.924×` |

A more tail-stable full-129 mean-only companion arm reached `0.8958×` pooled across all external cases, but still had a `1.477×` worst case.

### Decision

**The scalar threshold was the wrong diagnostic, but rescoring does not reveal a robust competition estimator.** The useful direction remains unstable under rotation. Preserve downstream-weighted exact replay as the authoritative gate for future anchors; do not revive these archived arms for submission.

---

## 5. Branch 2 — network-derived spherical Poisson controls

### Construction

For weight-derived input directions—mean gradient, gradient covariance modes and first-layer singular directions—we evaluated symmetrized spherical Poisson kernels

`0.5 * [P_r(<u,v>) + P_r(-<u,v>)]`

at radii `0.03, 0.06, 0.10, 0.15, 0.20`. Each control has exact spherical expectation one. Coefficients were learned only from protected Kerdock rows using six basis-group cross-fit folds. No extra network trajectories entered the deployable estimator beyond the weight-derived direction construction.

### Results

The training-selected arm (`all`, ridge `0.01`, alpha `2`) looked promising on four networks but failed on the 16-network frozen validation:

- pooled raw `1.1450×`;
- pooled cross `1.1884×`;
- 3/16 wins;
- worst `1.7713×`.

A post-hoc exploratory scan found a mid-radius arm:

- exploratory validation raw `0.92858×`;
- cross `0.90722×`;
- 9/16 wins;
- worst `1.2170×`.

It was frozen without refitting and evaluated on terminal seeds `3020–3035`:

- pooled raw `1.03794×`;
- pooled cross `1.04457×`;
- mean ratio `1.02191×`;
- 7/16 wins;
- worst `1.18595×`.

### Decision

**Close the tested Poisson dictionary.** The broader theorem-level statement remains: analytically integrable high-degree controls are not impossible. Empirically, this weight-derived Poisson family has no transferable gain.

---

## 6. Branch 3 — finite-width nonlinear analytic-plus-residual controls

### Construction

We formed a low-dimensional network-derived input subspace and exactly integrable nonlinear controls

`ReLU(a_j^T P^T x + b_j)`.

The exact expectation was computed in the correct radial-reduced angular geometry using 384-point Gauss–Jacobi quadrature. This correction was necessary: once `b != 0`, the Gaussian projection formula cannot be paired directly with fixed-radius Kerdock evaluations.

Configurations tested:

- `k=2`, `m=32`, bias scale `1`;
- `k=4`, `m=64`, bias scales `0`, `1`, and `2`;
- six basis-group cross-fit folds;
- ridge sweep `1e-4` through `10`.

### Results

The training-selected biased arm failed on 16 frozen validation networks:

- pooled raw `1.0059×`;
- pooled cross `1.0077×`;
- 7/16 wins.

A bias-free descendant selected on exploratory validation reached:

- exploratory raw `0.97115×`;
- cross `0.96252×`;
- 10/16 wins.

It was preregistered as `nonlinear_k4_m64_b0_0.001`, alpha `-2`, and tested on 16 terminal networks:

- raw `0.92724×`;
- cross `0.91451×`;
- 9/16 wins;
- worst `1.1950×`.

Because the pooled gain was concentrated in hard networks, the exact candidate was extended without refitting to 48 terminal networks. Final result:

- pooled raw `1.01268×`;
- pooled cross `1.01625×`;
- mean ratio `1.04232×`;
- 18/48 wins;
- p90 `1.19210×`;
- worst `1.31239×`;
- network-bootstrap 95% interval for pooled raw ratio `[0.95914, 1.06164]`;
- bootstrap probability of true pooled ratio below one: `0.292`;
- bootstrap probability below `0.95`: `0.0115`.

Difficulty-stratified pooled ratios were:

- easiest quartile: `1.0450×`;
- second: `1.0570×`;
- third: `1.0867×`;
- hardest: `0.9563×`.

The apparent gain was therefore a hard-network concentration fluctuation, not stable correction.

### Favorable optimized cost audit

Even granting an optimized single-pass fold-statistics implementation:

- gradient direction construction: about `6.44B` FLOPs;
- projection/features: about `0.17B`;
- fold sufficient statistics: about `2.71B`;
- solves and SVD allowance: about `0.08B`;
- total incremental: `9.398B`;
- compute ratio against `175.62B`: `1.05351`;
- projected adjusted ratio: `1.06687×`.

This excludes dispatch and memory overhead, so it is favorable to the candidate.

### Decision

**Close this nonlinear projected-control family.** It fails both raw performance and complete economics after sufficient frozen validation.

---

## 7. Branch 4 — signed outside-Kerdock derivative probes

### Construction

For each network, we built 48 signed near-collision difference features from 16 weight-derived center/tangent pairs at angular separations `0.02`, `0.05`, and `0.10`. A 16-feature random-pair family served as control. The features used nodes outside the Kerdock universe and therefore genuinely tested the signed/adaptive loophole.

Global signed weights were fitted on seeds `3000–3003` with ridge and total absolute-weight caps. The best training rule used ridge `1e-4` and cap `0.1`.

### Results on 64 frozen validation networks (`3004–3067`)

#### Network-derived pairs

- frozen global raw `1.55731×`;
- frozen global cross `1.71609×`;
- 8/64 wins;
- worst `5.40356×`.

Yet the per-network oracle was strong:

- oracle pooled raw `0.13118×`;
- oracle mean ratio `0.16007×`;
- median required absolute signed mass `0.09565`;
- p90 signed mass `0.15335`.

#### Random pairs

- frozen global raw `1.44183×`;
- frozen global cross `1.56771×`;
- 10/64 wins;
- worst `2.85153×`;
- oracle pooled raw `0.33681×`.

### Interpretation

The outside-universe signed loophole has a real same-network oracle ceiling. It does **not** have transferable phase. Global coefficients learned on four networks reverse or amplify error on new networks. The result repeats the project’s central empirical pattern in a genuinely new estimator class: support/direction capacity exists, but signed absolute phase is not observable from the tested weight-derived representation.

### Decision

**Close global signed near-collision probes.** Reopen only if a new analytic mechanism predicts per-network coefficients or their downstream-weighted phase; do not run another generic coefficient learner.

---

## 8. Competition economics

The current public position requires about `4.34×` adjusted improvement to reach the cited seventh-place gate.

- Best broad archived raw result: `0.852×`, only `1.17×` gain, with a `1.924×` tail.
- Best newly constructed candidate after sufficient validation: `1.013×` raw and `1.067×` adjusted.
- Signed-pair oracle: large, but illegal as a deployable oracle and catastrophically non-transferable.

Even the most optimistic surviving diagnostic is several times too small. No composition is justified because the candidate errors are unstable and several branches reverse sign out of sample.

---

## 9. Final closure map

### Closed in tested form

1. Weight-derived symmetrized Poisson kernels at the tested directions/radii.
2. Projected exact-mean ReLU dictionaries with the tested subspaces, widths and cross-fit regression.
3. Global signed coefficients on network-derived outside-Kerdock near-collision pairs.
4. Submission revival of archived companion/analytic anchors based solely on corrected downstream rescoring.

### Still mathematically open, but unsupported as competition paths

1. A genuinely independent absolute-phase observable for a late-layer defect.
2. A theorem-backed analytic per-network coefficient for outside-universe signed probes.
3. A different nonpolynomial exact-mean control family with a preregistered oracle ceiling before implementation.
4. A finite-width analytic decomposition that captures a substantial component exactly without learned signed phase.

These should not be described as active winning paths without new evidence.

---

## 10. Reproducibility and artifacts

Principal scripts:

- `reopened_path_experiments.py`
- `downstream_rescore.py`
- `extended_validation.py`
- `terminal_descendant_eval.py`
- `projected_relu_extension_eval.py`

Principal outputs:

- `FULL_WIDTH_SUMMARY.json`
- `EXTENDED_VALIDATION_SUMMARY.json`
- `TERMINAL_DESCENDANT_RESULTS.json`
- `PROJECTED_RELU_EXTENSION_RESULTS.json`
- `SIGNED_64_VALIDATION.json`
- `downstream_rescore/DOWNSTREAM_RESCORE_SUMMARY.json`
- 68 compressed per-network row/result packages.

A split registry, run manifest, hashes, scripts and all row-level arrays are included in the accompanying reproducibility archive.

---

## 11. Recommended project action

Do not submit any candidate from this campaign. Update the canonical tracker with tested-form closures and retain only:

- the downstream-weighted evaluation machinery;
- the exact angular-mean implementation;
- signed-pair oracle arrays as a phase-identifiability benchmark;
- the split/preregistration protocol.

The competition-winning mechanism, if one exists, still requires a new source of **stable network-specific signed phase**, not another larger dictionary or another flexible regression over the same observables.
