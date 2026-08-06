# Agent 5 continuation — covariant source geometry, gauge transport, and phase obstruction

**Date:** 2026-07-30  
**Competition:** WHestBench  
**Protected competition data opened:** **No**  
**Continuation disposition:** **STOP rank 4/5; QUARANTINE rank 12; CONDITIONALLY CONTINUE rank 20–32 only for a shared absolute contraction identity or estimator.**

## Executive conclusion

The continuation closes the most important loopholes left by the first Agent 5 report.

1. **A much broader source tournament does not rescue rank 4 or rank 5.** I evaluated 212 frozen target-free constructions per full-width case, including nonlinear output secants, state-whitened controllability, residual regression, direct odd references, Krylov and resolvent filters, commutators, generalized residual pencils, and hybrids. The best exact confirmation ratios remain `0.220745` at rank 12, while rank 4 and rank 5 remain approximately `0.442` and `0.404`.
2. **The best rank-12 construction is slightly better than the first report.** Ordering block combinations by the observed nonlinear final-output secant gives exact confirmation `r*=0.220745271`. With only the idealized `3.125%` final replay charge, its adjusted zero-noise ratio is `0.227643089`, barely below the competition target `0.230414747`.
3. **This does not make rank 12 viable.** Its remaining shared-estimator square-root margin is only `0.002896`; equivalently, even a free coefficient estimator may add only `0.002688` normalized MSE. Its confirmation worst case is `0.513`.
4. **The hidden source geometry is stable, but the signed action is not.** At rank 12, the same-network cross-rotation hidden-subspace overlap is `0.823`, versus `0.047` for random subspaces. After target-free polar/Procrustes transport, signed coefficient cosine is `-0.096`, while absolute-coordinate cosine is `0.754`. The source envelope transfers; phase does not.
5. **The natural singular dictionary is not four- or five-sparse even with target access.** On confirmation, oracle selection of the best five coordinates still leaves residual `0.266951`. Seven oracle-selected coordinates are required to cross the zero-cost target, and the median effective coordinate rank is `7.70`.
6. **No transferable four-to-twelve-dimensional common block source appears.** A development-oracle PCA source of dimension 12 leaves confirmation residual `0.32–0.37`. Fixed Walsh-character sources require about rank 24 to become comfortably useful.
7. **Direct odd phase observables fail under an intentionally favorable ceiling.** Even when every method receives the exact oracle coefficient magnitudes and must predict only signs, the development-selected phase policy reverses on validation and confirmation. Thus the tested phase-bearing summaries do not solve observability.
8. **A shared vector estimator avoids an automatic linear penalty in rank, but known full-pass estimators remain hopeless.** The exact shared-sampling frontier is `(sqrt(c0 r*) + sqrt(gamma tr Sigma))^2`. Applying it to one additional complete-Kerdock rotation gives confirmation scores `1.653`, `1.416`, and `1.341` at ranks 12, 24, and 32—far above the target `0.2304`.

The resulting diagnosis is now precise:

> The late hidden correction has a stable, network-specific **subspace envelope** of moderate dimension. Its amplitude profile is partly stable. The missing information is the rotation-specific **signed phase** inside that envelope. No tested legal odd reference contains enough phase, and no rank-4/5 truncation contains enough capacity.

## 1. Starting point and protocol

The canonical program requests a target-free, network-covariant late-interface source, followed by scalar contraction estimation and exact final-layer replay. The first Agent 5 round found that the strongest natural source required roughly rank 20 for aggregate `r*<0.15`, with ranks 4 and 5 decisively inadequate.

This continuation froze all candidate definitions before evaluating their oracle targets. All three rotations of each base network remained grouped. Development was used only to select among the declared candidate menu; validation and confirmation remained grouped and untouched by selection. No official or protected cohort was opened.

Evidence labels used below:

- **Proved:** exact algebraic statement.
- **Computer-assisted proof:** algebra plus a numerical verifier at floating-point tolerance.
- **Numerical discovery:** target-free construction evaluated against an oracle target.
- **Oracle diagnostic:** target information used only to determine capacity or an upper ceiling.
- **Deployable experiment:** legal source and legal coefficient estimator. No positive deployable result is claimed.

## 2. Canonical block-Hankel geometry

Let `B in R^(129 x d)` be the centered final-hidden block-mean matrix and `M in R^(d x m)` the exact tangent of the final mean-ReLU map at the baseline cloud. Define

`H = B M`.

Write the compact singular value decomposition

`H = L Sigma R^T`.

### Theorem A5C.1 — canonical source action

**Classification: Proved.**

For rank `r`, use hidden source rows `D_r = L_r^T B`. Then the tangent output map is

`D_r M = Sigma_r R_r^T`.

For output error `e`, the oracle tangent correction is

`R_r R_r^T e`,

and one canonical block-coordinate representation of the hidden shift is

`alpha_r = L_r Sigma_r^(-1) R_r^T e`.

The formulas are invariant to arbitrary sign or orthogonal changes within a repeated singular block. Under a hidden-neuron permutation, `B` and `M` transform contragrediently, leaving `H`, `alpha_r`, and the output correction invariant while the hidden shift transforms covariantly.

**Consequence.** Capacity depends only on alignment of `e` with the legal right singular subspace `R_r`; large singular values alone do not imply capacity.

### Theorem A5C.2 — exact shared-vector sampling frontier

**Classification: Proved.**

Suppose a rank-`r` source is whitened in the physical output metric and has oracle residual ratio `r*`. One shared sample returns all coefficient estimates jointly, with covariance `Sigma/n`; one shared sample costs `gamma` baseline units. Let fixed baseline-plus-replay cost be `c0`. Then

`J(n) = (r* + tr(Sigma)/n)(c0 + gamma n)`

is minimized at

`n* = sqrt(tr(Sigma) c0 / (r* gamma))`,

with exact continuous optimum

`J* = (sqrt(c0 r*) + sqrt(gamma tr(Sigma)))^2`.

Thus rank does **not** necessarily incur the independent-coordinate penalty `sum_j sqrt(v_j gamma_j)`: shared samples and vectorized contractions can make the relevant variance term `tr(Sigma)`. Rank still matters through source capacity, covariance trace, arithmetic cost, conditioning, and bias.

This theorem preserves the rank-20–32 branch as logically possible if a single cheap sample supplies the entire coefficient vector. It does not rescue any known estimator.

## 3. Expanded target-free source tournament

### 3.1 Candidate mechanisms

The tournament included:

- prior block-Hankel modes;
- SVD of exact nonlinear block-output secants;
- state-whitened tangent controllability;
- least-squares secant and nonlinear-residual maps in an orthonormal hidden block span;
- combined tangent/secant/residual operator maps;
- direct pullbacks of output references: ones, baseline output, gate and margin summaries, row norms, Gaussian residual, Jensen residual, and block dispersion;
- direct and inverse Krylov spaces;
- alternating operator pencils and commutators;
- generalized nonlinear-residual versus tangent eigenvectors;
- fixed and development-selected Walsh characters over Kerdock basis labels;
- all frozen hybrid allocations between network-specific modes and Walsh modes.

### 3.2 Exact low-rank results

| Source | Rank | Development `r*` | Validation `r*` | Confirmation `r*` | Confirmation worst |
|---|---:|---:|---:|---:|---:|
| Nonlinear secant | 4 | 0.431136 | 0.355713 | 0.441716 | 0.846 |
| Nonlinear secant | 5 | 0.369433 | 0.327463 | 0.404465 | 0.812 |
| Nonlinear secant | 8 | 0.266213 | 0.267758 | 0.290770 | 0.663 |
| Nonlinear secant | 12 | 0.200316 | 0.219456 | 0.220745 | 0.513 |

No rank-4/5 construction came remotely close to the zero-cost target. State-whitened tangent modes, direct phase-reference sources, and operator pencils were substantially worse than the block/secant family. Hybrid sources did not improve on pure secant or prior modes at a fixed rank.

### 3.3 High-rank exact frontier

| Source | Rank | Development `r*` | Validation `r*` | Confirmation `r*` | Confirmation worst | Replay-only adjusted ratio |
|---|---:|---:|---:|---:|---:|---:|
| Nonlinear secant | 20 | 0.123319 | 0.139125 | 0.127663 | 0.290 | 0.131653 |
| Nonlinear secant | 24 | 0.104428 | 0.121064 | 0.109470 | 0.274 | 0.112891 |
| Nonlinear secant | 32 | 0.072914 | 0.087877 | 0.081667 | 0.221 | 0.084219 |
| Fixed Walsh | 24 | 0.176763 | 0.172256 | 0.185460 | 0.384 | 0.191255 |
| Fixed Walsh | 32 | 0.129428 | 0.127861 | 0.135520 | 0.278 | 0.139755 |

The nonlinear secant is the strongest source. The fixed Walsh source is weaker but significant because it gives a simple common, network-independent block dictionary with no SVD gauge. It confirms that useful legal signal is spread across many structured Kerdock-block channels rather than a few bespoke modes.

## 4. Minimal coordinate support

The output modes of `H` form a complete target-free dictionary. Their oracle coefficient energies determine whether the source failure is merely bad mode ordering.

### Confirmation aggregate

| Modes retained | Controllability order residual | Oracle-selected coordinate residual |
|---:|---:|---:|
| 1 | 0.607100 | 0.541887 |
| 2 | 0.521850 | 0.421763 |
| 3 | 0.495351 | 0.351352 |
| 4 | 0.443019 | 0.302181 |
| 5 | 0.404117 | 0.266951 |
| 6 | 0.372849 | 0.237943 |
| 7 | 0.315643 | 0.214174 |
| 8 | 0.290453 | 0.193530 |
| 9 | 0.263060 | 0.176745 |
| 10 | 0.254469 | 0.162044 |
| 11 | 0.235823 | 0.148832 |
| 12 | 0.223529 | 0.137767 |


Even with target access, the best five singular coordinates leave `0.266951` residual. The first oracle-selected coordinate count below the zero-cost target is seven. Median confirmation mode statistics are:

- effective coordinate rank: `7.70`;
- modes for 50% of explainable energy: `3.0` after oracle sorting;
- modes for 75%: `8.5`;
- modes for 90%: `20.5`.

**Conclusion:** the requested four/five dimensions are not present as sparse coordinates in the strongest legal singular dictionary. The low checkpoint-increment rank and the legal source-coordinate rank are different estimands.

## 5. Source transport versus signed phase

### 5.1 Hidden source stability

For the same realized network under three Kerdock rotations, normalized hidden-subspace overlap is:

| Rank | Median overlap | Random-subspace expectation |
|---:|---:|---:|
| 4 | 0.8489 | 0.0156 |
| 5 | 0.8031 | 0.0195 |
| 8 | 0.8108 | 0.0312 |
| 12 | 0.8231 | 0.0469 |
| 20 | 0.7987 | 0.0781 |
| 24 | 0.7899 | 0.0938 |
| 32 | 0.7663 | 0.1250 |


The useful hidden envelope is therefore highly stable and is not a rotation-specific numerical accident.

### 5.2 Polar/Procrustes transport

A target-free Procrustes transport aligns each source frame to the first rotation using only source subspaces. At rank 12:

- median smallest cross-Gram singular value: `0.540`;
- median signed coefficient cosine after transport: `-0.096`;
- median absolute-coordinate cosine: `0.754`;
- median squared-energy-profile cosine: `0.464`.

The same pattern persists through rank 32. A Grassmann-mean common frame gives the same result: magnitudes partly transfer; signed phase does not.

**Numerical discovery:** the canonical gauge problem is not the remaining bottleneck in this family. Polar transport works where the cross-Gram is nonsingular. The signed coefficients themselves vary with the quadrature rotation.

## 6. Common-channel and fixed-dictionary tests

### 6.1 Development-oracle common PCA

For each case, the canonical block coefficient `alpha_r` was computed. A common source subspace was fitted using only development oracle coefficients and then frozen.

At source rank 24, a 12-dimensional development-PCA block source leaves confirmation residual `0.345222`. At source rank 32, it leaves `0.320936`. Normalizing coefficients before PCA does not rescue the result.

Thus even a target-informed development search does not reveal a transferable rank-12 common block source.

### 6.2 Fixed Walsh characters

A target-free orthonormal dictionary was built from the 127 nonconstant Walsh characters over the 128 Kerdock bases plus one axis-versus-nonaxis contrast. Exact confirmation results are:

- rank 20: `0.238995`;
- rank 24: `0.185460`;
- rank 32: `0.135520`.

This is a constructive result: a simple common physical basis has genuine capacity. But it reinforces, rather than overturns, the dimensional conclusion. The useful dimension is approximately 24–32.

### 6.3 Hybrid tournament

Every allocation combining network-specific modes with fixed Walsh modes was evaluated at total ranks 8, 12, 16, and 20. Development selection retained the pure network-specific source at every rank. No hybrid improved the validation/confirmation frontier.

## 7. Direct phase-observable ceiling

Legal output references included:

- all-ones output aggregation;
- baseline output and centered baseline;
- tangent, secant, and residual RMS;
- tangent, secant, and residual skew/asymmetry;
- columnwise tangent–secant and tangent–residual contractions;
- final tangent column norms.

For each singular mode, the sign of its contraction with a reference was used to predict the oracle coefficient sign. The method was granted the **exact oracle coefficient magnitude**, so this is a generous phase-only upper ceiling.

The development-selected per-mode reference policy has optimal-shrink development residual `0.472`, but validation and confirmation both revert to `1.000`: the optimal action is to turn the correction off. The best development-selected single global reference also fails validation.

This closes the tested direct-reference phase class. It does not prove that all legal phase observables are impossible.

## 8. Full-companion shared-estimator economics

The 16 independent complete-Kerdock reference rotations permit a favorable empirical variance audit. Treat one entire companion Kerdock rotation as one unbiased shared vector sample, costed at `gamma=1` baseline pass. Using Theorem A5C.2 and the idealized replay fixed cost gives:

| Rank | Confirmation source `r*` | One-rotation covariance trace | Best continuous adjusted score |
|---:|---:|---:|---:|
| 12 | 0.223529 | 0.653326 | 1.653001 |
| 24 | 0.108968 | 0.738801 | 1.415744 |
| 32 | 0.081215 | 0.762328 | 1.341494 |


All are catastrophically above the target `0.230415`. The result remains negative even though all coefficients share one pass and no independent-coordinate penalty is charged.

At rank 32, the half-reference cross-fit uses eight complete rotations per half and reaches exact confirmation residual `0.230356` before compute. This is only barely at the raw target and is noncompetitive once its enormous compute and replay are counted.

## 9. Direct answers to the Agent 5 questions

1. **Why is measured repair dimension approximately four or five?**  
   Target-informed checkpoint increments have low mutual coherence and require four/five components for most increment energy. This is repair-energy rank, not legal coordinate rank.

2. **Are the dimensions stable physical depth bands?**  
   A stable hidden source envelope exists, but neither five fixed depth bands nor five secant/Hankel modes have sufficient capacity.

3. **Do the same abstract channels exist across networks or rotations?**  
   Across rotations of the same network, hidden source subspaces overlap strongly. Their signed actions do not. A universal low-rank block-coordinate source did not transfer.

4. **Is there a canonical gauge or transport rule?**  
   Yes locally: source-only polar/Procrustes transport has good margins at small ranks. It aligns the source but does not reveal coefficient phase.

5. **Is the late-interface image well-conditioned?**  
   Yes for the tested small ranks; prior exact replay and condition audits remain benign. Gauge cross-Gram margins degrade at ranks 20–32 on some cases, requiring block treatment and abstention.

6. **Does the source collapse under exact nonlinear replay?**  
   No. Exact replay closely matches tangent capacity and sometimes marginally improves it.

7. **Can controllability be large while observability is negligible?**  
   Yes, both by exact counterexample and empirically. State-whitened/weight-only controllability modes have poor target capacity, and high-capacity source phase remains unobserved.

8. **Is there a rank lower bound showing four/five are insufficient?**  
   For the tested natural family, yes numerically and robustly. All 212 target-free constructions fail at rank 4/5, and even oracle selection of five singular coordinates leaves confirmation `r*=0.266951`. This is not a universal theorem over every imaginable source.

9. **Can rank 8 or 12 improve economics?**  
   Rank 8 cannot reach the target. Rank 12 barely passes replay-only capacity with the nonlinear secant, but leaves essentially no coefficient margin and has catastrophic tails.

10. **What is the minimal sufficient source dimension under the score?**  
   - zero-noise aggregate: rank 12, narrowly;
   - credible aggregate margin: rank 20–24;
   - confirmation worst case below the raw target: rank 32 for the secant source;
   - safe coefficient observability: unknown.

## 10. Claims I tried to disprove

### Claim: the first source ordering caused a false rank-4/5 negative

**Attack:** 212 candidate constructions, including nonlinear secants, state whitening, operator pencils, direct references, and hybrids.  
**Result:** disproved as an explanation. Rank 4/5 remain far above target.

### Claim: exact nonlinear replay destroys a promising tangent source

**Attack:** exact compiled replay for all shortlisted ranks and sources.  
**Result:** disproved. Replay is not the bottleneck.

### Claim: a target-informed top-five support exists inside the full legal dictionary

**Attack:** oracle-sort all 128 singular-coordinate energies.  
**Result:** disproved on confirmation aggregate; five coordinates leave `0.266951` residual.

### Claim: the source identity itself rotates unpredictably

**Attack:** compare hidden source projectors across rotations.  
**Result:** disproved. Hidden source overlap is very high.

### Claim: basis gauge ambiguity explains the coefficient instability

**Attack:** polar/Procrustes transport and a Grassmann-mean common frame.  
**Result:** disproved. Signed coefficient correlation remains approximately zero after transport.

### Claim: simple legal odd references reveal phase

**Attack:** grant exact oracle magnitudes and test only sign prediction.  
**Result:** disproved for the declared reference menu. Development selections reverse or turn off on validation/confirmation.

### Claim: a common fixed block basis must be useless

**Attack:** fixed Walsh-character sources.  
**Result:** disproved. Rank 24–32 has real capacity, although rank 4–12 does not.

### Claim: rank count alone makes rank 24 impossible

**Attack:** derive the shared-vector sampling frontier.  
**Result:** disproved as a general theorem. A joint estimator can avoid a linear rank penalty. Known full-pass shared estimators still fail badly.

## 11. Conflicts with existing ledger entries

1. **Rank-4/5 leading program.** The canonical program should remain a conceptual target, but the tested natural geometric families now warrant a `STOP` disposition at rank 4/5 rather than merely “open.” This does not rule out an entirely new source construction.
2. **Four/five checkpoint-repair components.** This remains correct. The new work clarifies that it is target-informed repair-energy rank, not legal source dimension.
3. **Rank-12 disposition.** The first Agent 5 report stated that prior rank 12 slightly lost after fixed replay even with perfect coefficients. The nonlinear-secant source changes this: rank 12 now barely passes replay-only zero-noise economics. It remains quarantined because the total estimator margin is only `0.002896` and tails are unsafe.
4. **Gauge theorem.** Existing polar-gauge theory is supported, not contradicted. The new result shows that successful gauge fixing does not imply phase observability.
5. **T72 independent-noise frontier.** The theorem remains correct within its model. A new shared-sampling theorem is needed for vector-valued samples with joint covariance and shared cost.
6. **Universal `r*<0.15` preference.** Rank 20–24 satisfies the aggregate preference. Rank 12 does not provide enough room despite crossing the hard zero-cost threshold.

## 12. Recommendation

### STOP

- rank-4 and rank-5 physical-band, Hankel, secant, state-whitened, reference-pullback, Krylov, commutator, and tested hybrid sources;
- rank 8;
- another generic source-ordering sweep;
- direct phase models over the tested output-reference dictionary;
- companion full-Kerdock rotations as coefficient samples.

### QUARANTINE

- nonlinear-secant rank 12. Preserve it as a boundary example and theorem test, not as a winning candidate.

### CONDITIONALLY CONTINUE

Preserve exactly two source representations:

1. **rank-20/24 nonlinear secant**, strongest capacity/economics;
2. **rank-24/32 fixed Walsh block source**, simpler common dictionary and potentially easier analytic contractions.

Continue only on one of these three obligations:

1. derive one **joint absolute vector identity** or estimator whose per-sample cost is largely shared across all coefficients;
2. discover a new phase-bearing observable that passes the oracle-magnitude sign ceiling before magnitude learning;
3. prove a conditional-variance or two-network obstruction for the full legal transcript, which would justify abandoning late-interface correction entirely.

Do not open protected data. Do not train another generic predictor until one of those obligations is met.

## 13. Next three decisive tests

1. **Vector band-potential implementation:** compute all rank-24 adjoint-band contractions in one batched pass, measure covariance trace and exact incremental arithmetic, and apply Theorem A5C.2.
2. **Walsh contraction algebra:** exploit the fixed GF/Walsh block dictionary to search for exact cancellations, character identities, or low-cost blockwise Stein formulas unavailable to an adaptive SVD basis.
3. **Phase obstruction pairs:** construct conditionally matched network/rotation pairs with nearly identical legal block operators but opposite signed contraction vectors; turn the observed phase instability into a quantitative lower bound.

## 14. Reproduction

Core commands are documented in `README.md`. Principal scripts:

- `scripts/agent5_operator_pencil_tournament.py`
- `scripts/run_exact_shortlist_fast.py`
- `scripts/run_exact_high_rank.py`
- `scripts/analyze_coefficient_geometry.py`
- `scripts/gauge_transport_analysis.py`
- `scripts/phase_reference_ceiling.py`

Machine-readable outputs are under `results/`; plots are under `plots/`.
