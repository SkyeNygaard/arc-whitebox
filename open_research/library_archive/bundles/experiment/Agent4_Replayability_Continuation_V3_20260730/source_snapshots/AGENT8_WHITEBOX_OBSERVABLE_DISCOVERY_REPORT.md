# Agent 8 — White-box discovery of new legal observables

**Competition:** WHestBench  
**Date:** 2026-07-30  
**Protected competition data opened:** **No**  
**Recommendation:** **Conditionally continue one narrow high-rank identity program; stop the current rank-4/5 natural-source and generic-feature programs.**

## Executive verdict

This campaign found a real positive result, but not a deployable estimator.

1. **The current rank-4/rank-5 program fails Gate A for every tested target-free natural source.** The best legal rank-5 source has pooled oracle residual ratios of approximately **0.369 development, 0.328 validation, and 0.404 confirmation**. Because the competition target ratio is `0.230415`, the confirmation result cannot win even with exact free coefficients.
2. **A new gauge-invariant direct-output source has enough oracle capacity at high rank.** Constructing PCA modes from the 129 basis-resolved output means already computed by the baseline gives rank-20 pooled ratios **0.123/0.142/0.128** and rank-40 ratios **0.0569/0.0686/0.0671** on development/validation/confirmation. Rank 40 is below the competition threshold on every one of the 36 cases; its worst ratios are **0.147/0.211/0.161**.
3. **The dimension is not four or five.** A target-free cumulative-energy rule frozen on development selects about **36 modes** and retains competition-level oracle capacity on every validation and confirmation case. This is a source-coordinate result, not a statement that the target-informed defect ensemble lacks a lower effective rank.
4. **All tested signed observables fail Gate B.** Transported gate/margin innovations, Gaussian and Edgeworth closures, checkpoint/output skew, Walsh block contrasts, axis-vs-MUB contrast, antipodal odd responses, and small interpretable combinations all degrade on untouched validation and confirmation. Sign accuracy remains near chance.
5. **Replay is not the cause of failure.** Exact nonlinear final-layer replay of the frozen rank-5 composite agrees with its linear oracle calculation within `5.6e-6` in residual ratio on the worst audited case of each split.
6. **The surviving research question is precise:** can the roughly 36 direct-output PCA contractions be obtained by an absolute identity or a shared low-cost estimator? If not, an information-theoretic obstruction for the sigma-algebra of all 129 baseline block means would close this route.

## Result labels

- **Proved:** target-freeness and exact linearity of the direct-output source; hidden-gauge invariance of the direct-output source; gauge invariance of the balanced output subspace.
- **Computer-assisted proof / exhaustive frozen-corpus audit:** exact baseline reproduction; source-capacity calculations over all 36 cases; exact replay audit; permutation/scaling audit.
- **Numerical discovery:** the approximately 36–40-dimensional legal source frontier.
- **Oracle diagnostic:** all reported source-capacity ratios.
- **Deployable experiment:** none passed all gates.
- **Speculation:** a harmonic-alias identity may estimate the high-rank contraction vector jointly.

## 1. Protocol and provenance

The campaign used the authenticated OGAP corpus: 12 base networks, three frozen rotations each, and independent reference estimates. The split was:

| Split | Base seeds | Cases |
|---|---|---:|
| Development | 910001, 910003, 910009, 910019 | 12 |
| Untouched validation | 910033, 910043, 910051, 910067 | 12 |
| Independent confirmation | 910079, 910081, 910089, 910103 | 12 |

Rotations were 31001, 31013, and 31033 and were always grouped by base network. Source definitions were computed before loading each case's reference target. Observable model classes and coefficients were frozen on development before validation and confirmation. No protected competition data was opened.

Three independent recomputation paths were used:

| Path | Cases | Total wall seconds | Mean seconds/case | Maximum baseline mismatch |
|---|---:|---:|---:|---:|
| Full hidden transcript / source tournament | 36 | 1254.1 | 34.84 | 0 |
| Direct-output source frontier | 36 | 318.1 | 8.84 | 0 |
| Direct-output block observables | 36 | 244.7 | 6.80 | `1.11e-16` |

## 2. New legal source: direct-output basis PCA

Let the baseline design be partitioned into its 129 natural 512-node groups: 128 antipodally paired Kerdock bases and the antipodally paired coordinate axes. For the realized network and rotation, define

\[
y_g=\frac1{512}\sum_{x\in G_g}f_W(x),\qquad
\bar y=\frac1{129}\sum_g y_g,
\]

so `bar y` is exactly the baseline Kerdock output. Define the target-free output covariance

\[
C_y=\frac1{129}\sum_g(y_g-\bar y)(y_g-\bar y)^\top.
\]

Let `V_k` contain the leading `k` eigenvectors of `C_y`. The correction family is direct output addition in `span(V_k)`.

### Why this source is legal

- It uses only output values already evaluated by the baseline and their known group labels.
- It never uses the unknown spherical expectation or any reference target.
- Its construction adds no network evaluations; storing 129 output means and an SVD is negligible compared with the baseline forward pass.
- It is invariant under every hidden-neuron permutation and positive diagonal ReLU rescaling because those reparameterizations leave each `y_g` unchanged.
- Applying a correction in output space is exactly linear. There is no checkpoint replacement or nonlinear replay approximation.

For orthonormal `V_k`, with baseline error `e=bar y-I`, the oracle residual is

\[
\|e\|_D^2-D\|V_k^\top e/D\|_2^2.
\]

The unknown object is only the signed contraction vector `b=V_k^T e/D`.

## 3. Source-capacity frontier

### Direct-output source

| Rank | Development pooled / worst | Validation pooled / worst | Confirmation pooled / worst |
|---:|---:|---:|---:|
| 5 | 0.3689 / 0.8174 | 0.3283 / 0.8312 | 0.4041 / 0.8142 |
| 8 | 0.2657 / 0.5452 | 0.2685 / 0.6864 | 0.2912 / 0.6691 |
| 12 | 0.1986 / 0.4686 | 0.2202 / 0.5413 | 0.2220 / 0.5179 |
| 16 | 0.1508 / 0.4468 | 0.1702 / 0.4997 | 0.1623 / 0.3764 |
| 20 | 0.1229 / 0.3758 | 0.1417 / 0.3623 | 0.1283 / 0.2876 |
| 32 | 0.0713 / 0.2682 | 0.0904 / 0.2465 | 0.0814 / 0.2138 |
| 40 | **0.0569 / 0.1467** | **0.0686 / 0.2113** | **0.0671 / 0.1613** |

**Oracle diagnostic:** rank 5 is decisively insufficient. Rank 20 gives strong pooled capacity but unsafe tails. Rank 40 has competition-level capacity on every tested case.

### Frozen adaptive rank

The development-selected rule retains the smallest number of leading modes containing `0.9939596` of the squared singular-value energy among the first 40 modes.

| Split | Mean rank | Rank range | Pooled ratio | Worst ratio |
|---|---:|---:|---:|---:|
| Development | 35.75 | 35–37 | 0.0633 | 0.2129 |
| Validation | 35.50 | 34–37 | 0.0782 | 0.2260 |
| Confirmation | 36.25 | 34–38 | 0.0749 | 0.1830 |

This is the strongest positive Gate-A result from Agent 8: a fully target-free, gauge-invariant source with safe competition-level oracle capacity across all frozen splits. It requires about 36 coefficients.

### Hidden-interface source tournament

The tournament included transported mean/gate/near-zero innovations in five depth bands, combined physical bands, checkpoint basis PCA, final-margin Krylov directions, output pullbacks, checkpoint-output Hankel directions, balanced controllability/observability modes, and hybrids.

| Frozen source | Development | Validation | Confirmation | Confirmation worst |
|---|---:|---:|---:|---:|
| Best rank-5 composite | 0.3564 | 0.3553 | 0.4032 | 0.8239 |
| Best rank-8 composite | 0.2492 | 0.2809 | 0.3328 | 0.7909 |
| Development-greedy rank 20 | 0.1422 | 0.1608 | 0.1748 | 0.3996 |
| Union of 40 unique directions | 0.0978 | 0.1032 | 0.1088 | 0.2815 |

The 40-direction hidden union has high pooled capacity but worse tails than the direct-output source. The direct output construction is therefore the cleaner source for any continuation.

## 4. Why the observed “repair rank 4–5” did not transfer

There is no contradiction between the prior target-informed repair dimension and this source frontier.

- The prior rank estimate concerns low-dimensional energy after target-dependent alignment across oracle defects.
- Agent 8 asks for a canonical, target-free coordinate system derived from the realized network and baseline transcript.
- A vector family can have low target-informed ensemble rank while requiring many target-free coordinates because its orientation changes with network and rotation.
- The current experiments show exactly that separation: target information compresses; legal source construction does not.

The correct statement is:

> The correction energy may be approximately four- or five-dimensional after revealing its target-dependent orientation, while the tested target-free legal source requires roughly 36–40 coordinates to retain safe capacity.

## 5. Signed-observable discovery catalog

### Hidden/checkpoint mechanisms tested

- Gaussian closure contraction.
- Third-order and third-plus-fourth Edgeworth closure contractions.
- Checkpoint-basis skew.
- Signed maximum Walsh coefficient of checkpoint basis projections.
- Output-basis skew.
- Final-layer active-margin flux.
- Near-zero crossing flux.
- Transported mean-gap, gate-gap, and near-zero-gap projections.

### Direct-output block mechanisms tested

- Axis-vs-MUB signed contrast.
- Antipodal odd mean.
- Mixed even-squared/odd moment.
- Third and fifth standardized moments of basis outputs.
- Seven fixed weight-one Walsh characters for even block means.
- Seven corresponding Walsh characters for antipodal odd means.
- Full-parity Walsh characters.
- Signed maximum Walsh coefficients.

All mechanisms were tested univariately before small combinations. Model classes used one to three shared mechanism coefficients, not unrestricted black-box learners.

### Frozen results

| Source | Development grouped ratio | Validation ratio | Confirmation ratio | Validation / confirmation sign accuracy |
|---|---:|---:|---:|---:|
| Hidden rank-5 composite | 0.9791 | 1.0478 | 1.0480 | 0.467 / 0.483 |
| Hidden rank-8 composite | 0.9842 | 1.0568 | 1.0783 | 0.500 / 0.385 |
| Hidden rank-20 composite | 0.9926 | 1.0430 | 1.0155 | 0.467 / 0.446 |
| Hidden union-40 | 0.9997 | 1.0012 | 1.0010 | 0.492 / 0.431 |
| Direct-output rank 5 | 0.9519 | 1.1358 | 1.2488 | 0.433 / 0.533 |
| Direct-output rank 20 | 0.9626 | 1.0472 | 1.0915 | 0.496 / 0.458 |
| Direct-output rank 40 | 0.9679 | 1.0967 | 1.1646 | 0.460 / 0.490 |

**Conclusion:** development improvements were selection noise or unstable instance-specific effects. No tested observable contains enough stable signed phase to improve the zero-correction baseline out of network.

## 6. Hostile reparameterization audit

All source families were exactly covariant under hidden-neuron permutations to numerical precision. Positive diagonal ReLU rescaling was more discriminating.

| Source | Hostile scaling minimum principal cosine | Projector Frobenius difference | Status |
|---|---:|---:|---|
| Raw checkpoint PCA | 0.9625 | 0.5198 | Not gauge-stable |
| Ridge output pullback | 0.9881 | 0.4172 | Not exactly gauge-stable |
| Cross-Hankel | 0.9905 | 0.2173 | Not exactly gauge-stable |
| Balanced metric | 1.0000 | `1.5e-14` | Gauge-invariant output subspace |
| Direct-output PCA | 1.0000 | 0 | Gauge-invariant |

### Algebraic explanation

Let `H` be the matrix of centered checkpoint group means, `C_h=H^T H/129`, and `J` the final linearized output map. A positive hidden rescaling `S` sends `C_h` to `S C_h S` and `J` to `J S^-1`, so

\[
J' C_h' J'^\top=J C_h J^\top.
\]

The balanced source's output subspace is the leading eigenspace of this invariant matrix. Direct-output PCA is even simpler because the exact group outputs themselves are unchanged.

## 7. Exact nonlinear replay audit

The worst frozen rank-5 composite case in each split was replayed through the actual final ReLU map.

| Split / case | Linear ratio | Exact ratio | Difference |
|---|---:|---:|---:|
| Development / seed910009_rot31001 | 0.8097808 | 0.8097831 | `2.27e-6` |
| Validation / seed910067_rot31033 | 0.7901215 | 0.7901222 | `0.72e-6` |
| Confirmation / seed910081_rot31013 | 0.8239134 | 0.8239189 | `5.53e-6` |

The correction shifts were small (`1.2e-4` to `2.5e-4` RMS in final preactivation units). Thus the rank-5 failure is source-span failure, not nonlinear replay failure. The direct-output source has no replay approximation at all.

## 8. Competition economics

For the recorded `4.34x` gap, the required adjusted ratio is `0.230415`. Under the simplified whitened frontier, a source with oracle ratio `r_*` has allowance

\[
S_{max}=\sqrt{0.230415}-\sqrt{r_*}
\]

for the aggregate source-noise-compute term.

| Confirmation source | Oracle ratio | Rank | Aggregate allowance | Equal-share scale indicator |
|---|---:|---:|---:|---:|
| Direct rank 5 | 0.4041 | 5 | negative | impossible at Gate A |
| Direct rank 20 | 0.1283 | 20 | 0.1218 | 0.00609 |
| Direct rank 40 | 0.0671 | 40 | 0.2209 | 0.00552 |
| Adaptive direct | 0.0749 | 36.25 mean | 0.2064 | 0.00569 |
| Hidden union 40 | 0.1088 | 40 | 0.1502 | 0.00375 |

The equal-share column is only a scale indicator, not an independence claim. It shows that distributing the allowance over dozens of separately noisy contractions is demanding. More importantly, the actual frozen estimators already have raw ratios above one before any added compute is charged. They therefore fail economics decisively.

## 9. Claims I tried to disprove

### “A natural rank-4/5 legal source should inherit the full checkpoint oracle.”

**Disproved for the tested broad family.** Physical depth bands, gate and near-zero innovations, checkpoint PCA, Krylov, Hankel, output pullbacks, balanced modes, and development-selected fusions all remain around 0.33–0.40 pooled on untouched rank-5 evaluation, with catastrophic cases above 0.8.

### “The failure is caused by linearization or nonlinear final-layer replay.”

**Disproved for the audited rank-5 corrections.** Exact replay and linear projection agree within a few parts in a million in residual ratio.

### “Output-aware geometry will compress the correction to five modes.”

**Disproved.** Output-aware modes improve the source slightly but preserve essentially the same rank curve. They become strategically strong only around 20–40 modes.

### “More source capacity will make the coefficients easier to predict.”

**Disproved for tested observables.** The union-40 source has oracle ratio near 0.10, yet its best frozen mechanism is indistinguishable from zero and slightly harmful on validation and confirmation.

### “Block-level differences provide free signed phase.”

**Disproved for axis/MUB, antipodal, fixed Walsh, and maximum-Walsh mechanisms.** Apparent development gains reverse on both untouched splits.

### “The source bases are automatically legal under equivalent network parameterizations.”

**Disproved for raw Euclidean checkpoint PCA, ridge pullback, and cross-Hankel.** Balanced and direct-output formulations repair this defect.

### “No target-free high-capacity source exists.”

**Disproved.** The direct-output rank-40 and adaptive rank-approximately-36 sources have enough oracle capacity on every tested case. The obstruction moved cleanly to coefficient observability and economics.

## 10. Conflicts with existing ledger entries

1. **“Repair dimension approximately four or five.”** This must be scoped to target-informed oracle alignment. It must not be read as a claim that a canonical target-free source has rank four or five.
2. **Leading rank-4/5 late-interface program.** The current natural family should be closed at Gate A. Continuing to add similar physical channels or fit better predictors inside them is unlikely to move competition value.
3. **Universal source gate near 0.15.** Rank-20 direct output passes this gate in pooled results but not tails; rank-approximately-36 is needed for safe competition-level capacity. Source-specific economics and worst-network constraints remain necessary.
4. **Existing OGAP conclusion that legal observables lack stable coefficient information.** Agent 8 strengthens rather than contradicts it by adding output-covariance, gauge-balanced, axis/MUB, antipodal, and structured Walsh mechanisms with independent confirmation.
5. **Exact replay theorem.** No conflict. The replay theorem works; the source and observability gates fail.

## 11. Ranked mechanism catalog and null results

The machine-readable catalog is `results/MECHANISM_CATALOG.csv`. The strongest univariate development effects were only approximately 1–2% and had unsafe folds:

- Hidden checkpoint skew: about 0.983 grouped development ratio.
- Hidden output skew: about 0.984.
- Hidden signed Walsh maximum: about 0.985.
- Direct rank-20 fixed even Walsh bit 2: about 0.982.
- Direct rank-20 even skew: about 0.983.
- Direct rank-20 signed even Walsh maximum: about 0.983.

None was promoted because no univariate mechanism demonstrated stable whole-network value. The small multivariate combinations failed both untouched splits.

## 12. Strategic recommendation

### Continue, conditionally

Continue only the following tightly specified branch:

> **Direct-output basis-PCA contraction identity / information-bound program.** Freeze the gauge-invariant adaptive source, then either derive a joint absolute estimator for its approximately 36 contractions or prove that the 129 baseline group means do not contain enough information.

This branch deserves limited further work because Gate A is genuinely passed and source construction/replay are solved.

### Next three decisive tests

1. **Identity test:** express each direct-output contraction in the Kerdock block/Walsh/harmonic alias basis and determine whether randomized basis relabeling, an analytically integrated companion, or a telescoping identity yields an unbiased signed estimate without another full network evaluation.
2. **Joint economics test:** measure the full bias/covariance/cost matrix of any such vector estimator and compare it with the confirmation allowance `S_max=0.2064` for the frozen adaptive source. Treat shared samples jointly; do not split the allowance naively into 36 independent budgets.
3. **Information obstruction:** construct pairs of networks or rotations with nearly identical 129 group-output transcripts but opposite direct-source contractions, or estimate a conditional-variance lower bound. If the lower bound exceeds the T72 margin, stop the entire baseline-reuse observability path.

### Stop

- Further generic regression or symbolic search over the tested feature dictionary.
- More rank-4/5 fusions of the current physical, PCA, Krylov, Hankel, margin, or transported-gap channels.
- Raw Euclidean checkpoint sources that fail positive-scaling gauge invariance.
- Protected-data promotion: no candidate passes Gates B and C.

### Quarantine

- Development-selected rank-20 hidden source: good oracle capacity but source selection is label-informed, tails remain unsafe, and its estimator fails.
- Direct block-contrast combinations: retain only as documented null results unless a new identity changes their interpretation.

## 13. Reproduction

See `README.md` for commands. The scripts reconstruct all observables before loading target arrays, use grouped base-network splits, and write frozen selections before untouched evaluation. The large per-case interface arrays are not duplicated in this compact bundle; they are deterministically regenerated from the authenticated OGAP archive.

## Final recommendation

**Conditionally continue** the direct-output adaptive source only for exact identity or information-lower-bound work. **Stop** the current rank-4/5 natural-source and generic-predictor programs. Do not open protected data.
