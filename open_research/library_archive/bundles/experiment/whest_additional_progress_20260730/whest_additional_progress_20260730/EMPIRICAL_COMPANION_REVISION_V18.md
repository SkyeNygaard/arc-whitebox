# Scoped Falsification Map for White-Box Neural Integration

## Abstract

We report a reproducibly labeled campaign of finite-width white-box correction experiments around a strong 66,048-point Kerdock baseline. The goal is not to prove universal impossibility, but to identify which tested information classes fail complete deployment after accounting for signed alignment, tail risk, selection chronology, reference noise and compute. Several legal methods carry real average signal, and per-network oracle corrections can be large. However, the tested fixed policies, scalar predictors, harmonic dictionaries, analytic anchors, Poisson controls, projected-ReLU controls, coresets and signed near-collision probes fail because their signed phase is unstable, their gains disappear under grouped frozen extension, their tails are unsafe, or their compute cost exceeds the benefit. These results define a falsification map and explicit reopening conditions; they do not rule out new observables, arbitrary nonlinear estimators or new signed geometry.

## 1. Evidence labels

Every result must carry one of four labels:

- **FROZEN EMPIRICAL** — selection frozen before a grouped evaluation set;
- **ORACLE DIAGNOSTIC** — uses unavailable target information or per-instance fitted coefficients;
- **EXPLORATORY** — post-selected or insufficiently separated;
- **PROVISIONAL / MISSING** — primary package cannot be reconstructed.

Oracle and deployable numbers must appear in separate tables and figures.

## 2. Correct performance decomposition

For baseline error `e` and correction `c`,

\[
\mathbb E\|e-c\|^2=\mathbb E\|e\|^2-2\mathbb E\langle e,c\rangle+\mathbb E\|c\|^2.
\]

The correct question is signed, downstream-weighted alignment, not whether a latent state is accurately predicted in an unweighted Euclidean norm.

For runtime information `G`, the best unrestricted `G`-measurable correction is

\[
\mathbb E[e\mid G],
\]

and the information value is

\[
V(G;e)=\mathbb E\|\mathbb E[e\mid G]\|^2.
\]

A large target-labeled oracle span does not imply that the legal runtime information contains the oracle phase.

## 3. Target-noise limitation

Suppose an observed learning target is

\[
Y=\theta+\varepsilon,
\]

with mean-zero reference noise independent of features and the latent target. Then

\[
R_Y^2=\frac{\operatorname{Var}(\mathbb E[\theta\mid X])}{\operatorname{Var}(Y)}
=\frac{\operatorname{Var}(\theta)}{\operatorname{Var}(Y)}R_\theta^2.
\]

Thus observed predictability is attenuated by target reliability. Negative or weak grouped `R^2` against a noisy reference cannot establish zero latent predictability. The Path-2 result is therefore retained only as a matched-model comparison in its specific information class.

## 4. Main empirical map

| Family | Strongest honest result | Failure mode | Status | Reopening condition |
|---|---|---|---|---|
| T4 fixed policy | raw ratio 1.127854 on development | low signed alignment and severe tails | Frozen development failure | new feature class frozen before grouped validation |
| Small degree-6+8 dictionary | raw 1.004439 on frozen validation | no stable gain | Frozen failure | preregistered richer live-degree family with oracle ceiling |
| Path-2 scalar models | no value over matched constants | feature-dependent scale not identified | Narrow negative | reliable target and new noninvariant features |
| Archived downstream anchors | pooled raw 0.851978 externally; worst 1.924 | rotation-dependent phase/tails | Diagnostic, no submission | stable downstream-weighted phase predictor and tail guard |
| Poisson controls | exploratory 0.929; terminal 1.038 | post-selection did not transfer | Closed tested dictionary | different directions/family with frozen oracle support |
| Projected ReLU controls | 16-network 0.927; 48-network 1.013; adjusted 1.067 | hard-network fluctuation and cost | Closed tested family | materially different subspace/control and preregistered extension |
| Outside-Kerdock signed probes | oracle 0.131; frozen global 1.557 | enormous capacity, nontransferable coefficients | Phase benchmark | analytic per-network coefficient or new signed observable |
| M146 | reported large oracle curve | primary artifacts missing; scalar gate noninvariant | Provisional/non-evidence | complete immutable package and independent reproduction |
| OGAP checkpoint campaign | layer-30 oracle ratios 0.0239 and 0.0252 on frozen and independent confirmation cohorts | large repair channel distributed across checkpoints; increments nearly incoherent | Frozen mechanism evidence | estimate defects legally in a downstream basis |
| M152 | reported 1,100-network corpus | no primary package | Removed | complete corpus, target, splits, predictions and code |

## 5. Reopened paths

### 5.1 Downstream-weighted anchor rescoring

The rescoring campaign confirms that unweighted latent-state precision can discard useful output directions. The best frozen external arm reaches pooled raw ratio `0.851978` across 28 external cases, but p90 is approximately `1.341`, worst case `1.924`, and the rotation-expansion subgroup reverses above one. This is meaningful diagnostic evidence and an unsafe deployable estimator.

### 5.2 Network-derived Poisson controls

A post-hoc exploratory mid-radius dictionary reaches approximately `0.929`, but the frozen terminal result is `1.038`. The tested directions and radii are closed; the existence of analytically integrable high-degree controls remains open.

### 5.3 Exact-mean projected ReLU controls

A candidate initially reaches `0.927` on 16 terminal networks. Without refitting, a 48-network extension reverses to `1.013`; even a favorable implementation model gives adjusted ratio `1.067`. This branch fails both statistical and economic gates.

### 5.4 Outside-support signed probes

Per-network target-labeled coefficients can reduce pooled error to approximately `0.131`, while one frozen global coefficient vector scores `1.557` with catastrophic tails. The correct conclusion is not “signed probes cannot work,” but “the tested weight-derived representation does not transport their signed phase.”


## 6. New oracle-gap and coherence campaign

A challenge-matched synthetic campaign used 12 new width-256, depth-32 base networks, three predetermined rotations per base, and 16 independent complete-Kerdock rotations for reference construction. Four development bases were separated from four frozen validation bases; after exposure, four candidates were frozen and evaluated on four independent confirmation bases. No official protected cohort was opened.

### Checkpoint repairability

Exact mean replacement followed by nonlinear suffix replay gave the following pooled ratios:

| Layer | Frozen validation | Independent confirmation |
|---:|---:|---:|
| 7 | 0.584 | 0.605 |
| 15 | 0.424 | 0.415 |
| 23 | 0.218 | 0.200 |
| 27 | 0.103 | 0.117 |
| 29 | 0.0528 | 0.0539 |
| 30 | 0.0239 | 0.0252 |

This refutes the idea that only a final-layer injection is repairable. Substantial correctable error is already present at middle checkpoints.

### Cross-checkpoint coherence

Successive checkpoint-repair increments contributed approximate energy fractions

`[0.395, 0.177, 0.235, 0.111, 0.0535, 0.0288]`.

Most off-diagonal increment cosines had magnitude below 0.10, with maximum 0.146. The cascade therefore has an approximately incoherent multi-checkpoint repair structure, not equal per-layer contributions and not one purely late injection. This closes the prior missing cross-layer-coherence diagnostic on a new development/validation/confirmation campaign.

### Direction-dependent anchor accuracy

At layer 30, equal Euclidean anchor errors had radically different effects. A half-defect-norm perturbation in the leading downstream-Jacobian direction raised the ratio to 1.575, while twice-defect-norm error in the trailing singular direction left the ratio essentially at the 0.0239 oracle floor. This directly validates downstream-weighted error plus gate-crossing remainder as the correct replacement metric.

### Source-span observability

In one five-source correction span, target-dependent global coefficients reached approximately 0.596 and per-case oracle coefficients 0.255, while frozen deployable combinations were weaker and unstable on confirmation. The source space therefore contains useful directions; the missing object is stable legal instance-specific coefficient information.

## 7. Group-invariant observability

If runtime diagnostics are invariant under a measure-preserving orientation reversal while the candidate correction direction changes sign, their conditional signed alignment is zero under the symmetry model. More generally, the T41 symmetry-defect theorem bounds squared normalized alignment by `((delta_e+delta_c)/2)^2`, where the defects measure imperfect anti-symmetry of the error and imperfect equivariance of the correction. Norms, Gram matrices, disagreement magnitudes and condition numbers can then predict capacity without predicting orientation. The actual WHestBench distribution has not been proved exactly symmetric; grouped cross-fitting should estimate the size of any invariant component rather than assume it vanishes.

## 8. Contamination and grouping rules

- Rotations of one base network are not independent observations.
- Every confidence interval and split must group by base network.
- Exposed Mini-100 or development panels are development only.
- Truth-stream construction and noise correction must be stated per row.
- Post-hoc descendants require a new terminal set.
- A frozen extension without refitting is stronger evidence than another small fresh split selected after inspection.

The accompanying `EMPIRICAL_EVIDENCE_AND_CONTAMINATION.csv` is the canonical chronology table.

## 9. Compute and tail safety

A candidate must report raw MSE, grouped wins, p90, worst case, reference uncertainty, incremental FLOPs, residual wall time and complete adjusted score. Mean gain alone is insufficient. The reopened campaign demonstrates that methods with positive pooled signal can still be unusable because the tails are larger than the mean benefit.

## 10. Claims removed

The article must not say:

- no statistical path exists;
- all legal corrections have zero alignment;
- scalar learning is impossible;
- the anchor tolerance is universally approximately `5e-4`;
- analytically integrable controls are low degree;
- degree-6+ controls cannot help;
- signed weights are globally closed;
- infinite-width harmonic shares are width-256 measurements.

## 11. Reopening map

Further work requires at least one genuinely new ingredient:

1. a legal absolute-phase observable not contained in the tested invariant feature classes;
2. an analytic per-network coefficient for off-support signed probes;
3. a live-degree exactly integrable surrogate with a preregistered oracle ceiling and residual-kernel recertification;
4. finite-width-specific arbitrary-node geometry;
5. a target with substantially higher reliability for learning studies;
6. an orientation-aware odd statistic that breaks a precisely frozen phase-flip symmetry model.

## 12. Conclusion

The experiments show a repeated separation between capacity and observability. Strong oracle directions and average legal signal exist, but the tested methods fail complete deployment because signed phase is unstable, reference labels are noisy, tails are unsafe, or compute dominates. This is a useful and reproducible negative map. It is not a universal no-free-lunch theorem.
