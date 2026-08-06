---
title: "WHestBench Independent Anchor and Structured External-Phase Research Report"
date: 2026-07-29
status: "Research synthesis; one promising but tail-unsafe compute-plausible candidate"
challenge: "ARC White-Box Estimation Challenge 2026"
primary_question: "Can a legal independent E[g_31] or joint-scalar anchor recover the dominant layer-31 mean defect at winning adjusted cost?"
---

# WHestBench Independent Anchor and Structured External-Phase Research Report

**Date:** July 29, 2026  
**Research status:** narrowed substantially; no submission-safe winner yet  
**Current leading package:** 112 original Kerdock bases plus a 17-basis independent partial-MUB companion feeding the frozen lower-order radial-Hermite anchor  
**Primary unresolved issue:** catastrophic phase tails, not average signal or compute

## 1. Executive verdict

The campaign established the following facts with high confidence:

1. **The dominant repairable channel is real and extremely large.** An exact layer-31 lower-order radial-Hermite anchor can reduce final MSE to roughly **0.09–0.10 times baseline** on full-width test cohorts. Earlier exact layer-31 replay evidence independently showed a roughly **4.57-fold** raw gain from correcting the true layer-31 mean.
2. **The connected-cubic term is not the immediate blocker.** For the frozen 128-probe construction, the exact lower-order center channel captures essentially all of the useful exact-anchor gain; connected-only corrections are neutral or harmful.
3. **Kerdock covariance is already good enough.** Replacing oracle covariance with the free full-Kerdock covariance changes the exact lower-anchor result negligibly. The observed relative covariance error was about **0.28%**, and oracle mean plus free covariance retained a pooled candidate/base ratio near **0.0874**.
4. **The unsolved object is the tiny absolute layer-31 mean defect**, or its 128 downstream-weighted contractions:

   \[
   \delta_{31}=Q_K(a_{31})-\mathbb E[a_{31}].
   \]

5. **Generic analytic closures fail on sign and scale.** Deterministic Gaussian closure, particle Gaussian ADF, Edgeworth-style recurrences, same-cloud empirical-shape recentering, and smooth shared depth profiles do not recover the absolute phase reliably.
6. **Generic learned phase prediction is closed for now.** A true width-256 grouped-rotation equivariant proxy had a large oracle support ceiling but negative learned phase alignment and score-negative replay economics.
7. **Independent complete designs are unbiased but converge too slowly.** Even 32 complete rotated companions gave only about **0.844 times baseline MSE**, despite an exact-anchor ceiling near **0.103**.
8. **The first compute-plausible candidate is a structured partial-MUB companion.** A frozen 112+17 package achieved **0.9427 times raw MSE on eight untouched validation networks** and approximately **0.935 times adjusted candidate/base before small anchor overhead**, but its worst network was **1.482 times baseline**.
9. **Paired replacement differences are much safer but weaker.** A 17-basis paired probe achieved about **0.959 times raw MSE, 10/12 wins, and 1.036 worst**, but costs too much. A two-basis version is close to adjusted break-even and therefore may be useful as a safety signal.

The most credible path toward a winning executable is now:

> Keep the 112+17 partial-MUB absolute anchor as the high-headroom estimator, and use a tiny 1–4-basis paired replacement probe only to detect disagreement, suppress, or replace dangerous corrections.

This is a much narrower problem than estimating the full mean, learning phase from weights, or reconstructing connected third moments.

---

## 2. Evidence-status convention

Because this campaign used multiple cohorts and several experiments were exploratory, every result in this report belongs to one of four evidence classes:

- **Frozen validation:** configuration, scale, and split fixed before opening the stated validation cohort.
- **Grouped development/calibration:** useful for mechanism and candidate selection, but exposed and not a final claim.
- **Actual-width synthetic proxy:** width 256 and depth 32 with real Kerdock constructions, but not the official hidden grader population or public high-precision corpus.
- **Incomplete / oracle diagnostic:** per-network tuning, partial cohorts, or experiments stopped after decisive failure. These establish mechanism or ceilings only.

No oracle-tuned result is promoted as deployable.

---

## 3. Canonical context and prerequisites

The canonical v2 experiment plan made the width-256 learned predictor conditional on first freezing one low-dimensional label from the anchor branches. The prerequisite label could be:

- joint-anchor residual functionals;
- sparse \(g_{31}\) residual coefficients; or
- V80 correction sign, scale, and confidence.

The available launch pack did not contain the official high-precision corpus or the missing frozen 128-probe arrays. Therefore, the campaign used two complementary environments:

1. a full-width/depth synthetic proxy for learned phase prediction; and
2. the complete 66,048-row Kerdock construction for exact-anchor, independent-expectation, and structured companion experiments.

Canonical Library artifacts referenced during the campaign:

- `whestbench_canonical_research_ledger_20260729_merged_v2.xlsx` — `file_000000008c74822f81b546678c590808`;
- `WHestBench_Experiment_Prompt_Pack_20260729_v2.md` — `file_00000000cd54822f878b47530f890d3d`;
- `EQUIVARIANT_WEIGHT_MODEL_RESEARCH_20260729.md` — `file_00000000b9a8820cafe11b4e1fe38f01`;
- `equivariant_weight_model_repro_20260729.zip` — `file_00000000af48822f9d5f3bb43fb1b6e7`;
- `WHestBench_Experiment_Launch_Pack_20260729_v2.zip` — `file_00000000c83081f5888ad228b3648779`.

---

## 4. Problem formulation

Let the bias-free width-256, depth-32 ReLU network have layer-31 activation \(a_{31}(x)\). The complete Kerdock quadrature estimate is

\[
m_K=Q_K(a_{31}),
\]

while the Gaussian target is

\[
\mu=\mathbb E[a_{31}(X)],\qquad X\sim\mathcal N(0,I).
\]

The required mean defect is

\[
\delta=m_K-\mu.
\]

The frozen radial-Hermite control constructs a low-dimensional surrogate \(g_{31}\) from 128 probes and applies the residual-control identity

\[
\widehat\mu_{31}=\widehat{\mathbb E[g_{31}]}+Q_K(a_{31}-g_{31}).
\]

For the lower-order center channel, the correction is determined by the small center displacement \(\delta\), a covariance-like coefficient matrix, and the 128-probe downstream regression map. The full 256-dimensional mean does not need to be estimated accurately in Euclidean norm; only its downstream-sensitive contractions matter.

### Evaluation metric

For every candidate, the primary raw metric is

\[
R=\frac{\operatorname{MSE}(\text{candidate},\text{truth})}
        {\operatorname{MSE}(\text{baseline},\text{truth})}.
\]

- \(R<1\) is a raw improvement.
- Final decisions must include propagation, anchor, replay, and inference cost.
- Tail metrics are mandatory because many apparently strong average estimators reverse sign on a minority of networks.

---

## 5. Core mechanism: exact anchor reproduction and decomposition

### 5.1 Exact sparse radial-Hermite anchor

The complete Kerdock reproduction succeeded immediately. On the first width-256 network, the frozen 128-probe exact anchor reduced final MSE to **0.0416 times baseline**.

On an eight-network full-design screen:

| Estimator | Pooled candidate/base | Wins |
|---|---:|---:|
| Exact joint anchor | **0.0928** | 8/8 |
| Full-depth Gaussian recurrence | 70.5 | 0/8 effectively |
| Best inherited Gaussian checkpoint | 31.9 | unsafe |
| Independent 4,096-point source | 2.75 | unsafe |

The exact mechanism therefore has far more than enough oracle ceiling. The challenge is estimating its absolute expectation, not finding a useful representation.

### 5.2 Lower-order versus connected-cubic channels

The anchor was decomposed into:

1. a lower-order center term involving the target mean and second moments; and
2. a connected-cubic correction.

Across the first screened networks, the **exact lower-order center channel alone captured essentially all of the exact-anchor gain**. The connected channel was small, neutral, or harmful. This agrees with earlier canonical component ablations in which lower-only helped materially while connected-only exceeded baseline error.

**Decision:** connected-cubic reconstruction is diagnostic, not the current production bottleneck.

### 5.3 Covariance is solved to useful precision

Replacing oracle covariance with the free Kerdock covariance had negligible effect:

- relative covariance error: approximately **0.2–0.3%**, pooled figure **0.28%**;
- oracle mean plus free covariance: **0.0874 times baseline** on the eight-network ablation;
- exact center-only anchor remained near the complete exact-anchor result.

This sharply narrows the task to the mean defect \(\delta\). Future methods should not spend major compute on full covariance or connected \(K_3\) reconstruction unless new evidence overturns this ablation.

---

## 6. Broad learned external phase prediction

A separate actual-width/depth proxy tested whether weights and ordinary baseline observables could predict 8–32 signed downstream-sensitive layer-31 coefficients.

### 6.1 Protocol

- width 256, depth 32;
- 144 base networks: 96 train, 16 calibration, 32 untouched validation;
- identity plus one grouped Haar rotation per base;
- frozen top-32 or top-8 downstream-sensitive directions;
- invariant ridge control;
- one 26,113-parameter bidirectional permutation-equivariant DWS-style model;
- exact final replay scoring;
- coefficient, correction-cosine, tail, and bootstrap diagnostics.

### 6.2 Results

| Target | Oracle projected ratio | Ridge | Equivariant DWS | DWS correction cosine |
|---|---:|---:|---:|---:|
| 32 modes | **0.528799** | 0.996410 | **1.002308** | **-0.0978** |
| 8 modes | **0.841705** | 0.999987 | **1.001258** | **-0.1105** |

The 32-mode oracle won on all 32 validation bases, but the model had negative coefficient correlation and negative downstream correction alignment. A confidence gate improved ridge to **0.996054 raw**, but the unavoidable replay moved it to **1.02718**, before model cost.

A positive control using an observable equivariant coefficient target reached validation correlation **0.504** and MSE **0.699 times the mean-predictor baseline**, showing the pipeline itself could learn a real equivariant signal.

**Closure:** do not spend another broad run on larger generic DWS models, more rotations, capacity sweeps, direct sign classifiers, or confidence gates. Learning may reopen only on the residual of a strong legal analytic anchor.

---

## 7. Analytic and recurrence-based expectation sources

### 7.1 Particle Gaussian ADF and checkpoint recurrences

The initial full-depth Gaussian assumed-density recurrence propagated a Gaussian approximation layer by layer with sampled moment integration. It failed catastrophically:

- full-depth recurrence: **70.5 times baseline**;
- best inherited Gaussian checkpoint: **31.9 times baseline**.

Increasing independent source sample count reduced noise, and individual 8k–16k pilots occasionally reached **0.36–0.70 times baseline**, but the optimal amplitude varied sharply across networks. One frozen global amplitude collapsed toward zero.

### 7.2 Deterministic full-covariance Gaussian closure

To separate closure bias from particle noise, a deterministic Gaussian closure used exact pairwise ReLU moments and propagated full covariance without trajectory sampling.

It still failed: the predicted mean-defect magnitude was roughly **4–9 times too large**, and the sign reversed across networks. The Gaussian approximation itself—not particle noise—was wrong for the tiny finite-width quadrature defect.

### 7.3 Edgeworth and same-cloud non-Gaussian recurrences

Edgeworth-style mean corrections, marginal particle propagation, and local same-cloud shape corrections were tested. They were occasionally directionally useful but not stable.

A full empirical marginal-shape recentering recurrence, which preserved the complete observed non-Gaussian marginal shape while removing the sample center before each ReLU, achieved:

- pooled ratio: **0.984**;
- wins: 4/8;
- worst: **1.032**.

This was safe but far below winning scale.

### 7.4 Full-depth source localization

Exact signed source decomposition showed the mean defect is not a short-suffix phenomenon:

- the final 12 layers explain only about one-third to one-half of the downstream-sensitive anchor;
- roughly 24 layers are needed for approximately **76–85%** cumulative signed recovery, depending on the contraction metric;
- this remains below the desired 90% retention threshold.

A same-cloud local-shape source recurrence gave a pooled result around **0.997**, 4/8 wins, and mean correction cosine **0.093**.

### 7.5 Smooth target-contracted depth profiles

A 24-network source matrix exposed all 30 layerwise source contractions separately. Only smooth, low-dimensional shared depth profiles were fit; unrestricted 240-feature regression was intentionally excluded.

The grouped fit selected a heavily regularized second-moment profile, calibration reversed its sign, and untouched validation retained only **0.14% of the exact-anchor benefit**, with correction cosine **0.021**.

**Closure:** insufficient depth resolution is not the problem. The observable layerwise sources do not have a stable shared signed profile across networks.

---

## 8. Independent complete-design expectation estimation

### 8.1 Complete rotated Kerdock companions

Each Haar-rotated complete Kerdock design is an unbiased angular estimator for a positively homogeneous bias-free ReLU network. The design radius was already chosen as \(E\|X\|\), so no radial correction was missing.

On eight untouched networks with the frozen shrinkage law, the indirect layer-31 anchor produced:

| Number of complete companions | Pooled ratio | Wins | Worst |
|---:|---:|---:|---:|
| 1 | 1.089 | 4/8 | 1.384 |
| 2 | 0.866 | 5/8 | 1.245 |
| 4 | **0.822** | 6/8 | 1.239 |
| 8 | 0.836 | 6/8 | 1.262 |
| 16 | 0.845 | 6/8 | 1.226 |
| 32 | 0.844 | 6/8 | 1.187 |

The exact-anchor ceiling on the same network block was roughly **0.103**. Thirty-two complete companions therefore recover only a modest fraction of the available benefit.

### 8.2 Direct final-output averaging

The same complete companions can estimate the final output directly. Results were nearly identical to the indirect anchor:

- four companions: approximately **0.814** pooled;
- eight companions: approximately **0.823** pooled.

This proves the radial anchor is not materially amplifying independent-design noise. The companion mean itself converges too slowly in high-dimensional angular phase.

### 8.3 First-layer balancing

The exact Gaussian first-layer ReLU mean was used as a free rotation-quality control. Candidate rotations were selected by raw or standardized first-layer error and by linear transport of that error.

Development and validation reversed. Some point estimates reached approximately **0.78**, but validation worst cases ranged from **1.55 to 2.15**. The first-layer control is correlated with downstream quality but cannot safely rank rotations across networks.

### 8.4 Spherical Sobol

Normalized spherical Sobol directions were tested to remove irrelevant radial variance exactly. On the initial full-width screen:

- 65,536 spherical Sobol points: **3.32 times baseline**;
- ordinary Gaussian Sobol: **1.60 times baseline**.

High-dimensional angular integration, not radial noise, is the dominant difficulty.

---

## 9. Independent empirical-shape sources

Companion designs were also used only for their centered layerwise marginal shapes, not their absolute means.

The incomplete four-companion screen produced:

| Network | Exact-anchor ratio | Four companion shapes |
|---:|---:|---:|
| 24 | 0.0682 | **0.9868** |
| 25 | 0.1032 | **0.4110** |
| 26 | 0.1006 | **1.2421** |

Network 25 showed that independent shape can carry real phase information, with mean-defect and correction cosines near **0.77**. Networks 24 and 26 showed severe heterogeneity and sign reversal.

**Status:** incomplete and not promotable. The existing three-network evidence is already too unstable to justify a large expansion unless the source is paired with a strong observable safety statistic.

---

## 10. Analytically zero-mean spherical-harmonic controls

The Kerdock rule is exact through degree five. Therefore, degree-6 and higher spherical harmonics are the first exact-zero-mean controls capable of representing the missing cubature phase without external trajectories.

### 10.1 Generic zonal controls

Degree 6, 8, and 10 zonal Gegenbauer features were constructed along sensitive directions and fit by crossfit regression.

A frozen final-output control achieved **0.980 times pooled MSE on 12 development networks**, 7/12 wins, but reversed on untouched validation:

- validation pooled ratio: **1.034**;
- all 8 validation networks harmed;
- correction cosine changed from positive in development to negative in validation.

Estimating the 128 radial-feature expectations was stronger locally: individual networks reached **0.86–0.97**, and mean-defect cosine reached approximately **0.47, 0.32, 0.77, and 0.58** on the first four networks. However, the transferable gain remained only a few percent.

A 12-source bank of four direction systems crossed with degrees 6, 8, and 10 retained only about:

- **1.2% of exact correction energy**;
- **2.3% of final-MSE benefit**.

### 10.2 Mixed degree-6 interactions

Products of degree-3 modes along pairs of sensitive directions were centered using their analytic spherical expectation. A target-probe pair reached approximately **0.968** on one network, but other direction systems reversed.

### 10.3 Rule-specific defect modes

The degree-6 moment defect of the Kerdock rule itself was optimized without using network outputs. This produced 18 nearly orthogonal fixed modes with normalized design bias around **0.10**, versus about **0.006** for generic directions—a roughly **17-fold stronger design defect**.

These modes produced the strongest harmonic point result:

- one network: **0.619 times baseline**;
- correction cosine: **0.905**;
- no extra trajectories.

However, other networks were near neutral. Degree 6, 8, and 10 each won on different networks, and fold agreement did not predict the winning degree. A stability/abstention selector either abstained everywhere or selected harmful corrections.

### 10.4 Rotation heat extrapolation

Small correlated rotations were used to damp spherical-harmonic errors at known degree-dependent rates and extrapolate to the Haar limit. Multi-degree fits were ill-conditioned; a bounded degree-6 extrapolation used approximately \(-0.96\) baseline plus \(1.96\) smoothed mean.

It failed on the first network by amplifying a wrong-direction correlated error. The route was closed before an expensive eight-design package.

**Harmonic conclusion:** exact-zero-mean controls contain real phase signal, sometimes very large, but the active defect degree and orientation are network-specific. No observable stability statistic yet identifies the safe mode.

---

## 11. Compute-neutral direct-pilot allocation

A direct compute trade removed 17 original Kerdock bases and spent the savings on ordinary independent pilots.

Individual networks occasionally improved sharply—for example, approximately **0.459 times baseline with 4,096 pilot points**—but another network preferred no anchor. Across 12 development networks, the best fixed package effectively chose no direct anchor.

The reduced 112-basis radial control alone achieved approximately:

- pooled ratio: **0.988**;
- worst: about **1.17**.

Ordinary trajectories therefore do not use the removed basis budget efficiently.

---

## 12. Structured partial-MUB companion: leading candidate

### 12.1 Construction

The main design retains 112 of 129 original Kerdock bases:

- original basis IDs 0–110;
- coordinate basis 128.

The 17 removed full-depth bases are replaced, for anchor estimation only, by 17 bases from one independent Haar-rotated Kerdock family. The companion is propagated only to the anchor layer. The final output is corrected directly through the frozen radial-Hermite regression map; no additional final replay is required.

The frozen companion family was:

- basis geometry: first 17 chirp bases (`chirp17`);
- deterministic companion replicate: third fixed seed family (`r3`, corresponding to the third preregistered rotation in the implementation);
- global anchor amplitude: **0.20**.

### 12.2 Frozen results

| Cohort | Networks | Pooled ratio | Wins | Worst |
|---|---:|---:|---:|---:|
| Training | 12 | **0.9336** | 7/12 | 1.191 |
| Calibration | 4 | **0.9634** | 3/4 | 1.001 |
| Untouched validation | 8 | **0.9427** | 4/8 | **1.482** |
| All exposed | 24 | **0.9436** | 14/24 | 1.482 |

This is the first candidate with both substantial raw gain and plausible compute economics.

### 12.3 Compute accounting

Using basis-depth accounting,

\[
\text{propagation ratio}
=\frac{112\cdot32+17\cdot30}{129\cdot32}
\approx0.9918.
\]

Thus the package slightly reduces propagation relative to the full 129-basis baseline while adding only small radial-anchor arithmetic. Combining the raw **0.9427** validation ratio with the **0.9918** propagation factor gives an approximate adjusted candidate/base ratio of

\[
0.9427\times0.9918\approx0.935,
\]

before small anchor overhead.

### 12.4 Failure mode

The average economics are winning-scale, but the validation tail is unacceptable:

- only 4/8 validation wins;
- worst ratio **1.482**;
- the bad networks are sign reversals, not merely small no-headroom cases.

The candidate must not be submitted without a target-free safety mechanism.

---

## 13. Multi-rotation partial-MUB diversification

The same 17-basis companion budget was split across 2, 4, 8, or 17 independent rotations to reduce phase variance without adding rows.

Individual networks often improved dramatically under different split counts, with several ratios around **0.49–0.72**. However, no fixed split transferred. Across the exposed 24-network pool, the best fixed diversified construction was approximately **0.998 times baseline**.

Interpretation: spreading bases across orientations reduces phase variance but destroys some of the low-degree cancellation and coherent MUB geometry within one complete family.

**Decision:** retain one coherent 17-basis family for the absolute anchor; do not diversify the entire budget blindly.

---

## 14. Paired replacement probes

### 14.1 Mechanism

Rather than estimate an absolute companion mean, compare a small set of original basis blocks with rotated replacements. For a block set \(S\), form the correlated difference

\[
d_S=\frac{|S|}{129}\left(\bar a_S^{\mathrm{orig}}-\bar a_S^{\mathrm{rot}}\right).
\]

The common subset bias cancels. This directly estimates whether replacing those blocks moves the full design toward or away from the Gaussian mean.

### 14.2 Seventeen-basis paired probe

Small rotations and Haar replacements were tested on fixed block geometries. Several networks reached **0.42–0.79 times baseline**, and the pooled moderate-shrinkage result was:

- raw ratio: **0.959**;
- wins: **10/12**;
- worst: **1.036**.

This is much safer than the absolute partial companion, but 17 extra anchor-layer bases cost about 12% of baseline propagation and erase the score gain.

### 14.3 Compute frontier

Block counts 1, 2, 4, 8, 12, 17, 24, and 32 were evaluated with a frozen scale grid.

Key pooled result:

- **two paired bases:** raw ratio **0.9816**;
- extra propagation: approximately

  \[
  \frac{2\cdot30}{129\cdot32}=1.45\%;
  \]

- approximate adjusted ratio: **0.996** before tiny anchor arithmetic.

A safer scale gave:

- raw ratio **0.9847**;
- 8/12 wins;
- worst **1.064**;
- approximately neutral adjusted score.

The paired frontier is therefore not a standalone winner, but it is cheap enough to serve as a **phase-disagreement or safety probe**.

---

## 15. Adaptive basis selection

Observable block scores were constructed from:

- layer-31 block-mean deviation;
- final-output block deviation;
- radial-feature deviation;
- radial-control leverage;
- a standardized combined score.

For each score, the top 1, 2, 4, or 8 blocks were selected without target access and compared with one fixed independent orientation.

Only network 0 was completed before the run ended. Oracle scale sweeps on that network included:

- one feature-selected block: **0.7986**, scale \(-8\);
- four mean- or final-selected blocks: **0.9026**, scale \(-1\);
- two mean-selected blocks: **0.9495**, scale \(-2\).

These are **not deployable results** because the scale was oracle-selected and the strongest sign was negative. One network cannot establish transfer.

**Status:** incomplete. This is the immediate experiment to continue only within a strictly preregistered safety framework.

---

## 16. What is closed

The following tested hypotheses should not be reopened without materially new information:

1. broad weight-to-phase DWS prediction;
2. reducing the learned target from 32 to 8 raw phase coefficients;
3. generic confidence gates on learned correction norm;
4. Gaussian ADF, deterministic Gaussian closure, or simple Edgeworth mean recurrence;
5. short-suffix source truncation;
6. one global smooth depth profile over observable source contractions;
7. ordinary 4k–16k independent trajectories as a compute-neutral anchor;
8. many complete Haar-rotated designs as a practical estimator;
9. first-layer-error rotation selection;
10. normalized spherical Sobol directions;
11. same-orientation empirical-shape recentering as a standalone source;
12. generic degree-6/8/10 harmonic controls with a fixed global configuration;
13. fold-stability selection among harmonic candidates;
14. small-rotation heat extrapolation;
15. splitting the entire 17-basis companion budget across many rotations.

A failed tested representation does not prove mathematical impossibility. It does show that further sweeps in the same family are poor uses of competition compute.

---

## 17. Live hypotheses

Only three hypotheses remain active:

### H1. Partial-MUB plus tiny paired safety probe

The high-headroom 112+17 absolute companion is retained when a 1–4-basis paired probe agrees with its correction direction. On disagreement, either suppress the absolute correction or substitute the safer paired correction.

### H2. Target-free adaptive block selection for the safety probe

Select the 1–4 paired blocks using only baseline block leverage or radial-feature deviation. The objective is not to maximize oracle gain; it is to detect the catastrophic-tail networks at minimal cost.

### H3. Residual learning only after H1 is frozen

If H1 yields a safe legal anchor with positive adjusted score, a very small invariant or equivariant predictor may estimate only its residual scale or abstention probability. Broad phase prediction remains closed.

---

## 18. Recommended preregistered next experiment

### 18.1 New immutable cohort

Generate at least **32 genuinely new width-256/depth-32 base networks** not overlapping any exposed network seeds.

Freeze:

- 16 development bases;
- 8 calibration bases;
- 8 final validation bases;
- all rotations, basis IDs, target layers, and reference seeds before generation.

Do not inspect final validation until one package is frozen.

### 18.2 Fixed estimators

Evaluate exactly these candidates:

1. full 129-basis baseline;
2. 112-basis reduced radial control without external source;
3. frozen 112+17 `chirp17_r3`, amplitude 0.20;
4. two-basis paired probe, fixed first-two chirp blocks;
5. two-basis paired probe, target-free leverage-selected blocks;
6. four-basis paired probe, target-free leverage-selected blocks;
7. one hybrid safety rule selected on calibration only.

No additional basis families, rotation grids, or scale sweeps should be added after validation begins.

### 18.3 Safety features

The hybrid rule may use only observable quantities:

- cosine between the absolute partial-MUB correction and paired correction;
- ratio of correction norms;
- paired correction fold/rotation agreement;
- main-design basis-block stability;
- radial-feature leverage of selected blocks.

A simple preregistered family is:

\[
\widehat c=
\begin{cases}
0.20\,c_{17}, & \cos(c_{17},p_k)\ge \tau_1\text{ and }r\in[r_-,r_+],\\
\beta p_k, & \cos(c_{17},p_k)<\tau_1\text{ and paired stability}\ge\tau_2,\\
0, & \text{otherwise.}
\end{cases}
\]

Here \(c_{17}\) is the frozen absolute 17-basis correction, \(p_k\) is the 2- or 4-basis paired correction, and \(r=\|p_k\|/\|c_{17}\|\). Thresholds and \(\beta\) must be selected only on calibration.

### 18.4 Promotion gate

Promote only if untouched validation shows all of:

- raw pooled ratio **<= 0.95**;
- positive adjusted score after exact propagation and anchor accounting;
- at least **75% wins**;
- worst candidate/base **<= 1.10**, preferably <= 1.05;
- positive mean correction cosine;
- no single network contributes more than 25% of total gain;
- no split or rotation family dominates the result.

Failure should close the tested safety-rule family, not the 112+17 absolute mechanism itself.

---

## 19. Implementation notes

### 19.1 Full Kerdock geometry

- 129 basis blocks;
- 512 antipodal rows per block;
- 66,048 total rows;
- chirp/Walsh generation through FWHT;
- coordinate basis retained explicitly.

### 19.2 Partial companion implementation

`partial_companion_anchor.py` implements:

- fixed 17-basis families: `chirp17`, `spread17`, and `coord16`;
- four fixed rotation seeds;
- matched ordinary direct controls at 8,192 and 16,384 points;
- 112-basis main design using basis IDs 0–110 plus coordinate basis 128;
- frozen 128-probe radial feature construction;
- six-fold basis-block crossfit ridge;
- global amplitude grid.

### 19.3 Paired probe implementation

`paired_replacement_probe.py` implements:

- coherent 17-basis geometries;
- two Haar rotations;
- small relative rotations at scales 0.12, 0.24, and 0.48 with positive/negative signs;
- antithetic small-rotation averages;
- paired difference scaling by \(|S|/129\);
- exact lower-order anchor application.

`paired_block_frontier.py` evaluates block counts 1, 2, 4, 8, 12, 17, 24, and 32.

`paired_adaptive_blocks.py` evaluates observable block-selection statistics for 1, 2, 4, and 8 blocks. Only network 0 was completed.

### 19.4 Memory and performance engineering

Repeated complete-design rotations exposed a long-lived Python allocator problem. The exact companion mean was rewritten as a chirp-batched streaming transform holding about 4,096 trajectories at once. The streaming result matched the full allocation to roughly **4e-18 relative error** and was faster. Eight rotations per fresh process were used to guarantee allocator release.

This engineering detail matters for any continuation: process boundaries and cached immutable Kerdock chirps should be retained.

---

## 20. Reproducibility manifest

Available surviving artifacts and SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| `ACTUAL_WIDTH_EXTERNAL_PHASE_REPORT.md` | `5bfbaf5de1e3cd6e41233fd4d0d105fe6794ae824b353b8a4f9125324377b75b` |
| `RESULT_SUMMARY.json` | `6173bd74871631fa9ffef6b65c4ec30cddb2d33cf31a523343093c89a606fdfe` |
| `final_companion_screen8.json` | `fca13a83c92817ec1369e20d8c9a2a3564e8d2c5e98d31c625c59b4bb592d107` |
| `first_layer_selection_8.json` | `4f9af60e7e5b2741e21693ec026615e13dbc6a6a3befd06340467f29b98fbe75` |
| `cross_shape_net24.json` | `bac32e914f6f3b5bb842a556fc2c3d191e4e0b90bd4798d7c6ba75f252f37806` |
| `cross_shape_net25.json` | `16a169f022208e37425f689d42a1b7e50b25121069566fc331cb31a808370784` |
| `cross_shape_net26.json` | `86cdaeda16cc4f2dc49d0b701c998d8a1971bdd10954c5dac2e9d6008ec77b5f` |
| `partial_companion_anchor.py` | `5222445296de668d65aeeb3011cf47e72c7d7658e353bd1de137ea754567423b` |
| `multi_rotation_partial_anchor.py` | `42accb7eddf31b0cb2617b8172b1f252ab26de68e11f7a552ba75fbdceb3edb3` |
| `paired_replacement_probe.py` | `22939c78c34d5968b69b08e092f7e1f2ea3e31d0869293eaca84e39970a90f21` |
| `paired_block_frontier.py` | `f25f10df69ad9d682b754efbfe8157d54615585ac24d6bed3de659f6806e9929` |
| `paired_adaptive_blocks.py` | `9ef63a7045b6ac99c1e896ca725b8fe73a96db1e96b7e5f24dc5fbfd8f6dddc8` |
| `adaptive_net0.json` | `547a994985f7b9cb11a1a185d5112f434db26b8008f93b29a4211504cf36d528` |

Some intermediate full-design arrays and result matrices existed only in the execution workspace and were not preserved as standalone files. Their reported aggregate results are included here because they were produced and checked during the campaign, but they should be treated as conversation-derived evidence until independently reproduced from the surviving scripts or canonical launch pack.

---

## 21. Scope limitations

1. Much of the campaign uses actual-width/depth synthetic networks, not official hidden grader networks.
2. Several cohorts are now exposed and must not be relabeled as fresh holdout.
3. The exact 128-probe anchor is an oracle mechanism unless its expectation source is legal and independently computed.
4. Some strong point results used oracle amplitude sweeps and are diagnostic only.
5. The leading 112+17 package has a genuine untouched validation block, but only eight networks; its tail remains too large for promotion.
6. The adaptive paired-block experiment has only one completed network and no transfer evidence.
7. Exact production FLOP and wall-time accounting must be measured inside the final subprocess package before submission.

---

## 22. Final research decision

The campaign should no longer be described as searching broadly for an independent \(E[g_{31}]\). The problem is now much narrower:

- the exact lower-order anchor works;
- covariance is free;
- the mean defect contains structured external phase;
- a coherent 17-basis partial-MUB family captures enough of that phase for winning average economics;
- a tiny paired difference captures safer local phase at near-break-even cost;
- the missing component is a **target-free tail gate** connecting those two estimators.

The next experiment should therefore be one bounded safety-layer study on a genuinely new cohort. It should not reopen Gaussian closure, generic harmonics, complete companion averaging, ordinary pilots, or broad phase learning.

**Current candidate status:** promising mechanism, positive average adjusted proxy, not submission-safe.  
**Current go/no-go criterion:** preserve approximately 0.95 raw pooled MSE while reducing worst candidate/base below 1.10 using no more than four paired anchor-layer bases.
