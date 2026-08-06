# Priority 6 — Coherent Companion Orientation Codebook

**Date:** 2026-07-29  
**Verdict:** **Close the tested legal selector architecture; do not promote to official immutable validation.**  
**Positive result retained:** the coherent-orientation oracle ceiling is large and survives a fresh, higher-precision cohort. The blocker is legal orientation identification, not codebook existence or probe cost alone.

## Executive result

An eight-orientation codebook of complete coherent 17-basis companions has a strong research-only ceiling: on 24 untouched cases grouped into eight fresh base networks and three predetermined input rotations each, best-of-eight achieved **0.596 noise-corrected raw ratio**, **0.651 projected adjusted ratio**, **21/24 wins**, and **1.241 worst**.

No legal selector retained that ceiling:

- the preregistered largest-two-basis-companion-norm selector scored **1.008 raw**, **1.103 adjusted**, **9/24 wins**, and **2.502 worst**;
- the paired-probe-norm comparator scored **1.018 raw**, **1.113 adjusted**, **9/24 wins**, and **2.100 worst**;
- a frozen consistency gate with full-129 fallback capped many tails but applied the correction on only **4/24** cases and scored **0.958 raw / 1.067 adjusted**, with **3/24 wins** and **1.241 worst**.

The post-hoc norm signal that motivated the confirmation run therefore **did not replicate** after preregistration and doubled reference precision.

## Protocol and provenance

### Development audit

- 12 fresh width-256, depth-32 synthetic base networks.
- Three orthogonal input rotations per base: 36 grouped cases.
- Eight deterministic coherent orientations, each containing all 17 companion bases.
- Main design: bases 0–110 plus coordinate basis 128.
- Anchor layer: 29; frozen amplitude: 0.20; 128 radial-control probes; ridge 0.1.
- Independent reference: 16 complete-Kerdock rotations per base, split 8+8 for reference-noise estimation.
- Development, calibration, and first test blocks were separated by base network.

### Fresh preregistered confirmation

- Eight new base networks, three rotations each: 24 untouched cases.
- Exact same eight-orientation codebook and frozen amplitude.
- Preregistered before case generation in `VALIDATION_PREREGISTRATION.json`.
- Final reference doubled to **32 complete-Kerdock rotations per base**, four independent groups of eight.
- Mean/p90/worst estimated reference-noise fractions were **3.0% / 4.3% / 5.8%**.
- Development and validation base IDs were disjoint; all rotations remained grouped.
- Repeated execution of a duplicate case was bitwise identical in every stored numeric quantity.

## Frozen validation arms

| Arm | Raw ratio | Projected adjusted | Wins | Median | p90 | Worst | Mean correction cosine | Grouped adjusted 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed orientation r3 | 1.163 | 1.154 | 9/24 | 1.133 | 2.097 | 3.440 | 0.161 | [1.006, 1.319] |
| Largest nested c2 norm | 1.008 | 1.103 | 9/24 | 1.085 | 1.856 | 2.502 | 0.364 | [0.994, 1.220] |
| Largest paired p2 norm | 1.018 | 1.113 | 9/24 | 1.146 | 1.552 | 2.100 | 0.327 | [0.980, 1.264] |
| Consistency gate + full-129 fallback | 0.958 | 1.067 | 3/24 | 1.000 | 1.000 | 1.241 | 0.329 | [0.992, 1.128] |
| Oracle best of 8 | 0.596 | 0.651 | 21/24 | 0.736 | 0.989 | 1.241 | 0.646 | [0.565, 0.762] |

The legal primary fails every promotion requirement: raw ratio, adjusted ratio, wins, worst tail, and grouped interval. The safety fallback improves pooled raw MSE but does so with low coverage and unfavorable compute.

## Oracle codebook frontier

The subsets were frozen from development and always included the preserved r3 orientation.

| Orientations | Oracle raw | Oracle adjusted | Wins | Worst | Fraction of 8-way ceiling captured |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.163 | 1.154 | 9/24 | 3.440 | 0.0% |
| 2 | 0.794 | 0.799 | 16/24 | 2.118 | 65.1% |
| 4 | 0.649 | 0.671 | 19/24 | 1.375 | 90.7% |
| 8 | 0.596 | 0.651 | 21/24 | 1.241 | 100.0% |

Two orientations capture only **65.1%** of the eight-way ceiling and retain a **2.118** worst case. Four orientations capture **90.7%**, but their oracle worst is still **1.375**. Eight orientations are needed for the strongest average and tail ceiling in this cohort.

## Answers to the Priority 6 questions

### Can a paired probe select the correct orientation without estimating the correction accurately?

**No, not with the tested legal summaries.** The paired two-basis direction had mean cosine only **0.219** with its orientation's full 17-basis correction, and its median norm was **0.302×** the full-correction norm. Selecting by paired-probe norm matched the oracle orientation only **8.3%** of the time.

Even a research-only selector allowed to rank orientations by how well each paired probe pointed toward the true repair vector achieved only **0.879 raw**, **10/24 wins**, and **2.330 worst**. Thus the two-basis paired objects are not merely amplitude-noisy versions of a reliable orientation phase signal.

### How many orientations are needed?

Two give meaningful oracle headroom but not enough tail coverage. Four capture about 91% of the eight-way pooled ceiling. Eight are materially better on the worst rotations. Increasing beyond eight is not justified until a selector can recover the existing eight-way ceiling.

### Is orientation preference base-network stable?

**No.** Across the three rotations of each fresh base network, oracle orientation identity agreed pairwise only **16.7%** of the time; the mean modal fraction was **50.0%**. The legal c2 and p2 selectors were similarly unstable. Orientation preference is substantially rotation-specific.

### Does rotation equivariance suggest a deterministic transformation rule?

No simple rule is visible in this codebook. Random orthogonal input rotations largely reshuffle the oracle identity rather than permuting it consistently. A future equivariant rule would need an explicit algebraic action on the codebook; identity or rank stability does not provide one.

### Is the codebook selecting a missing absolute offset?

**Mostly no.** Across orientations, the mean pairwise correction cosine was **0.102**, with p10 **-0.407**. The shared mean of all eight corrections scored **0.853 raw**, well behind best-of-eight at **0.596**. The centered first orientation mode carried a median **52.9%** of orientation variation. This is primarily a direction/phase choice, not selection of one scalar absolute offset.

### Does direction or amplitude create the codebook benefit?

Both matter, but **direction dominates**:

- fixed r3, fixed amplitude: **1.163**;
- fixed r3 with oracle scale: **0.901**;
- best orientation, fixed amplitude: **0.596**;
- best orientation with oracle scale: **0.545**.

Changing orientation removes far more error than rescaling the fixed orientation. Giving the failed c2 selector an oracle amplitude still reaches only **0.840**, confirming that wrong orientation, not only wrong scale, is the primary selector failure.

### Would choosing between two orientations capture most of the ceiling?

No. The frozen two-way codebook captures **65.1%** of the eight-way gain. Four orientations capture **90.7%** and are the smallest plausible ceiling-preserving codebook, but no legal four-way selector transferred.

### Does selector cost erase the gain?

The **oracle** gain survives cost: projected adjusted ratios are **0.799**, **0.671**, and **0.651** for 2/4/8 orientations. The **legal** gain does not: the primary selector is **1.103** adjusted, and the safety package is **1.067**.

### Are there no-headroom cases?

Yes. **3/24** validation cases had best-of-eight ratio at least 1.0. Orientation selection cannot solve these; a deployable package still needs abstention or a full-baseline fallback.

## Why the legal selectors fail

1. **Probe/full-companion mismatch.** Two-basis objects are only weakly aligned with the complete coherent correction.
2. **Rotation-specific preference.** The best identity is not stable within a base network.
3. **Orientation vectors are diverse.** Low pairwise cosine means consensus and common-probe rules often suppress the genuinely useful outlier orientation.
4. **The tail and average require different behavior.** A strict consistency rule can cap the tail only by falling back almost everywhere.
5. **Fallback compute is expensive.** A full-129 fallback after eight two-basis probes has projected cost ratio 1.116, so safe abstention loses adjusted economics.

## Cost and implementation audit

### Exact dense-equivalent trajectory accounting

| Package | Basis-layer units | Dense-equivalent FLOPs | Projected effective compute | Relative cost |
|---|---:|---:|---:|---:|
| Fixed 112+17 | 4,094 | 274.744B | 174.174B | 0.992 |
| Two-orientation selector | 4,154 | 278.770B | 176.726B | 1.006 |
| Four-orientation selector | 4,274 | 286.823B | 181.831B | 1.035 |
| Eight-orientation selector | 4,514 | 302.929B | 192.042B | 1.094 |
| Eight-way probes + full-129 fallback | 4,608 | 309.238B | 196.041B | 1.116 |

These are exact dense trajectory counts mapped to the canonical 175.62B effective baseline. They are **not** official FlopScope subprocess measurements.

### Prototype wall time

Five-repeat dense NumPy/BLAS medians on the available sandbox:

- fixed package: **1.276s**;
- eight-way selector: **1.395s**;
- safety fallback: **1.441s**;
- benchmark peak RSS: **400.7 MiB**.

The full research case generation averaged **6.02s** per candidate case after reference construction, with peak process RSS **1181.9 MiB**. Packaging, startup, immutable-array compatibility, and official residual wall remain unmeasured in the unavailable challenge subprocess.

## Promotion gate

| Requirement | Best legal result | Pass? |
|---|---:|:---:|
| Raw ratio ≤ 0.95 | 1.008 primary; 0.958 safety | **No** |
| Worst ≤ 1.10–1.15 | 2.502 primary; 1.241 safety | **No** |
| Wins ≥ 75% | 37.5% primary; 12.5% safety | **No** |
| Adjusted ratio < 1 | 1.103 primary; 1.067 safety | **No** |
| Grouped interval favors improvement | primary [0.994, 1.220] | **No** |
| Selector retains most oracle gain | 1.008 vs 0.596 raw | **No** |

## Closure and handoff

Close the following tested Priority 6 forms:

- ranking coherent orientations by two-basis companion norm;
- ranking by two-basis paired-probe norm;
- common-probe and consensus geometry;
- nested convergence, first-layer discrepancy, and simple linear/ridge rankers over these features;
- bounded consistency abstention with zero, fixed-orientation, or full-129 fallback;
- codebook expansion as a substitute for selector quality.

Retain the eight-way oracle labels as a target for **Priority 9 tail attribution** or **Priority 10 tiny bounded residual/gate modeling**, but do not spend deployable compute on additional orientations before a new observable predicts orientation benefit across grouped rotations.

## Validation checks

- Exact asset SHA-256: `58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad`.
- Rotation orthogonality max error: `1.718e-08`.
- Sample mutual-unbiasedness max error: `1.085e-08`.
- Duplicate-case maximum numeric difference: `0.0e+00`.
- Preregistration hash matched and predates validation cases: **True**.
- Development/validation base overlap: **none**.

## Scope limitation

This is an exact-geometry, width-256 synthetic investigation using the retained challenge asset and frozen estimator algebra. It is not an official repository/subprocess validation. The statistical selector gate already fails decisively, so the missing official FlopScope measurement cannot rescue the branch.
