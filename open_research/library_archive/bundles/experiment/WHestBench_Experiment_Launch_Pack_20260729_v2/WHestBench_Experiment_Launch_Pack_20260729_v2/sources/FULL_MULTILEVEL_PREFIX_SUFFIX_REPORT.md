# Full multilevel prefix/suffix estimator — research report

**Date:** 2026-07-29  
**Challenge:** ARC White-Box Estimation Challenge 2026  
**Scope:** Architecture-matched synthetic experiments and reuse of the project's frozen Kerdock/closure harnesses. These are not official Mini-100 results.

## Executive conclusion

The generic full multilevel prefix/suffix idea does **not** survive in its current stable-gate or Gaussian-closure forms.

Three increasingly targeted tests were run:

1. **Stable-gate suffix MLMC:** the residual is tiny pointwise, but allocating arbitrary Kerdock basis blocks across levels destroys the complete-design cancellation. The best two-level i.i.d.-variance projection suggested a 3.4% ideal gain, while the actual coupled block estimator was **26.5% worse** than complete Kerdock on the screen network.
2. **Depth-local layer-31 residual control:** a homogeneous Gaussian closure found a 10.9% raw gain on the four-network discovery half, then reversed to **12.3% worse** on the frozen four-network validation half. The correction direction changes sign by network.
3. **Independent pilot coefficient estimation:** 4–8 independent pilot rotations did not recover the network-specific sign. The frozen selected rule was **1.5% worse at layer 31** before final replay, while adding about 18.5B proxy FLOPs.

The useful conclusion is narrower and stronger:

> Keep stable-gate compilation as arithmetic optimization, not statistical MLMC. Reframe the multilevel branch around the **layer-31 mean defect**, using a substantially better surrogate such as adjoint-compressed connected-K3/checkpoint defect transport. Do not build more levels until a single corrected level has positive frozen residual correlation and adjusted score.

## 1. Estimator tested

The conventional hierarchy is

\[
\widehat\mu
=Q_{N_0}(g_0)
+\sum_{\ell=1}^{L-1} Q_{N_\ell}(g_\ell-g_{\ell-1})
+Q_{N_L}(f-g_{L-1}).
\]

For independent samples and known level variances/costs, the continuous optimum is

\[
N_\ell\propto\sqrt{V_\ell/C_\ell}.
\]

The critical complication here is that a complete 129-basis Kerdock rule is not merely 129 exchangeable Monte Carlo blocks. Its low-degree cancellations are global. A variance allocation that uses 133 surrogate blocks but only 4 exact blocks can look attractive under block-variance algebra while losing the geometry that makes the baseline strong.

## 2. Experiment A — stable-gate suffix hierarchy

### Construction

- Width 256, depth 32, He-initialized bias-free ReLU MLP.
- Complete Kerdock points grouped into 129 antipodal orthonormal-basis blocks.
- Cheap suffixes classify coordinates from a balanced pilot as stable-off, stable-on, or kink.
- Stable-off paths are dropped; stable-on paths are propagated linearly; kink coordinates retain ReLU propagation.
- Tested suffix depths 4, 8, 12, and 16 and multiple rare-switch thresholds.
- Exact residuals were coupled on nested basis blocks.
- Fixed budget: 129 × 32 layer-block equivalents, including pilot cost.

### Best apparent two-level frontier

| Quantity | Result |
|---|---:|
| Suffix | 4 layers |
| Rare threshold | 16 |
| Mean output correlation | 0.999438 |
| Residual block variance / exact block variance | 0.000564 |
| Cheap cost | 30.163 layer-equivalents/block |
| Exact residual cost | 4.0 layer-equivalents/block |
| Continuous/integer allocation | 135 cheap, 9 residual blocks |
| Ideal i.i.d.-block variance ratio | **0.9656** |

This looks like a 3.4% gain, but it is an i.i.d.-block calculation.

### Actual coupled estimator screen

The most conservative tested two-level rule used a four-layer suffix and a stricter rare threshold of 3:

| Quantity | Result |
|---|---:|
| Allocation | 133 cheap, 4 exact residual blocks |
| Baseline proxy MSE | 3.6567e-7 |
| Candidate proxy MSE | 4.6239e-7 |
| Candidate / baseline | **1.2645** |
| Median candidate / baseline | **1.5077** |

Three- to five-level hierarchies were also worse, with mean ratios from **1.2598 to 1.3373**.

### Interpretation

The stable-gate surrogate is excellent *pointwise*, but the statistical opening is mostly illusory:

- A four-layer approximation still pays roughly 28–31 of 32 layers per coarse block.
- Its low residual variance only just compensates for that weak cost separation under i.i.d. assumptions.
- Arbitrary residual-block allocation sacrifices complete-Kerdock cancellation.
- Requiring a complete exact Kerdock residual stream collapses the construction back to the existing suffix compiler: nearly unchanged statistics, modest arithmetic savings.

**Decision:** close stable-gate MLMC as a statistical estimator. Retain the compiler as an arithmetic layer.

## 3. Experiment B — layer-31 residual estimator

A newer oracle attribution result indicates that correcting the layer-31 post-ReLU mean and replaying only the final layer is the dominant repairable channel. This motivated the depth-local estimator

\[
\widehat\mu_{31}
=Q_K(a_{31})+\beta\left(E[g_{31}]-Q_K(g_{31})\right).
\]

The actual layer-31 particle cloud is translated coordinatewise to match \(\widehat\mu_{31}\), then the true final layer is replayed.

### Cheap surrogate

- Homogeneous Gaussian-closure ridge surrogate.
- Base and early dominant-eigenvalue-ramp closures.
- Ranks 2, 4, 8, 16, 32.
- Fixed beta grid.
- Four-network discovery / four-network frozen validation split.
- Reference target: average of eight independent complete-Kerdock rotations per network.

### Frozen result

| Split | Selected rule result |
|---|---:|
| Discovery final raw gain | **1.1086×** |
| Discovery wins | 3/4 |
| Discovery mean correction cosine | +0.2080 |
| Frozen validation final gain | **0.8775×** |
| Frozen wins | 0/4 |
| Frozen worst candidate/baseline | **1.4235×** |
| Frozen mean correction cosine | **−0.0708** |
| Added compute proxy | 13.61B FLOPs |
| Frozen adjusted proxy | **0.8143×** |

The best post-hoc fixed setting over all eight networks produced only a 1.0194× raw gain and a **0.9460× adjusted proxy**, so even the pooled favorable view is score-negative.

### Interpretation

The layer-31 target is correct, but this surrogate is not. It predicts a correction with approximately the right scale and inconsistent orientation. The discovery/validation reversal is a sign error, not merely insufficient shrinkage.

**Decision:** close generic homogeneous Gaussian closure as the layer-31 residual surrogate.

## 4. Experiment C — independent pilot coefficient

To test whether a true multilevel residual stream could estimate the missing sign, the coefficient was learned without reference targets from independent partial Kerdock rotations:

- 4 or 8 rotations.
- 1, 2, 4, or 8 bases per rotation, capped at 32 basis evaluations.
- Regression in layer-31 space or through the approximate final-layer Jacobian.
- Ranks 2 and 4; shrinkage 0.5, 0.75, 1.0.

### Selected frozen result

| Quantity | Discovery | Frozen validation |
|---|---:|---:|
| Rule | base closure, rank 2, 4 rotations × 1 basis, layer-31 regression, 0.75 shrink | same frozen rule |
| Layer-31 gain | 1.0031× | **0.9850×** |
| Wins | 3/4 | 2/4 |
| Worst candidate/baseline | 1.0265× | **1.1765×** |
| Estimated/true-alpha correlation | −0.431 | −0.060 |
| Added compute proxy | 18.47B | 18.47B |
| Adjusted proxy | 0.9076× | **0.8912×** |

Some validation-postselected configurations showed tiny raw gains, but none approached compute break-even and their win counts/tails were poor.

**Decision:** independent exact pilots cannot rescue a surrogate whose residual direction is not transferable.

## 5. What remains alive

The multilevel principle remains plausible only with a better level-0 object. The ledger currently identifies two related mechanisms:

1. **Corrected connected-K3 quadratic-Hermite control:** a prespecified 70% shrink reached 0.881 held-out raw MSE, showing transferable signal, but full tensor propagation costs roughly 272B FLOPs.
2. **Adjoint-compressed K3/checkpoint contractions:** proposed to transport only the few contractions used by the control, with an estimated 10–40B cost. This is not yet frozen end-to-end.

These are better aligned with the newly identified layer-31 mean-defect channel than either stable-gate suffixes or a broad Gaussian closure.

## 6. Recommended continuation: checkpoint-contracted layer-31 MLMC

Do not immediately implement a many-level hierarchy. Run the following staged experiment.

### Level definitions

- **Level 0:** cheap adjoint/expected-gate transport of two connected-K3 control contractions from checkpoint layer 20 or 24 to layer 31.
- **Level 1:** same contractions plus regenerated local cumulant sources over the final 4–8 layers.
- **Level 2:** exact independent pilot correction at layer 31, evaluated only in the final-sensitivity subspace.
- **Output:** translate the full Kerdock layer-31 particle cloud to the corrected mean and replay the exact final layer.

### Measure before allocation

For every frozen network, record in the final-sensitivity metric:

\[
V_0=\operatorname{Var}(g_0),\quad
V_1=\operatorname{Var}(g_1-g_0),\quad
V_2=\operatorname{Var}(a_{31}-g_1),
\]

along with exact billed cost \(C_0,C_1,C_2\). Continue only if:

- the correction cosine is consistently positive on untouched networks;
- \(\sqrt{V_\ell C_\ell}\) decreases materially across levels;
- the one-level frozen raw gain is at least 1.3× before adding more levels;
- final adjusted score improves after exact replay, packing, and residual wall time;
- no severe per-network reversal occurs.

### Sampling rule

Never treat individual Kerdock bases as generic independent samples for the main estimator. Use one of:

- analytic expectation for the cheap level;
- complete Kerdock designs/rotations as level units;
- a correction that is algebraically zero-mean on the complete design.

Partial bases may be used for diagnostics or a conservative sign gate, but not as a replacement for the complete exact residual rule unless a fresh holdout demonstrates that the lost cancellation is repaid.

## 7. Final verdict

| Branch | Verdict |
|---|---|
| Stable-gate full MLMC | **Closed statistically; retain as compiler** |
| Generic Gaussian layer-31 residual | **Closed after frozen sign reversal** |
| Pilot-estimated Gaussian coefficient | **Closed; residual direction not learnable from cheap pilots** |
| Connected-K3 full tensor | **Mechanism real, deployment score-negative** |
| Adjoint/checkpoint-compressed K3 at layer 31 | **Best continuation of the multilevel idea** |

The research direction should therefore be renamed from “full multilevel prefix/suffix estimator” to **checkpoint-contracted layer-31 residual estimator**. It preserves the multilevel principle but targets the only residual channel currently known to have major oracle headroom.
