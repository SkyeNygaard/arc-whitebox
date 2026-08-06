# Agent 4 continuation v2 — exact common-shift range, source/replay architecture bifurcation, and canonical correction

**Competition:** WHestBench  
**Date:** 2026-07-30  
**Agent:** 4 — replayability, replacement bias, and nonlinear transfer  
**Protected data opened:** No  
**New evidence used:** post-v25 Agent 1, Agent 3, Agent 5, and Agent 6 reports; synthetic exact verifier only  
**Disposition:** **STOP the rank-4/rank-5 program; RETAIN rank-20/24/32 only as an output-linear signed-mixture source unless a nonlinear combined-shift advantage is demonstrated; CORRECT the canonical T70–T73 architecture wording.**

---

## Executive update

The first Agent-4 report correctly established that:

- shifting every checkpoint particle by a declared vector can be replayed exactly through the final affine–ReLU layer;
- shifting by the true checkpoint **mean defect** is not generally the same as replacing the checkpoint law by the true law;
- mean and covariance do not determine the final ReLU expectation;
- T77 is not a fixed-network theorem and is unnecessary at the final one-layer interface.

This continuation finds two deeper facts.

### New fact 1 — the full common-shift family has zero oracle representation error

Let `Z` be the baseline final-preactivation cloud and

\[
G(s)=\frac1N\sum_i \operatorname{ReLU}(Z_i+s)
\]

coordinatewise. Every coordinate `G_j` maps the real line continuously onto `[0,∞)`. Therefore, for every nonnegative target output `y`, there is a preactivation shift vector `s*` with

\[
G(s^*)=y.
\]

If the final matrix has full row rank—as a square Gaussian matrix does almost surely—there is a physical hidden common shift `δ*` satisfying `Wδ*=s*`.

Thus the earlier `0.0252` residual is **not an expressivity floor of common translation**. It is the residual of the special action `δ=Ph-Qh`, the checkpoint mean defect. An unrestricted target-informed common shift can interpolate the final target exactly.

This does not produce an estimator. The inverse shift `s*=G^{-1}(y)` is simply a coordinatewise reparameterization of the unknown target output. It supplies no new information.

### New fact 2 — the current linear source program and a single combined hidden shift are different estimators

Suppose frozen hidden directions `b_k` are replayed separately to create output columns

\[
a_k=G(Wb_k)-G(0).
\]

The T70/T72 linear source estimator is

\[
\widehat y_{\rm lin}=G(0)+A c,
\qquad A=[a_1,\ldots,a_r].
\]

A single physical combined shift instead gives

\[
\widehat y_{\rm state}=G\!\left(W\sum_k c_kb_k\right).
\]

In general,

\[
\boxed{\widehat y_{\rm lin}\ne\widehat y_{\rm state}.}
\]

ReLU replay is exact for either declared candidate, but T73 does not make the map linear in source coefficients. Fixed output-space normal equations `A^Te` are exact for the first architecture, not the second.

The output-linear estimator has an exact physical interpretation: it is integration under a **mass-one signed mixture of the baseline checkpoint cloud and separately translated copies**. This is the clean architecture that preserves T70/T72 and exact replay.

### Canonical recommendation

The late-interface program must choose one of two architectures explicitly:

1. **Output-linear signed-mixture source — recommended.**  
   Replay each frozen shift separately, linearly combine the resulting output columns, and use exact Gram normal equations. This is compatible with T70/T72 and Agent 6 economics. It needs `r` final-cloud scans, not one combined replay.

2. **Single combined hidden-state shift — nonlinear research variant.**  
   Apply `δ=Bc` once and optimize the exact piecewise-affine map. Fixed `A^Te` contractions are not sufficient globally; the required contraction directions depend on the current coefficient cell.

Post-v25 Agent 1 has now closed the tested rank-4/rank-5 natural source family even under exact separately replayed output columns and oracle coefficients. Agent 5 independently finds meaningful legal capacity only around rank 20–32 in aggregate, with safe tails requiring still larger rank. The rank-4/rank-5 priority should therefore be removed.

---

## 1. Exact final-interface setup

Let:

- `H in R^(N x d)` be the baseline penultimate post-ReLU particle cloud;
- `W in R^(d x m)` be the final affine matrix under column-output convention;
- `Z=HW in R^(N x m)` be cached final preactivations;
- `G:R^m -> R^m` be

\[
G_j(s_j)=\frac1N\sum_{i=1}^N (Z_{ij}+s_j)_+;
\]

- `y_Q=G(0)` be the baseline output;
- `y_P>=0` be the true post-ReLU target output;
- `e=y_P-y_Q` be the desired correction.

A hidden common shift `δ in R^d` induces the preactivation shift

\[
s=W^T\delta
\]

under the corresponding vector convention. All representation questions at the final layer therefore factor into:

1. which `s` vectors are physically reachable through `W`;
2. which output vectors lie in the range of the separable map `G`;
3. how a low-dimensional source constrains `s`;
4. whether source coefficients are combined before or after nonlinear replay.

---

## 2. A4-T8 — exact common-shift interpolation theorem

**Classification: proved; computer-assisted verifier included.**

### Theorem

For a finite scalar cloud `z_1,...,z_N`, define

\[
g(s)=\frac1N\sum_i(z_i+s)_+.
\]

Then:

1. `g` is continuous, convex, and nondecreasing;
2. `g(s)=0` for `s<=-max_i z_i`;
3. `g` is strictly increasing once it is positive;
4. `lim_{s->∞}g(s)=∞`;
5. `g(R)=[0,∞)`;
6. for every `y>0`, there is a unique `s` with `g(s)=y`.

Consequently, for the vector map `G`, every `y in [0,∞)^m` has a coordinatewise inverse shift `s*=G^{-1}(y)`.

If `W^T:R^d->R^m` is surjective, then there exists a hidden common shift `δ*` with

\[
W^T\delta^*=s^*,
\qquad
G(W^T\delta^*)=y.
\]

For `d=m=256` and a continuous random final matrix, surjectivity holds almost surely.

### Proof

Continuity and convexity follow termwise. Below the smallest activation threshold every term is zero. At every `s` for which `g(s)>0`, at least one term is active and every one-sided slope is at least `1/N`, so the function is strictly increasing on its positive range. For large positive `s`, all terms are active and `g(s)=s+mean(z)`, which diverges. The scalar range is therefore exactly `[0,∞)`. Apply the argument coordinatewise, then solve the linear system for `δ*` when `W^T` is surjective. `square`

### Verifier

On a random full-rank finite network instance, the supplied verifier obtains:

- maximum target-output interpolation error: `1.55e-14`;
- preactivation-shift solve error: `2.07e-14`;
- inverse round-trip output error: `4.44e-16`.

### Strategic interpretation

The full common-shift family is not merely high-capacity. It is final-output universal for this architecture. Therefore:

- `0.0252` is the oracle ratio of the **checkpoint-mean-defect shift**, not the best arbitrary common shift;
- “replacement bias of common translation” should be renamed **mean-defect action residual** when discussing that intervention;
- no richer checkpoint law is required for pure oracle final-output representability;
- richer representations matter only because they may expose a lower-dimensional or more observable parameterization.

---

## 3. A4-C2 — exact no-free-lunch inverse equivalence

**Classification: proved.**

For positive target coordinates, `G` is invertible coordinatewise. Hence the exact output-matching shift and the target output contain exactly the same information:

\[
y_P \longleftrightarrow s^*=G^{-1}(y_P).
\]

The full 256-dimensional shift target is therefore not an information reduction. Estimating all entries of `s*` exactly is algebraically equivalent to estimating all entries of `y_P` exactly.

A source becomes useful only if one can establish that `s*`, or the final correction, lies near a legal low-dimensional family whose coefficients have cheaper absolute estimators.

A canonical target-free full basis exists when `W` is invertible:

\[
b_j=W^{-T}e_j.
\]

Each `b_j` shifts exactly one final preactivation coordinate. The verifier reconstructs the identity action to `1.55e-14`. This is an exact rank-256 source, not a competition solution.

---

## 4. A4-T9 — architecture bifurcation theorem

**Classification: proved; exact counterexample and verifier included.**

Let hidden source directions be `B=[b_1,...,b_r]`, and put induced preactivation directions

\[
M=[W^Tb_1,\ldots,W^Tb_r]\in R^{m\times r}.
\]

Define separately replayed output columns

\[
a_k=G(M_{:k})-G(0),
\qquad A=[a_1,\ldots,a_r].
\]

There are two distinct estimators.

### Architecture L — linear output source

\[
C_L(c)=Ac.
\]

Its oracle coefficients solve the exact linear Gram problem

\[
\min_c\|e-Ac\|_H^2,
\qquad
A^THA\,c=A^THe.
\]

T70 scalarization and the exact Agent-6 Gram risk identity apply.

### Architecture S — single combined state shift

\[
C_S(c)=G(Mc)-G(0).
\]

This is exact physical common-shift replay, but it is nonlinear and piecewise affine in `c`.

### General inequivalence

In general,

\[
\boxed{C_L(c)\ne C_S(c).}
\]

### Exact one-particle counterexample

Take one output, one particle, baseline preactivation `z=-1/2`, and one source shift `M=1`.

The separately replayed source column is

\[
a=((-1/2+1)_+-(-1/2)_+)=1/2.
\]

For target correction `e=1/4`, the exact linear oracle coefficient is `c=1/2`, so

\[
C_L(c)=1/4.
\]

But the combined state shift is only `cM=1/2`, which reaches the gate threshold and gives

\[
C_S(c)=((-1/2+1/2)_+-0)=0.
\]

The mismatch is exactly `1/4`. The verifier reproduces it.

### Consequence

The following chain is invalid without an additional gate-cell theorem:

> replay each direction -> form output columns -> solve `A^Te` -> apply the coefficient combination as one hidden shift.

Exact replay of columns does not imply exact replay of their linear combination.

---

## 5. A4-T10 — exact signed-mixture realization of the linear source

**Classification: proved; verifier-backed.**

The output-linear architecture has an exact checkpoint-measure interpretation.

Let `nu` be the baseline empirical checkpoint measure and let `T_{b_k}#nu` be its translate by hidden shift `b_k`. For coefficients `c`, define the mass-one signed measure

\[
\nu_c=
\left(1-\sum_k c_k\right)\nu
+
\sum_k c_k\,T_{b_k}\#\nu.
\]

Then

\[
\boxed{
\int\phi_W\,d\nu_c
=G(0)+Ac.
}
\]

This is an exact identity for arbitrary real coefficients. When `c_k>=0` and `sum c_k<=1`, it is an ordinary probability mixture. Otherwise it is a signed mass-one checkpoint rule, equivalent to a real linear control correction.

The verifier gives maximum realization error `1.11e-16`.

### Why this matters

This theorem cleanly reconciles:

- physical source directions;
- exact final-layer replay;
- fixed linear output columns;
- exact normal equations;
- arbitrary correlated coefficient errors in Agent 6's Gram metric.

The cost is not one final-cloud scan. It is one scan per translated source column, although all columns can be batched and share `Z`.

At WHestBench dimensions, the raw cloud has

\[
N m=66{,}048\times256=16{,}908{,}288
\]

entries. Rank 24 therefore touches about `405.8 million` shifted preactivation entries before counting additions, comparisons, reductions, memory traffic, and source construction. This is plausibly small relative to dense propagation, but must be measured in the production subprocess.

---

## 6. Convex-mixture versus single-average-shift geometry

For nonnegative mixture weights summing to one, coordinatewise convexity gives

\[
\sum_k c_kG(s_k)
\ge
G\!\left(\sum_kc_ks_k\right).
\]

Thus an ordinary mixture of shifted clouds is generally richer than shifting once by the average shift. It adds a nonnegative Jensen/crossing correction in every output coordinate.

The verifier found strictly positive coordinatewise gaps on its random instance, from `0.00237` to `0.03642`.

This clarifies the role of mixtures:

- a symmetric mixture cannot reduce a coordinate relative to its center shift;
- a general signed mixture can move either way;
- the output-linear source is not an approximation to one combined shift—it is a different, exactly replayable signed-measure family.

---

## 7. A4-T11 — exact nonlinear normal equations for a combined shift

**Classification: proved; finite-difference verifier included.**

For Architecture S, let

\[
C(c)=G(Mc)-G(0),
\qquad
R(c)=\|C(c)-e\|_H^2.
\]

Away from gate hyperplanes, define active fractions

\[
p_j(c)=\frac1N\#\{i:Z_{ij}+(Mc)_j>0\}
\]

and `D(c)=diag(p(c))`. The exact Jacobian is

\[
J(c)=D(c)M.
\]

Therefore

\[
\boxed{
\nabla R(c)
=2M^TD(c)H[C(c)-e].
}
\]

An interior stationary point satisfies

\[
\boxed{
M^TD(c)He
=M^TD(c)HC(c).
}
\]

The right side is legal once `c` is proposed. The unknown side consists of `r` scalar contractions of `e` with the **coefficient-dependent** output directions

\[
u_k(c)=H D(c)M_{:k}.
\]

T70 can scalarize each such contraction, but a global nonlinear solve may require new contractions whenever the active-fraction cell changes.

The verifier matches the analytic gradient to centered finite differences with maximum error `4.58e-10`.

### Fixed gate-cell corollary

Inside a coefficient cell with no gate crossings, `D(c)=D_0` is constant and

\[
C(c)=C(c_0)+D_0M(c-c_0)
\]

exactly. The objective is then one quadratic, and fixed linear normal equations are valid for the cell-local Jacobian source.

The verifier constructs a no-crossing step and obtains affine error `2.00e-16`.

### Practical implication

A nonlinear combined-shift program is legal but materially more complex. It needs one of:

1. a certified coefficient region lying in one gate cell;
2. an exact event/cell optimizer with adaptive contractions;
3. a global action `c_0` followed by a cell-local residual correction;
4. a proof that the linear output-source solution and combined-shift solution are sufficiently close.

At present, there is no evidence that this complexity buys enough capacity over the output-linear architecture.

---

## 8. Reconciliation with the new parallel-agent evidence

### 8.1 Agent 1 — rank-4/rank-5 natural physical sources are closed

Agent 1 froze 186 target-free physical source rules and evaluated exact separately replayed output-column spans on independent confirmation.

The selected exact rank-5 source has pooled residual `0.557039`; rank 4 has `0.581350`. Both fail badly even with oracle coefficients. The report explicitly states that it replays each source column and evaluates the exact output-column span; it does **not** claim that a combined hidden shift equals the linear sum of finite replays.

This is exactly Architecture L. Therefore Agent 1's negative result is fully compatible with this continuation and is not weakened by the architecture correction.

**Decision:** stop the tested rank-4/rank-5 natural output-linear source family.

### 8.2 Agent 5 — legal capacity appears at rank 20–32, not rank 4/5

Agent 5's late-block output-Hankel family has confirmation oracle ratios:

- rank 12: `0.22377`, but essentially no score margin and unsafe tails;
- rank 20: `0.12968`;
- rank 24: `0.10820`;
- rank 32: `0.08170`.

It also reports that exact combined-shift replay differs little from tangent projection for the tested fitted actions. This suggests—but does not prove—that Architecture S adds little beyond a well-scaled linear source in the current regime.

**Decision:** if the late-block family continues, default to Architecture L so that exact Gram economics and fixed scalar contractions remain valid. Retain Architecture S only as a paired capacity comparison.

### 8.3 Agent 6 — exact economics applies to frozen linear output sources

Agent 6 proves the exact nonorthogonal Gram risk identity for a frozen linear output source and explicitly warns that the simplification fails when nonlinear replay changes the action.

This continuation identifies the exact replay architecture that preserves that theorem: the signed mixture of separately translated clouds.

### 8.4 Agent 3 — conic source is a direct output source

The rank-30 conic family has pooled exposed-development oracle ratio about `0.138`, but it is a direct final-output source. It needs no checkpoint replay theorem. Its issue is fixed-network contraction observability and economics, not replacement bias.

Do not merge conic and hidden-state source claims merely because both have scalar normal equations.

---

## 9. Corrected error decompositions by architecture

### 9.1 Linear output / signed-mixture source

Let `A` be frozen and let `c*` be the oracle linear coefficient. With estimate `chat`,

\[
e-A\widehat c
=
(e-Ac^*)+A(c^*-\widehat c).
\]

The first term is pathwise orthogonal to `col(A)` in the physical Gram metric. Therefore, for frozen `A`, source residual and coefficient error add exactly in squared risk, with arbitrary coefficient covariance handled by `G=A^THA`.

There is no nonlinear replay term. There is no separate replacement-bias term beyond the measured source residual: the signed-mixture family is the declared source.

### 9.2 Single combined state shift

For `C(c)=G(Mc)-G(0)`, choose oracle `c*`. Then

\[
e-C(\widehat c)
=
[e-C(c^*)]+[C(c^*)-C(\widehat c)].
\]

These terms are not generally orthogonal. Their risk includes a cross term. A tangent coefficient metric is only a local bound or cell-specific identity.

If one compares against the mean-defect action `delta_mean`, then

\[
e-C(\delta_{\rm mean})
\]

is the complete mean-defect action residual. Calling part of it replacement bias is useful analytically, but it is not a floor on arbitrary shifts.

---

## 10. Corrected canonical program

The v25 program should be replaced by the following stage gate.

### Gate A — choose the replay architecture

Every source report must declare one of:

- `OUTPUT_LINEAR_SIGNED_MIXTURE`;
- `SINGLE_COMBINED_SHIFT`;
- `CENTERED_AFFINE_DEFORMATION`;
- `DIRECT_OUTPUT_SOURCE`.

No report may use output-span oracle coefficients and then silently deploy them as a combined hidden shift.

### Gate B — source capacity

For ranks 20/24/32, report separately:

1. output-linear signed-mixture oracle ratio;
2. nonlinear combined-shift oracle ratio;
3. their exact difference on every network and rotation;
4. worst tails;
5. batched replay cost.

### Gate C — contractions

- Architecture L: estimate fixed `A^THe`; Agent 6 exact Gram economics applies.
- Architecture S: estimate adaptive `M^TD(c)He` or certify a fixed gate cell.

### Gate D — production replay

- Architecture L: count `r` shifted-cloud scans, batched memory traffic, reductions, and numerical cancellation.
- Architecture S: count one shifted-cloud scan plus nonlinear coefficient-solver cost.

### Gate E — validation

Do not open protected data. Rank 4/5 is closed for the tested natural family. Rank 12 is a boundary diagnostic. Only rank 20/24/32 or the separately quarantined conic rank-30 source merit bounded continuation.

---

## 11. Three decisive next tests

### Test 1 — architecture-paired late-block tournament

Using exactly the frozen Agent-5 rank-20/24/32 hidden frames, evaluate:

1. tangent/output-linear source columns at an explicitly frozen shift scale;
2. oracle output-linear signed-mixture action;
3. oracle exact single-combined-shift action;
4. exact cell-crossing counts and coefficient norms.

**Gate:** continue Architecture S only if it improves complete residual enough to pay for adaptive contractions and remains stable on every grouped network. Otherwise canonicalize Architecture L.

### Test 2 — exact batched replay accounting

Implement a fused kernel that takes cached `Z` and an `m x r` shift matrix and accumulates all `r` output columns in one pass over blocks.

Report:

- FlopScope operations;
- residual wall time;
- peak memory;
- accumulation precision;
- direct-versus-batched equality;
- cost at ranks 20, 24, and 32.

The old one-extra-dense-layer `3.125%` charge and the one-scan `0.018%` common-shift estimate are both wrong for Architecture L unless explicitly reconciled with `r` scans.

### Test 3 — shared block contractions under the chosen architecture

For Architecture L, derive one shared transform for the entire rank-24 vector `A^THe`; do not create 24 unrelated estimators.

For Architecture S, freeze a global action and determine whether all development cases remain in one or a small number of gate cells. If not, the adaptive contraction burden is likely fatal.

---

## 12. Claims I tried to disprove

### “The `0.0252` result proves common shifts have unavoidable replacement bias.”

**Disproved.** It measures the checkpoint-mean-defect shift. The full common-shift family can interpolate any nonnegative final target when the final matrix is surjective.

### “An exact output-matching shift is a breakthrough.”

**Disproved as an estimator claim.** It is `G^{-1}(y_P)`, an invertible reparameterization of the unknown target output.

### “Exact replayed source columns make combined replay linear.”

**Disproved by an exact one-particle counterexample.** Column replay and coefficient combination do not commute with ReLU.

### “The linear source is not physically replayable.”

**Disproved.** It is exactly replayable as a mass-one signed mixture of the baseline cloud and separately shifted copies.

### “A mixture is equivalent to one average shift.”

**Disproved.** Convexity gives a nonnegative Jensen gap; mixtures change distributional shape.

### “T70/T72 can be used unchanged for a nonlinear combined shift.”

**Disproved globally.** The exact contraction directions are `HD(c)M`, which depend on the coefficient cell.

### “Agent 1's negative result may be an artifact of this nonlinear mismatch.”

**Disproved.** Agent 1 evaluates the exact output-column span—Architecture L—and explicitly does not claim combined-shift equivalence.

### “Rank 4/5 should remain the canonical priority.”

**Disproved for the broad tested natural families.** Independent Agent 1 and Agent 5 tournaments place useful legal capacity at materially higher rank.

### “Single combined replay is automatically cheaper and therefore preferable.”

**Not established.** It uses one scan but requires nonlinear/adaptive coefficient recovery. Architecture L uses multiple scans but preserves exact linear Gram economics. The total score, not scan count alone, decides.

---

## 13. Conflicts with existing ledger entries

### Conflict 1 — common-shift replacement-bias wording

**Old interpretation:** common translation intrinsically has replacement bias.  
**Correction:** the **mean-defect common shift** has residual `0.0252`; the unrestricted common-shift family is output-universal under final-matrix surjectivity.

### Conflict 2 — T70/T72 plus T73 composition

**Old interpretation:** replay directions into columns, solve linear coefficients, then apply one combined shift.  
**Correction:** T70/T72 applies to the output-linear column source. One combined shift is a different nonlinear estimator unless a fixed gate-cell theorem is supplied.

### Conflict 3 — exact replay cost

**Old interpretations:** either one extra dense final layer or one negligible scan.  
**Correction:** Architecture L needs `r` translated-cloud scans; Architecture S needs one. Report architecture-specific production cost.

### Conflict 4 — rank-4/rank-5 leading program

Post-ledger Agent 1 and Agent 5 evidence closes the tested natural low-rank family. The canonical constructive question is now shared observability/economics for rank 20/24/32, or the independent conic rank-30 source.

### Conflict 5 — source residual versus replacement bias

For a frozen output-linear source, all representation limitations belong in the source oracle residual. Adding a separate replacement-bias penalty double counts. For a chosen state action, a replacement decomposition may be informative, but cross terms must remain explicit.

---

## 14. Proposed ledger updates

The companion `agent4_v2_ledger_patch.csv` contains provisional rows.

| ID | Classification | Claim | Action |
|---|---|---|---|
| A4-T8 | Proved + verifier | Full-rank final common-shift family interpolates every nonnegative final target | Promote theorem |
| A4-C2 | Proved | Exact output-matching shift is information-equivalent to target output | Promote no-free-lunch corollary |
| A4-T9 | Proved + counterexample | Output-linear source and single combined hidden shift are generally inequivalent | Promote architecture guard |
| A4-T10 | Proved + verifier | Linear output source is exact mass-one signed mixture of translated checkpoint clouds | Promote recommended replay theorem |
| A4-T11 | Proved + verifier | Nonlinear combined-shift gradient uses coefficient-dependent active-fraction contractions | Promote nonlinear obligation |
| A4-M2 | Reconciliation | Agent-1 exact rank-4/5 closure applies to output-linear architecture | Merge post-v25 evidence |
| A4-M3 | Reconciliation | Agent-5 viable region is rank 20/24/32; architecture comparison still required | Update priority |
| A4-Q2 | Quarantine | Do not combine linear oracle coefficients into one hidden shift without cell certification | Add promotion gate |

---

## 15. Reproduction

Run:

```bash
python verify_agent4_architecture_bifurcation.py \
  --output agent4_v2_results.json
```

The verifier checks:

- exact full-dimensional common-shift interpolation;
- inverse stop-loss round trip;
- canonical coordinate-shift hidden basis;
- exact linear-source/combined-shift counterexample;
- signed-mixture replay identity;
- convex-mixture Jensen gap;
- exact nonlinear gradient formula;
- exact affinity inside a fixed gate cell.

All checks pass.

---

## Final recommendation

### Rank 4/5: **STOP**

The post-v25 source tournaments close the broad tested natural family before coefficient estimation.

### Rank 20/24/32 late-block source: **CONDITIONALLY CONTINUE, output-linear architecture first**

Use separately replayed output columns and interpret their linear combination as a signed mixture of translated checkpoint clouds. This retains exact Gram risk and fixed scalar contractions. Measure the true `r`-scan production cost.

### Single combined hidden shift: **QUARANTINE as a paired nonlinear alternative**

It is exactly replayable and full-rank universal, but low-rank coefficient recovery is nonlinear. Continue only if a direct paired oracle experiment demonstrates material capacity or cost advantage over the output-linear source.

### Conic rank 30: **SEPARATE CONDITIONAL CONTINUATION**

It is a direct output source. Replayability is solved by construction; observability and covariance-cost economics remain.

> The replay problem is now fully classified. The remaining winning question is not whether a late correction can be replayed. It is whether a 20–32 dimensional legal output source has a shared absolute contraction transform cheap enough to beat the score frontier.
