# Agent 4 continuation v3 — replay cost closure, exact stop-loss compilation, and a finite-transcript no-go theorem

**Competition:** WHestBench  
**Date:** 2026-07-30  
**Agent:** 4 — replayability, replacement bias, nonlinear transfer, and implementation closure  
**Protected data opened:** **No**  
**Evidence added:** exact algebra; public-design constructive counterexample; local full-shape synthetic kernel benchmarks; official FlopScope-formula accounting; post-v2 Agent 2/5/8 evidence reconciliation  
**Disposition:** **PROMOTE the direct-output rank-approximately-36 source to the leading Gate-A representation; DEMOTE hidden translated-cloud replay to a solved fallback primitive; STOP searching for a universal exact identity from the 129 baseline group-output transcript alone; run one source-specific convex contraction audit before any further estimator work.**

---

## Executive conclusion

The replay question is now closed more strongly than in v2.

1. **Translated-cloud replay is not economically fatal.** The validated cached-preactivation primitive costs `0.0847B` FLOPs per translated cloud. Even serialized rank-24 replay therefore costs about `2.0328B`, only `1.165%` of the `174.5B` current operating point. The earlier universal `3.125%` replay multiplier materially overstated the cost of the output-linear signed-mixture architecture.
2. **All common-shift queries can be compiled exactly.** Sorting each final-preactivation coordinate once and storing prefix sums turns every shifted ReLU mean into a binary-search query. This compiles both the output and the active-fraction Jacobian exactly. Under the published FlopScope 0.9.1 formulas, the dominant full-width preprocessing cost is about `1.200488B`; with a conservative query envelope, rank 32 costs at most `1.202077B`, saving `1.508323B` versus serialized translated scans.
3. **Replay nevertheless ceased to be the leading issue.** Agent 8 found a gauge-invariant direct-output basis-PCA source constructed from the 129 group-resolved baseline output means. Its frozen adaptive rule uses about 36 modes and reaches confirmation pooled/worst oracle ratios `0.0749 / 0.1830`. It is exactly linear and needs **no replay**.
4. **The baseline transcript cannot support a universal exact identity.** Using the public 66,048-node design, I construct a width-2, three-ReLU-layer homogeneous network that is exactly zero on every design node but has strictly positive Gaussian mean. Adding it to another network leaves every one of the 129 group-output means unchanged while changing the true target. Scaling the blind component makes the target gap arbitrarily large without changing the transcript.
5. **The remaining branch is now sharply defined.** A successful estimator must use information beyond the finite baseline group-output transcript—most plausibly network weights/checkpoint features plus a source-specific exact telescope—or exploit a random-network prior probabilistically. Agent 2's checkpoint-gauge SOCP is the correct bounded falsifier for this branch.

The new canonical statement is:

> **Source construction and replay are solved. Absolute signed contraction observability is the sole remaining late-interface bottleneck. A transcript-only universal identity is impossible; the next test must be weight-aware and source-specific.**

---

## 1. Evidence hierarchy and frozen inputs

### 1.1 Proved algebraically

- exact sorted stop-loss representation of shifted ReLU means;
- exact active-fraction/Jacobian queries from the same table;
- exact output-linear signed-mixture replay after compilation;
- finite-node ReLU blind-spot theorem;
- transcript indistinguishability corollary;
- architecture-specific error and cost separations from v2 remain valid.

### 1.2 Computer-assisted proof / verifier-backed

- exact equality of sorted-table queries and direct scans to floating-point tolerance;
- explicit blind network is zero on all 66,048 public nodes;
- its exact Gaussian mean is positive;
- 129 group transcripts are identical for the constructed network pair;
- production-shape local replay benchmark and memory audit;
- FlopScope formula arithmetic and score-frontier calculations.

### 1.3 Numerical discoveries imported from frozen external agent work

- Agent 5 nonlinear-secant rank-20/24/32 source capacity;
- Agent 8 direct-output basis-PCA rank-20/32/40 and adaptive-rank capacity;
- Agent 2 checkpoint-gauge convex frontier and hostile generic screens.

No official protected cohort or sealed target was accessed.

---

## 2. Correct architecture taxonomy

The v2 distinction remains mandatory.

### Architecture L — output-linear translated-cloud source

For baseline preactivations `Z in R^(N x m)` and declared shift vectors `s_k in R^m`, define

\[
G(s)=\frac1N\sum_{i=1}^N (Z_i+s)_+,
\qquad
A_k=G(s_k)-G(0).
\]

The candidate is

\[
\widehat y_L=G(0)+A c.
\]

It is exactly the output under the mass-one signed checkpoint measure

\[
\nu_c=\left(1-\sum_kc_k\right)\nu+\sum_kc_k\,T_{b_k}\#\nu.
\]

This is a frozen linear source. Exact Gram risk, normal equations, arbitrary correlated coefficient errors, and Agent 6 economics apply.

### Architecture S — single combined hidden shift

For a preactivation shift frame `M in R^(m x r)`,

\[
\widehat y_S(c)=G(Mc).
\]

This is nonlinear in `c` across gate cells. Output-column coefficients from Architecture L cannot be silently deployed as one hidden shift.

### Architecture O — direct output source

For a frozen output basis `V in R^(m x r)`,

\[
\widehat y_O=y_Q+V\widehat\theta.
\]

There is no checkpoint replacement and no final-layer replay. The only obligations are:

- legal target-free construction of `V`;
- oracle source capacity;
- estimation of the absolute contraction vector `theta=V^T(y_P-y_Q)`;
- exact joint score economics.

**Post-v2 correction:** Architecture O is now the leading representation because Agent 8 established a safe high-rank Gate-A source.

---

## 3. A4-T12 — exact sorted stop-loss compilation theorem

**Classification:** proved; verifier-backed.

Fix one output coordinate and let

\[
z_{(1)}\le\cdots\le z_{(N)}
\]

be its sorted baseline preactivations. Let

\[
S_k=\sum_{i=1}^k z_{(i)},\qquad S_0=0.
\]

For a shift `s`, define

\[
k(s)=\#\{i:z_i\le -s\}.
\]

Then

\[
\boxed{
G(s)=\frac{S_N-S_{k(s)}+(N-k(s))s}N.
}
\]

The strict active fraction is

\[
\boxed{
D(s)=\frac{N-k(s)}N.
}
\]

### Proof

Every index `i <= k(s)` satisfies `z_(i)+s <= 0` and contributes zero. Every remaining index is active, so

\[
\sum_i(z_i+s)_+
=\sum_{i=k(s)+1}^N z_{(i)}+(N-k(s))s
=S_N-S_{k(s)}+(N-k(s))s.
\]

Divide by `N`. The active-fraction formula follows from counting the same suffix. ∎

### Vector form

Sort and prefix-sum each of the `m` columns independently. Then for any shift matrix `S in R^(r x m)`, every entry of `G(S)` and `D(S)` is obtained by one `searchsorted` plus constant arithmetic.

### Consequence for Architecture L

Every source column

\[
A_k=G(s_k)-G(0)
\]

is exact after table construction. No translated cloud must be materialized.

### Consequence for Architecture S

For

\[
C(c)=G(Mc)-G(0),
\]

the exact Jacobian away from irrelevant kink conventions is

\[
\boxed{J(c)=\operatorname{diag}(D(Mc))M.}
\]

Thus repeated nonlinear optimization can query both objective and Jacobian without rescanning the cloud. This removes replay compute as an excuse for not comparing Architecture S; it does **not** make the target contractions observable.

---

## 4. Exact and projected replay cost

### 4.1 Validated serialized translated-cloud primitive

The archived production audit validates one cached-preactivation translation at

\[
0.0847\text{B FLOPs}.
\]

For rank `r`, the conservative serialized Architecture-L charge is

\[
C_L(r)=0.0847r\text{B}.
\]

| Rank | Replay B | Fraction of current `174.5B` | Oracle `r*` | Replay-adjusted zero-noise ratio | Aggregate pass | Worst raw case below target |
|---:|---:|---:|---:|---:|:---:|:---:|
| 4 | 0.3388 | 0.194% | 0.441716 | 0.442574 | no | no |
| 5 | 0.4235 | 0.243% | 0.404465 | 0.405447 | no | no |
| 8 | 0.6776 | 0.388% | 0.290770 | 0.291899 | no | no |
| 12 | 1.0164 | 0.582% | 0.220745 | 0.222031 | yes | no |
| 20 | 1.6940 | 0.971% | 0.127663 | 0.128902 | yes | no |
| 24 | 2.0328 | 1.165% | 0.109470 | 0.110745 | yes | no |
| 32 | 2.7104 | 1.553% | 0.081667 | 0.082935 | yes | yes |

This closes the old cost misconception:

- rank 12 is not killed by replay; it is killed by tiny margin and unsafe tails;
- ranks 20/24/32 retain large aggregate estimator headroom;
- rank 32 is the first Agent-5 source in this table whose worst raw case is below the competition target.

### 4.2 Official-formula sorted-table projection

For `N=66,048`, `m=256`, and `ceil(log2 N)=17`, the published FlopScope 0.9.1 formulas give:

\[
C_{sort}=4mN\lceil\log_2N\rceil
=1,149,763,584,
\]

\[
C_{cumsum64}=2(Nm-m)
=33,816,064.
\]

Including one conservative `Z` cache write gives a dominant core of

\[
1,200,487,936
=1.200488\text{B}.
\]

The companion calculation adds a deliberately conservative query/indexing/output envelope. It is a **projected upper bound under published formulas**, not yet an official subprocess measurement.

| Rank | Sorted upper B | Serialized B | Saving B | Cost ratio | Replay-adjusted `r*` | Aggregate pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 1.201084 | 1.0164 | -0.184684 | 1.182 | 0.222265 | yes |
| 20 | 1.201481 | 1.6940 | 0.492519 | 0.709 | 0.128542 | yes |
| 24 | 1.201680 | 2.0328 | 0.831120 | 0.591 | 0.110224 | yes |
| 32 | 1.202077 | 2.7104 | 1.508323 | 0.444 | 0.082230 | yes |

The dominant-core break-even is rank

\[
r>14.17.
\]

Therefore:

- direct scanning remains preferable for one or a few shifts;
- sorted compilation becomes economically attractive around rank 15;
- it is especially useful for repeated Architecture-S objective/Jacobian queries;
- an official immutable-array FlopScope implementation remains a promotion gate.

---

## 5. Full-shape local benchmark

### 5.1 Protocol

- synthetic `Z` of shape `66,048 x 256`;
- float32 preactivations;
- float64 reductions;
- single-thread BLAS environment;
- ranks 20, 24, and 32;
- no protected data.

### 5.2 Direct scan batching

| Rank | Batch 1 median s | Batch 2 | Batch 4 | Batch 8 | Best |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.3863 | 0.4189 | 0.4162 | 0.4570 | 1 |
| 24 | 0.4608 | 0.4891 | 0.5002 | 0.6098 | 1 |
| 32 | 0.5914 | 0.6671 | 0.6907 | 0.7877 | 1 |

Larger source batches were slower. The scan is memory-bandwidth dominated; rank batching does not supply a hidden wall-time rescue.

### 5.3 Sorted all-column table

- preprocessing median: `1.3921s`;
- incremental sorted/prefix memory: `198.5 MiB`;
- rank-32 query median: `0.0068s`;
- rank-32 max absolute discrepancy from direct scan: `1.072e-08`.

### 5.4 Memory-bounded blockwise table

| Column block | Median s | Peak RSS delta MiB | Max absolute discrepancy |
|---:|---:|---:|---:|
| 4 | 0.5172 | 0.0 | 1.090e-08 |
| 8 | 0.5308 | 0.0 | 1.090e-08 |
| 16 | 0.5329 | 0.0 | 1.090e-08 |
| 32 | 0.6268 | 0.0 | 1.090e-08 |
| 64 | 0.7601 | 3.0 | 1.090e-08 |
| 128 | 0.9267 | 96.9 | 1.090e-08 |
| 256 | 0.8896 | 96.8 | 1.090e-08 |

Blocks 4–16 were fastest locally and bounded transient table memory. This prototype is not an official timing result, but it supplies the production implementation shape: sort a small output block, answer every source query, release it, and continue.

---

## 6. The direct-output source supersedes hidden replay

Agent 8's direct-output source uses the 129 natural group-resolved output means already produced by the baseline. Let

\[
y_g\in\mathbb R^m,\qquad
\bar y=\frac1{129}\sum_g y_g,
\]

and

\[
C_y=\frac1{129}\sum_g(y_g-\bar y)(y_g-\bar y)^T.
\]

The leading eigenspace of `C_y` is:

- target-free;
- hidden-permutation invariant;
- positive-ReLU-rescaling invariant;
- constructed without additional network evaluations;
- an exact direct output source.

### Frozen confirmation frontier

| Source | Rank | Pooled oracle ratio | Worst ratio | Aggregate pass | Tail pass | Root T72 allowance | `gamma tr Sigma` ceiling |
|---|---:|---:|---:|:---:|:---:|---:|---:|
| direct_rank20 | 20.00 | 0.128298 | 0.287579 | yes | no | 0.121828 | 0.014842 |
| direct_rank32 | 32.00 | 0.081362 | 0.213839 | yes | yes | 0.194774 | 0.037937 |
| direct_rank40 | 40.00 | 0.067136 | 0.161300 | yes | yes | 0.220910 | 0.048801 |
| adaptive_direct | 36.25 | 0.074900 | 0.183000 | yes | yes | 0.206337 | 0.042575 |

### Strategic consequence

The hidden translated-cloud branches no longer lead:

1. rank-20 direct output has strong pooled capacity but unsafe tails;
2. rank-32 direct output passes the raw target on every tested confirmation case;
3. rank-40 and adaptive rank approximately 36 have substantial safe capacity;
4. none requires replay;
5. all tested signed observables nevertheless fail out of network.

Sorted stop-loss compilation should therefore be retained as:

- an exact replay fallback;
- an Architecture-S nonlinear comparison tool;
- a verifier for hidden-source claims;
- not the primary path to victory.

---

## 7. A4-T13 — finite-design homogeneous-ReLU blind-spot theorem

**Classification:** proved constructively; public-design verifier-backed.

### Theorem

Let `X={x_1,...,x_N}` be any finite nonzero subset of `R^d` with `d>=2`. There exists a nonzero bias-free homogeneous ReLU network `g:R^d->[0,infinity)` of width 2 and three ReLU layers such that

\[
g(x_i)=0\quad\text{for every }i,
\]

but

\[
\mathbb E[g(G)]>0,
\qquad G\sim N(0,I_d).
\]

### Construction

Project the finite set onto a generic two-dimensional plane so no projected node is zero. The projected directions occupy finitely many angles, hence there is an open empty angular interval `(alpha,beta)`.

Choose inward unit normals `a,b` whose positive halfspaces intersect exactly in a smaller wedge inside that interval. Define

\[
\boxed{g(x)=\min\{(a^Tx)_+,(b^Tx)_+\}.}
\]

Every design point lies outside the wedge, so at least one hinge is zero and `g(x_i)=0`.

The function is represented by the width-2 network

\[
p=(a^Tx)_+,
\qquad q=(b^Tx)_+,
\]

\[
d=(p-q)_+,
\qquad
\boxed{g=(p-d)_+}.
\]

Because `p,q>=0`, `p-(p-q)_+=min(p,q)`.

### Exact Gaussian mean

Let the wedge width be `Delta=beta-alpha`. In polar coordinates,

\[
\mathbb E[g(G)]
=\frac{\mathbb E[R]}{2\pi}
\int_0^\Delta\min(\sin t,\sin(\Delta-t))dt
\]

\[
=\frac{\sqrt{\pi/2}}\pi
\left(1-\cos\frac\Delta2\right)>0.
\]

This is an exact closed form.

### Public WHestBench design certificate

The verifier reconstructs the public 66,048-node design and obtains:

- largest empty projected angular gap: `0.001124672142` radians;
- certified inner wedge width: `0.000562336071` radians;
- maximum `g` on all design nodes: `0.0`;
- design nodes above `1e-13`: `0`;
- exact Gaussian mean: `1.576928348949532e-08`;
- explicit network algebra maximum error: `4.066e-20`.

---

## 8. A4-C3 — no universal transcript-only exact contraction identity

Let `T(f)` denote the complete vector of the 129 baseline group output means of a network `f`.

Choose a scalar homogeneous ReLU network `h` whose group means are nonconstant, and use the blind network `g` above. Then

\[
T(h)=T(h+g),
\]

because `g` vanishes on every baseline node, but

\[
\mathbb E[h(G)]\ne\mathbb E[(h+g)(G)].
\]

The public certificate has:

- group-mean variance of `h`: `1.632869746636187e-07`;
- maximum transcript difference: `0.0`;
- target difference: `1.576928348949532e-08`.

Multiplying `g` by any positive scalar preserves the transcript and scales the target difference. Therefore no deterministic function of the 129 group-output transcript can be a universally exact Gaussian-mean estimator over the full homogeneous ReLU class. The same statement applies to direct-source contractions by embedding this scalar output in one coordinate.

### Exact scope

This theorem closes only the **universal transcript-only identity** branch. It does **not** exclude:

- identities using the realized network weights;
- checkpoint states or pathwise derivatives not recoverable from group outputs;
- high-probability claims under the random He-network ensemble;
- biased estimators with favorable expected score;
- conditional estimators that abstain outside a certified model class.

This distinction is essential. The counterexample is a structural no-go, not a claim that Agent 8's source is useless on the competition distribution.

---

## 9. Reconciliation with Agent 2's checkpoint-gauge theorem

Agent 2 proves that for arbitrary checkpoint-control matrices, terminal contractions admit exact telescoping identities. For a frozen terminal source `V`, the entire declared linear checkpoint-control family can be optimized as a convex SOCP, with direct estimation included as a feasible point.

This is now the correct continuation for the direct-output source.

### Source-specific gate

For target ratio `p=0.2304147465`, source residual `r_*`, and root contraction difficulty `S`, T72 requires

\[
\boxed{(\sqrt{r_*}+S)^2<p.}
\]

The maximum allowable root difficulty is

\[
S_{max}=\sqrt p-\sqrt{r_*},
\]

and the shared-sampling product must satisfy

\[
\gamma\operatorname{tr}\Sigma<S_{max}^2.
\]

For the leading direct sources:

- rank 32: `S_max=0.194774`, squared ceiling `0.037937`;
- rank 40: `S_max=0.220910`, squared ceiling `0.048801`;
- adaptive rank approximately 36: `S_max=0.206337`, squared ceiling `0.042575`.

### Required bounded experiment

Freeze the Agent-8 source before any target access and solve the Agent-2 SOCP over a preregistered checkpoint family. Report:

1. source rank, physical Gram, pooled and per-network `r_*`;
2. full joint covariance blocks for all candidate checkpoint controls;
3. direct, fixed-adjoint, expected-gate, greedy, and globally optimal objectives;
4. primal and dual SOCP values;
5. untouched covariance validation;
6. resulting T72 lower bound against the exact source-specific ceiling;
7. condition numbers, regularization sensitivity, and tails.

**Decision rule:**

- if the validated dual lower bound exceeds `S_max`, stop the entire declared linear weight-aware checkpoint branch;
- if it passes with material slack, implement only that certified joint vector estimator;
- do not train another generic coefficient predictor first.

---

## 10. What is now solved and unsolved

### Solved

- exact replay of any declared common shift;
- exact distinction between output-linear mixtures and one combined shift;
- exact rank-linear translated-cloud cost model;
- exact stop-loss compilation of arbitrary common-shift queries;
- source/replay separation;
- broad natural rank-4/5 source closure;
- existence of a legal high-capacity direct-output source;
- impossibility of a universal exact estimator from finite group outputs alone.

### Unsolved

- an absolute signed contraction identity for the direct-output source;
- a source-specific weight-aware estimator below the T72 ceiling;
- a random-He conditional-variance lower bound for the full legal transcript;
- official subprocess measurement of the sorted table primitive;
- whether nonlinear Architecture S has any source-capacity advantage worth its adaptive contractions.

---

## 11. Claims I tried to disprove

### “Rank-24 signed-mixture replay is too expensive.”

**Disproved.** Validated serialized replay is about `2.0328B`, `1.165%` of current effective compute.

### “Batch all source shifts to make replay fast.”

**Disproved locally.** Batch 1 was fastest at ranks 20, 24, and 32. Larger batches increased wall time.

### “Sorting is obviously too expensive.”

**Disproved above moderate rank under published accounting.** The projected break-even is about rank `14.17`, and rank-32 projected savings are about `1.508B`.

### “Sorted replay solves the competition path.”

**Disproved strategically.** It solves candidate evaluation, not the unknown absolute contractions. The latest direct-output source avoids replay entirely.

### “The 129 group-output means might admit a universal exact identity.”

**Disproved over the unrestricted homogeneous ReLU class.** An explicit width-2, depth-3 blind component has identical finite transcript and positive Gaussian mean.

### “The no-go kills all direct-output work.”

**Disproved as an overstatement.** It does not cover weight-aware identities or probabilistic random-network estimators. Agent 2's source-specific telescope remains open.

### “The old four/five-dimensional repair diagnosis implies a four/five-dimensional legal source.”

**Disproved by independent source tournaments.** Target-informed orientation compresses; canonical target-free coordinates require roughly 32–40 dimensions for safe capacity.

### “Replay remains the deciding architecture criterion.”

**Disproved.** Architecture O has stronger safe source capacity and zero replay. Observability dominates.

---

## 12. Conflicts and canonical corrections

### Conflict 1 — universal fixed replay multiplier

**Old:** charge every source a `1.03124786` multiplier.  
**Correction:** report architecture-specific replay. Direct-output sources pay zero; translated-cloud sources pay approximately `0.0847B` per column or a sorted-table charge; one combined shift pays one translation query plus coefficient-solver cost.

### Conflict 2 — v2 rank-20/24/32 hidden source priority

**Old:** hidden output-linear signed mixture is the leading continuation.  
**Correction:** retain it as a solved fallback. The direct-output adaptive rank-approximately-36 source is now the leading Gate-A representation.

### Conflict 3 — information-bound target

**Old:** attempt an exact identity from the sigma-algebra of 129 group means.  
**Correction:** a universal exact identity from that finite transcript is impossible. Restrict the identity program to weight-aware/checkpoint information or formulate an ensemble-specific lower bound.

### Conflict 4 — sorted compilation status

**Do not call it an official measured cost.** The algebra and dominant FlopScope formulas are exact; the total `~1.20B` figure is a conservative projected upper bound pending official immutable-array execution.

### Conflict 5 — source dimension language

“Repair rank 4–5” must always be labeled target-informed. Safe target-free direct-output capacity currently requires roughly 36 modes.

---

## 13. Canonical recommendation

### Leading branch: direct-output adaptive PCA, rank approximately 36

**Conditionally continue only for one exact source-specific contraction audit.** Source construction and replay are solved. The adaptive confirmation frontier has `r*=0.0749`, worst `0.1830`, and root allowance `0.20634`.

### Secondary branch: fixed direct-output rank 32 or 40

Prefer rank 32 for lower contraction dimension if the SOCP passes; rank 40 supplies more capacity and tail slack but raises observability burden.

### Archived implementation primitive: sorted stop-loss replay

Promote the theorem and keep the code. Use it for:

- exact hidden-source audits;
- repeated nonlinear combined-shift optimization;
- any future high-rank translated-cloud source;
- not as the default representation while direct output dominates.

### Stop

- rank-4/rank-5 natural source search;
- another generic signed-feature predictor;
- universal exact identities using only the 129 group-output transcript;
- treating source columns and one combined hidden shift as interchangeable;
- opening protected data before a source-specific mathematical/economic pass.

### One next experiment

> **Freeze Agent 8's direct-output adaptive source and solve Agent 2's full checkpoint-gauge SOCP with a primal/dual certificate and untouched covariance validation.**

This is the shortest route to a conclusive answer. A pass gives the first mathematically justified deployable contraction estimator. A fail closes the full declared linear weight-aware late-interface branch, rather than merely falsifying one regression.

---

## 14. Reproduction

Run:

```bash
python benchmark_signed_mixture_replay.py --output replay_benchmark_results.json
python benchmark_blockwise_sorted_stoploss.py --output blockwise_sorted_benchmark.json
python derive_agent4_v3_economics.py --output agent4_v3_economics.json --csv agent4_v3_economics.csv
python derive_sorted_stoploss_flops.py --output sorted_stoploss_flops.json
python derive_direct_output_source_economics.py --output direct_output_source_economics.json
python sorted_stoploss_kernel.py --output sorted_stoploss_kernel_selftest.json
python verify_finite_design_relu_blind_spot.py \\
  --asset /path/to/public/kerdock_mub5_seed3.npz \\
  --output finite_design_relu_blind_spot.json
python verify_agent4_v3.py --root . --output agent4_v3_verification.json
```

The full bundle contains scripts, machine-readable results, provenance hashes, a proposed ledger patch, and a SHA-256 manifest.
