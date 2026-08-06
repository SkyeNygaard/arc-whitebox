# WHestBench best-path continuation — exact contracted checkpoint program

**Date:** 2026-07-30  
**Starting state:** canonical v22  
**Disposition:** new exact identities and a tighter competition gate; no deployable candidate claimed.

## Executive conclusion

The strongest constructive continuation is now more specific:

> Compress the multi-checkpoint repair into a four- or five-dimensional **late-interface output source**, estimate only its signed scalar contractions, and use exact compiled replay at the last hidden checkpoint.

Three exact results make this sharper.

1. Every output-space contraction of the true quadrature error has an exact adjoint-potential representation as the discrepancy of one scalar, network-dependent function.
2. The unpartitioned potential is algebraically equivalent to the original output projection, so it is not a free estimator. Value can arise only from a band whose expectation can be computed or estimated more cheaply than a full network evaluation.
3. For unbiased contraction estimators, source capacity, estimator variance, and added compute obey an exact score frontier. This shows that the old oracle source gate `0.20–0.22` is only a zero-cost hard gate and is generally too weak for a practical method.

The authenticated Oracle source snapshot also changes the architectural emphasis. Exact correction of the recorded post-ReLU layer 30 checkpoint gave confirmation MSE ratio `0.0252`, while layer 29 gave `0.0539`. These late checkpoints leave far more score margin than layer 23 (`0.200`). The multi-component upstream repair should therefore be **compressed and applied once at the late interface**, rather than replayed as four corrections through the full suffix.

---

## 1. Setup

Let a bias-free depth-`L` ReLU network be

\[
h_0(x)=x,\qquad
z_\ell(x)=W_\ell h_{\ell-1}(x),\qquad
h_\ell(x)=\sigma(z_\ell(x)),
\]

where `σ(t)=max(t,0)` coordinatewise. Let `P` be exact spherical expectation and `Q` the baseline cubature functional. Define

\[
\delta_\ell=(P-Q)h_\ell.
\]

Because `P` and `Q` both integrate coordinates exactly,

\[
\delta_0=(P-Q)x=0.
\]

Use the exact identity

\[
\sigma(t)=\frac12(t+|t|).
\]

---

## 2. T70 — exact adjoint-potential contraction identity

For any output direction `u`, define the deterministic backward vectors

\[
q_\ell(u)
=
2^{-(L-\ell)}W_{\ell+1}^\top\cdots W_L^\top u,
\]

with `q_L(u)=u`, and define the scalar potential

\[
\Phi_u(x)
=
\frac12\sum_{\ell=1}^L
q_\ell(u)^\top |z_\ell(x)|.
\]

Then

\[
\boxed{u^\top(P-Q)h_L=(P-Q)\Phi_u.}
\]

### Proof

The ReLU decomposition gives the exact recurrence

\[
\delta_\ell
=
\frac12W_\ell\delta_{\ell-1}
+
\frac12(P-Q)|z_\ell|.
\]

Unrolling from `δ_0=0` gives

\[
\delta_L
=
\frac12\sum_{\ell=1}^L
2^{-(L-\ell)}
W_L\cdots W_{\ell+1}(P-Q)|z_\ell|.
\]

Taking inner product with `u` produces the displayed identity.

### Source-space corollary

Let `A=[a_1,...,a_r]` be any legal target-free output source matrix and let

\[
e=(P-Q)h_L.
\]

The complete oracle normal-equation vector is

\[
b=A^\top e.
\]

Its entries have the exact scalar form

\[
\boxed{b_j=(P-Q)\Phi_{a_j}.}
\]

Thus a rank-four source does not require learning a 256-vector target. It requires four scalar integration problems.

---

## 3. T71 — exact band decomposition and the no-free-lunch corollary

For a partition

\[
0=t_0<t_1<\cdots<t_m=L,
\]

define

\[
\Phi_{u,j}(x)
=
\frac12
\sum_{\ell=t_{j-1}+1}^{t_j}
q_\ell(u)^\top|z_\ell(x)|.
\]

Then

\[
\boxed{
u^\top(P-Q)h_L
=
\sum_{j=1}^m(P-Q)\Phi_{u,j}.
}
\]

This is an exact physical checkpoint-channel decomposition. It suggests four bands near the authenticated checkpoint boundaries, but the precise partition should be frozen from source-capacity evidence.

There is also an exact skeptical identity:

\[
\boxed{
\Phi_u(x)
=
u^\top h_L(x)
-
2^{-L}u^\top W_L\cdots W_1x.
}
\]

Because `(P-Q)x=0`, evaluating the **total** potential with the same cubature is exactly equivalent to evaluating the original output projection. Therefore:

> The adjoint potential is a representation theorem, not automatically a variance reduction method.

A real innovation must make at least one band cheaper or more exactly integrable than the full output. Simply recomputing the total potential on the Kerdock cloud is a pathwise no-op.

---

## 4. T72 — exact source–noise–compute competition frontier

Normalize baseline MSE to one. Let `A` be whitened in the physical output metric,

\[
A^\top A=I_r,
\]

and let the oracle residual ratio of its span be

\[
r_*
=
\|e-AA^\top e\|^2.
\]

Suppose each oracle coefficient `b_j=a_j^T e` is estimated independently and unbiasedly using `n_j` samples, with

\[
\operatorname{Var}(\widehat b_j-b_j)=\frac{v_j}{n_j},
\]

and one sample for coefficient `j` costs a fraction `γ_j` of baseline compute. Then

\[
\mathbb E R
=
r_*+\sum_j\frac{v_j}{n_j},
\qquad
C
=
1+\sum_j\gamma_jn_j.
\]

Put

\[
S=\sum_j\sqrt{v_j\gamma_j}.
\]

Then, allowing continuous allocations,

\[
\boxed{
\min_{n_j>0}
\left(r_*+\sum_j\frac{v_j}{n_j}\right)
\left(1+\sum_j\gamma_jn_j\right)
=
(\sqrt{r_*}+S)^2.
}
\]

The optimum allocation is

\[
n_j^*
=
\frac1{\sqrt{r_*}}
\sqrt{\frac{v_j}{\gamma_j}},
\]

when `r_*>0`.

### Proof

For total added cost `c=Σγ_j n_j`, Cauchy–Schwarz gives

\[
\left(\sum_j\frac{v_j}{n_j}\right)c
\ge
\left(\sum_j\sqrt{v_j\gamma_j}\right)^2
=S^2.
\]

Equality uses `n_j proportional to sqrt(v_j/γ_j)`. The remaining scalar objective is

\[
(r_*+S^2/c)(1+c),
\]

whose minimum occurs at `c=S/sqrt(r_*)` and equals `(sqrt(r_*)+S)^2`.

### Winning condition

For the recorded `4.34×` gap, the target score ratio is

\[
t=1/4.34=0.230414746544.
\]

The path can win only if

\[
\boxed{
S<\sqrt t-\sqrt{r_*}.
}
\]

This is the correct source gate. `r_*<t` is merely the zero-cost hard gate.

| Combined contraction difficulty `S` | Maximum permissible oracle source ratio `r_*` |
|---:|---:|
| `0.000` | `0.230415` |
| `0.010` | `0.220914` |
| `0.025` | `0.207039` |
| `0.050` | `0.184913` |
| `0.075` | `0.164037` |
| `0.100` | `0.144412` |
| `0.150` | `0.108910` |
| `0.200` | `0.078409` |

Consequences:

- At `r_*=0.22`, only `S<0.010974` is allowed.
- At `r_*=0.20`, only `S<0.032802` is allowed.
- A realistic estimator with `S=0.10` requires `r_*<0.144412`.
- Therefore a source near `0.20–0.22` is worth pursuing only if its contractions are essentially exact and nearly free.

### Late-checkpoint margin

The authenticated confirmation checkpoint oracle ratios imply:

| Full checkpoint oracle | Oracle ratio `r_*` | Maximum allowable `S = sqrt(t)-sqrt(r_*)` |
|---|---:|---:|
| Layer 15 checkpoint oracle | `0.415000` | `-0.164190` |
| Layer 23 checkpoint oracle | `0.200000` | `0.032802` |
| Layer 27 checkpoint oracle | `0.117000` | `0.137963` |
| Layer 29 checkpoint oracle | `0.053900` | `0.247852` |
| Layer 30 checkpoint oracle | `0.025200` | `0.321270` |

Layer 15 cannot reach the competition target even with perfect, free recovery. Layer 23 barely leaves room. Layers 29 and 30 leave substantial room for a noisy or nonzero-cost estimator.

---

## 5. T73 — exact compiled replay at the last hidden checkpoint

Let `h_i` be the baseline activation cloud at the last hidden checkpoint and let the final ReLU layer have rows `w_j`. Define

\[
z_{ij}=w_j^\top h_i.
\]

Apply a common checkpoint shift `δ` to every particle. Put

\[
s_j=w_j^\top\delta.
\]

The exact change in output coordinate `j` is

\[
\boxed{
g_j(s_j)
=
\frac1n\sum_i
\left[
\sigma(z_{ij}+s_j)-\sigma(z_{ij})
\right].
}
\]

Hence the exact output correction is

\[
\boxed{
F(\delta)=g(W\delta)
}
\]

coordinatewise.

Each `g_j` is convex, nondecreasing, piecewise linear and 1-Lipschitz. If

\[
p_j=\frac1n\#\{i:z_{ij}>0\},
\]

then

\[
g_j(s)=p_js+r_j(s),
\]

where the exact crossing remainder is

\[
\boxed{
r_j(s)
=
\frac1n\sum_i
|z_{ij}+s|
\mathbf1_{z_{ij}(z_{ij}+s)<0}
\ge0.
}
\]

After sorting the thresholds `-z_ij` and storing prefix sums, each `g_j(s)` is evaluable in `O(log n)`. A full replay requires one matrix-vector product `Wδ` and 256 scalar searches, rather than another full-network pass.

### Strategic consequence

If the multi-checkpoint modes can be represented as a four- or five-dimensional shift at the late interface, the difficult deep-replay theorem is avoidable. Apply the repair once at the final hidden checkpoint and replay the final layer exactly.

---

## 6. Revised best path

The best constructive program is now:

1. **Late-interface source, not four internal interventions.**  
   Build four or five legal, network-covariant directions in the post-ReLU layer-30 center space that represent distinct depth bands.

2. **Measure the true source margin.**  
   Report the oracle output-span ratio `r_*` and the allowable contraction difficulty
   \[
   S_{\max}=\sqrt{1/4.34}-\sqrt{r_*}.
   \]
   Do not use a universal `0.20–0.22` gate.

3. **Exact source columns.**  
   Replay each frozen center direction through the compiled final-layer map to create target-free output columns.

4. **Only scalar labels.**  
   Estimate `A^T e` using the banded adjoint potentials. Never train all 128 A51 outputs.

5. **Variance/cost certificate before learning.**  
   For every proposed contraction observable, estimate or bound `v_j` and `γ_j`. Stop unless
   \[
   \sum_j\sqrt{v_j\gamma_j}<S_{\max}.
   \]

6. **Constant-first only.**  
   If a fixed global action exists, estimate a bounded residual around it; zero residual remains the default.

---

## 7. What remains genuinely unknown

This continuation does **not** produce the missing source directions or an absolute estimator for the band expectations. Those are now the only central constructive unknowns.

The strongest next mathematical target is:

> Find a network-covariant late-center basis whose oracle ratio is comfortably below `0.15`, and an absolute band-contraction estimator whose certified `S` fits the exact margin.

A ratio merely below `0.22` is no longer persuasive unless the contraction identity is analytic or essentially costless.

---

## 8. Provenance

The checkpoint ratios and layer-30 perturbation results were independently re-read from the authenticated source snapshot:

`oracle_continuation_20260730/source_snapshots/ORACLE_GAP_FINAL_REPORT.md`

inside `oracle_continuation_experiments_20260730.zip`. That report describes 12 base networks × 3 grouped rotations, 16 independent complete-Kerdock reference rotations per base, no protected data, and reports confirmation ratios `0.200`, `0.117`, `0.0539`, and `0.0252` at corrected post-ReLU layers 23, 27, 29, and 30 respectively.

The new identities T70–T73 are analytic deductions in this memo and should receive hostile line-by-line review before manuscript promotion.


---

## 9. Independent numerical verifier

`verify_whestbench_best_path_v23.py` checks T70–T73 on independent random finite functionals and networks.

Headline errors:

- T70 scalar projection identity: `1.344e-16`
- T70 four-column source identity: `5.967e-16`
- T71 band sum: `1.136e-16`
- T71 pointwise total-potential equivalence: `3.331e-16`
- T72 closed-form optimum: `2.776e-17`
- T73 direct replay versus coordinate formula: `2.276e-15`
- T73 sorted-threshold evaluator: `2.942e-15`
- T73 explicit crossing remainder: `2.269e-15`

All checks pass.

### Scope caveats

T72 is deliberately optimistic in three ways:

1. estimators are unbiased;
2. scalar estimation noises are independent after source whitening;
3. the score is exactly MSE ratio times deterministic compute ratio.

A biased estimator adds the physical bias penalty

\[
\mu^\top G^{-1}\mu,
\]

and correlated estimators require the corresponding joint covariance design. Neither can be ignored in a real continuation. Therefore the T72 inequality is a necessary first gate, not evidence that an estimator exists.

T73 assumes the chosen correction is applied at a checkpoint with exactly one ReLU layer remaining. The production layer indexing and scaling constants must be audited before promotion, although the finite-cloud identity itself is elementary.
