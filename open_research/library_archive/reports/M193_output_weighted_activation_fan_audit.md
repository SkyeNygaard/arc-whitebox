# M193 Output-Weighted Activation-Fan Audit

**Date:** 2026-07-31  
**Verdict:** **FAIL — close M193 as a low-output-weighted-normal-rank compression route.**

This is a scoped empirical closure, not a theorem that every boundary-current identity is impossible. The exact facet-current representation remains valid, but the declared compression mechanism fails its first promotion criterion on the development networks.

## 1. Exact identity

Let the bias-free row-vector ReLU network be

\[
h_0(x)=x,\qquad p_\ell(x)=h_{\ell-1}(x)W_\ell,\qquad h_\ell(x)=\operatorname{ReLU}(p_\ell(x)),\qquad f(x)=h_L(x).
\]

On a complete activation history \(R\), the network is linear. For a realized facet of gate \((\ell,j)\), write

\[
g_F=\nabla_x p_{\ell j}(x),\qquad n_F=\frac{g_F}{\|g_F\|},\qquad s_F=\frac{\partial f}{\partial h_{\ell j}}.
\]

With output-by-input Jacobian convention, opening that gate creates the rank-one jump

\[
J_f^+-J_f^-=s_F g_F^\top.
\]

Hence, distributionally,

\[
\Delta f=\sum_F s_F\|g_F\|\,\mathcal H^{d-1}\!\restriction_F.
\]

Positive homogeneity gives \(f(x)=J_f(x)x\) almost everywhere, and Gaussian Stein integration gives

\[
\boxed{
\mathbb E[f(X)]
=
\mathbb E[\Delta f(X)]
=
\sum_F\int_F s_F(x)\|g_F(x)\|\phi_d(x)\,d\mathcal H^{d-1}(x).
}
\]

The sum is over realized facet pieces, not merely named gates. A named gate surface is subdivided by upstream and downstream activation histories because both its input gradient and downstream adjoint are piecewise constant. Codimension-two intersections have zero surface measure but create first-order smoothing bias near intersecting facets.

An equivalent coarea approximation is

\[
J_F
=
\lim_{\varepsilon\downarrow0}
\mathbb E\!ig[s_F(X)\|g_F(X)\|^2\kappa_\varepsilon(p_F(X))\big],
\]

where \(\kappa_\varepsilon\) is an approximate delta kernel. Finite sample size, finite bandwidth, stochastic gradient-norm estimation, and history aggregation are approximations.

For a first-layer gate, \(p_{1j}(x)=w_j^\top x\) is globally linear, so the surface integral has an exact conditional form:

\[
\boxed{
c_j
=
\phi(0)\|w_j\|\,
\mathbb E\!\left[s_{1j}(X)\mid w_j^\top X=0\right].
}
\]

A conditional sample is generated exactly by \(X=Z-n_j(n_j^\top Z)\), with \(Z\sim N(0,I)\).

## 2. Operators audited

For each first-layer gate, let \(c_j\in\mathbb R^{256}\) be its integrated final-output current and \(n_j=w_j/\|w_j\|\).

### Input-normal energy operator

\[
C_N=\sum_{j=1}^{256}\|c_j\|_2^2 n_jn_j^\top.
\]

Equivalently, \(C_N=BB^\top\) for

\[
B=[\|c_1\|n_1,\ldots,\|c_{256}\|n_{256}].
\]

This directly answers whether the output-relevant input normals lie in a small shared input subspace.

### Output-direction-preserving operator

To avoid discarding the directions of the output currents, define columns

\[
b_j=c_j\otimes n_j.
\]

Its Gram matrix is

\[
G_{ij}=(n_i^\top n_j)(c_i^\top c_j).
\]

The singular spectrum of this operator was computed from \(G\) without materializing a \(65{,}536\times256\) matrix.

### Signed-capture diagnostic

For the leading input-normal eigenspace \(U_r\), define \(a_j^2=\|U_r^\top n_j\|^2\) and

\[
J_{1,r}=\sum_j a_j^2 c_j.
\]

Reported signed ranks are the first \(r\) for which

\[
1-\frac{\|J_1-J_{1,r}\|^2}{\|J_1\|^2}
\]

crosses 90%, 99%, or 99.9%. This is an oracle diagnostic along the energy-optimal normal basis, not a proof that no specially optimized signed subspace could do better.

## 3. Experimental design

- Networks: development-screen seeds 3000–3007.
- Width/input/output dimension: 256.
- Depth: 32.
- Exact bundled float32 weight generator.
- Seeds 3000–3003: 1,024 conditional samples per first-layer gate, or 262,144 conditioned paths per network.
- Seeds 3004–3007: 512 conditional samples per gate, or 131,072 conditioned paths per network.
- No target values were used to construct the normal basis or weights.
- The archived Gaussian references were used only for current-scale and identity diagnostics.
- Independent half-sample current cosine: 0.746–0.923; energy ranks were substantially more stable than the signed aggregate current.

## 4. Audit A results

| Seed | Samples/gate | Stable rank | Energy r90 | r99 | r99.9 | Output-preserving r90 | r99 | r99.9 | Signed r90 | r99 | r99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 1,024 | 31.63 | 100 | 176 | 218 | 128 | 228 | 252 | 120 | 154 | 191 |
| 3001 | 1,024 | 27.63 | 107 | 183 | 221 | 150 | 235 | 253 | 97 | 153 | 185 |
| 3002 | 1,024 | 33.65 | 112 | 185 | 222 | 159 | 238 | 254 | 93 | 141 | 181 |
| 3003 | 1,024 | 32.74 | 114 | 187 | 224 | 166 | 240 | 254 | 67 | 128 | 172 |
| 3004 | 512 | 30.53 | 104 | 179 | 220 | 139 | 231 | 253 | 46 | 125 | 169 |
| 3005 | 512 | 23.07 | 111 | 186 | 223 | 163 | 241 | 254 | 85 | 138 | 180 |
| 3006 | 512 | 30.21 | 105 | 182 | 221 | 151 | 238 | 254 | 87 | 131 | 170 |
| 3007 | 512 | 29.20 | 103 | 179 | 220 | 134 | 232 | 253 | 91 | 132 | 168 |

Aggregate:

- ordinary rank: **256/256 on every network**;
- stable rank: mean **29.83**, range **23.07–33.65**;
- input-normal energy rank:
  - 90%: mean **107.0**, range **100–114**;
  - 99%: mean **182.1**, range **176–187**;
  - 99.9%: mean **221.1**, range **218–224**;
- output-direction-preserving energy rank:
  - 90%: mean **148.8**, range **128–166**;
  - 99%: mean **235.4**, range **228–241**;
  - 99.9%: mean **253.4**, range **252–254**;
- signed capture along the leading normal-energy basis:
  - 90%: mean **85.8**, range **46–120**;
  - 99%: mean **137.8**, range **125–154**;
  - 99.9%: mean **177.0**, range **168–191**.

The normal geometry is not being inflated by duplicate directions. Across all eight networks, the largest absolute pairwise normal cosine was only **0.265**. There were zero pairs with absolute cosine at least 0.9, 0.95, 0.99, or 0.999, so parallel-normal grouping leaves all 256 first-layer normals separate.

For the four higher-sample networks, the leading input-normal subspace captures only about 30–33% of energy at rank 16, 48–53% at rank 32, 71–77% at rank 64, and 93–95% at rank 128. Using the exact Gaussian sign-mismatch formula, the current-energy-weighted gate sign mismatch remains approximately:

- rank 16: **31–32%**;
- rank 32: **23.5–25.3%**;
- rank 64: **14.1–16.7%**;
- rank 128: **5.1–7.3%**.

This is not a small normal fan by any winning-scale interpretation.

## 5. Audit B: localization and cancellation

### First-layer neuron currents

The integrated first-layer currents cancel strongly in absolute mass:

- higher-sample cancellation ratio \(\|\sum_jc_j\|/\sum_j\|c_j\|\): **2.7–6.0%**;
- the first-layer signed current norm is **3.3–7.9%** of the final mean norm.

This cancellation cannot be treated as permission to discard the first layer. The per-coordinate squared norm of the omitted first-layer current is approximately \(1.19\times10^{-3}\) to \(5.00\times10^{-3}\) on the four primary networks, thousands of times larger than a competitive final MSE. A valid method therefore needs an exact or tightly certified telescope, not heuristic pruning.

### Exploratory all-layer coarea audit

A corrected Gaussian-kernel coarea estimator was run with all ancestor sources included. On seed 3000 with 8,192 Gaussian samples, 8 Hutchinson input-gradient probes, and bandwidth 0.05, it reconstructed the archived final mean with:

- norm ratio: **1.025**;
- cosine: **0.988**;
- relative vector error: **0.158**;
- estimated standard-error norm ratio: **0.117**.

The four eight-layer block currents had norms of approximately **0.732, 0.769, 0.893, and 0.960** times the final-mean norm, with signed projection fractions **+0.667, +0.382, −0.331, +0.296**. This is broad, alternating cancellation, not late-layer domination or an adjacent-layer telescope.

The same finite-bandwidth estimator was not accurate enough on seeds 3001–3003: relative identity errors were roughly 0.55–1.05 at bandwidth 0.05. Those layerwise values are therefore rejected rather than used as evidence. The failure itself confirms that naive boundary-current sampling is high-variance and not an economical estimator.

**Audit-B conclusion:** strong cancellation exists, but no stable weight-derived localization rule or exact telescope was found. The available evidence points to broad, delicate cancellation rather than a small certified subset.

## 6. Exact map-sharing audit

Generic exact BDD/ZDD merging cannot rescue the branch. Distinct feasible activation histories that differ on a live input-output path have distinct affine maps almost surely under an absolutely continuous weight law. Exact collisions remain possible only through dead paths, redundant constraints, or metric-annihilated differences. No result here relies on map equality.

## 7. Cost

The primary first-layer audit uses 262,144 conditioned paths per network. Each path requires 32 forward dense transforms and 31 tangent/adjoint dense transforms:

\[
262{,}144\times63\times2\times256^2
=
2.165\times10^{12}\text{ FLOPs/network}.
\]

That is about **7.81×** the dense-equivalent 129-basis 32-layer path cost of 277.0B FLOPs. Four primary networks cost about **8.66T FLOPs**. This is acceptable as an offline oracle falsifier, not as a submission algorithm.

Even an oracle truncation to roughly 100 input-normal directions would still require evaluating conditional boundary integrals and downstream sensitivities for those directions. No legal analytic rule for those integrals was found, and the retained rank is already too large to create winning economics.

## 8. Promotion-gate decision

1. **Very small output-weighted normal span:** **FAIL.** First layer alone is full rank; 90% energy needs 100–114 input directions and 128–166 output-preserving directions.
2. **Strong signed localization or telescope:** **FAIL / not demonstrated.** First-layer currents cancel, but require many leading normal directions; the validated all-layer slice shows broad alternating blocks, and the cross-network smoother is too noisy for certification.
3. **Weight-derived stable selection:** **FAIL.** No rule selects a small subset or subspace with a certified residual.
4. **Cheaper winning-scale estimator:** **FAIL.** No candidate was formulated because the prerequisite audits did not pass.

# Final conclusion

## **FAIL**

Close **M193 as the low-output-weighted-normal-rank, current-localized activation-fan route**.

Preserve only:

- the exact facet-current identity;
- rank-one gate-opening Jacobian updates;
- the theorem that exact live-map sharing is generically absent;
- boundary-current machinery as a possible internal tool for a separately derived algebraic identity.

Do **not** preserve M193 as a deployable or high-priority competition branch merely because absolute currents cancel. Reopening would require a new proved all-layer telescope that analytically eliminates the high-rank early currents and includes inherited ancestor terms. That would be a new theorem/class, not continuation of the failed low-normal-rank audit.

## Proposed ledger patch

| ID | Evidence | Result | Verdict | Status |
|---|---|---|---|---|
| M193 | Exact first-layer conditional boundary integration on screen seeds 3000–3007; approximate all-layer coarea diagnostic | First-layer output-weighted normal operator has rank 256 on every network. Input-normal r90/r99/r99.9 = 100–114 / 176–187 / 218–224; output-preserving r90/r99/r99.9 = 128–166 / 228–241 / 252–254. No near-parallel grouping. Strong but delicate current cancellation; no certified all-layer telescope or cheap boundary integral. | Close the low-normal-rank/current-localized facet-DAG route. Exact boundary identity remains valid but non-deployable. | Closed bounded falsifier |
