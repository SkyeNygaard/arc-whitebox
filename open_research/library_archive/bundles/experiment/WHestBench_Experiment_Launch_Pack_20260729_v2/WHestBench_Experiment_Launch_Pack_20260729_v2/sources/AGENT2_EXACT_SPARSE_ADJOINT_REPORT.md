# Agent 2 — Exact adjoint compression for 128 sparse probes

## Decision

**The 128 sparse probes do not require 128 dense adjoints.**  For the Agent-1 feature

\[
K^C_{u_p,v_p}=u_p^T C_{21}v_p
=\mathbb E[(u_p^T(x\odot x))(v_p^Tx)],
\]

the exact terminal dual is

\[
P_L^{(p)}=\operatorname{diag}(u_p),\qquad q_L^{(p)}=v_p.
\]

All probes can be pulled back through two shared bases.  Let

- \(S_U=\cup_p\operatorname{supp}(u_p)\), \(r_U=|S_U|\);
- \(V_L\in\mathbb R^{256\times r_V}\) be a basis for \(\operatorname{span}\{v_p\}\).

Set \(U_L=E_{S_U}\).  Each probe is encoded by a sparse diagonal matrix
\(G_p=\operatorname{diag}(u_{p,S_U})\) and coefficient vector \(h_p\) such that
\(v_p=V_Lh_p\).  Then

\[
P_\ell^{(p)}=U_\ell G_pU_\ell^T,\qquad q_\ell^{(p)}=V_\ell h_p,
\]

with the exact shared recursion

\[
U_\ell=A_\ell^TU_{\ell+1},\qquad V_\ell=A_\ell^TV_{\ell+1}.
\]

The number of probes affects only cheap scalar assembly; the expensive work scales with
\(r_U+r_V\), not 128.

**Recommended implementation:** exact common-basis pullback + implicit grouped source evaluation + a checkpoint/suffix estimator.  At 4,096 source rows, a 32+32 dimensional shared basis fits below 10B through suffix depth 12.  A worst-case 128+128 basis needs suffix depth 6 for a 32-coordinate replay, or depth 8 if replay is limited to 8–16 layer-31 coordinates.  A full-depth 4,096-row source rollout remains about 18–22B and should be rejected.

The branch is **not submission-ready** because the library still does not expose the actual frozen \(U,V\) arrays or a real ARC source-localization run.  The next experiment is sharply defined and can close or validate the branch.

---

## 1. Exact terminal states for all requested feature types

Let \(x\) be centered and \(T(P,q)=\mathbb E[(x^TPx)(q^Tx)]\).

### Agent-1 elementwise quadratic-Hermite feature

For

\[
(u^T(x\odot x-\operatorname{diag}\Sigma))(v^Tx),
\]

the cubic population anchor is

\[
T(\operatorname{diag}u,v)=u^TC_{21}v.
\]

Thus

\[
P_L=\operatorname{diag}(u),\qquad q_L=v.
\]

The Hermite subtraction is a lower-order linear term.  Its expectation is zero only in the same true-centered coordinates; it cannot be silently discarded in finite-rule/sample-centered control algebra.

### Directional square

For

\[
((a^Tx)^2-a^T\Sigma a)(v^Tx),
\]

use

\[
P_L=aa^T,\qquad q_L=v.
\]

### Coordinate c21 feature

For \(C_{ij}=\mathbb E[x_i^2x_j]\),

\[
P_L=e_ie_i^T,\qquad q_L=e_j.
\]

The marginal diagonal \(C_{ii}=\mathbb E[x_i^3]\) is the special case \(j=i\).
For a cubic Hermite pointwise feature \(x_i^3-3\Sigma_{ii}x_i\), keep the same cubic dual and carry the explicit linear term separately.

### Symmetric and antisymmetric c21 components

Let

\[
P_{ij}^{\times}=\frac12(e_ie_j^T+e_je_i^T),
\quad q_{ij}^{\pm}=\frac12(e_i\pm e_j).
\]

Then

\[
T(P_{ij}^{\times},q_{ij}^{+})=\frac{C_{ij}+C_{ji}}2,
\qquad
T(P_{ij}^{\times},q_{ij}^{-})=\frac{C_{ij}-C_{ji}}2.
\]

These use the same two-coordinate quadratic projection and differ only in the final linear combination.

---

## 2. Exact multi-probe adjoint formula

For probe \(p\),

\[
T_L^{(p)}=T_0(P_0^{(p)},q_0^{(p)})+
\sum_{\ell=0}^{L-1}\langle S_\ell,P_{\ell+1}^{(p)}\otimes q_{\ell+1}^{(p)}\rangle.
\]

With common bases,

\[
P_\ell^{(p)}=U_\ell G_pU_\ell^T,
\qquad q_\ell^{(p)}=V_\ell h_p.
\]

For a source cloud \(X_\ell\in\mathbb R^{H\times256}\), compute once

\[
Z_\ell=X_\ell U_\ell\in\mathbb R^{H\times r_U},
\qquad H_\ell=X_\ell V_\ell\in\mathbb R^{H\times r_V}.
\]

Then all 128 contractions are

\[
T_\ell^{(p)}=\frac1H\sum_n
(z_{\ell n}^TG_pz_{\ell n})(H_{\ell n}^Th_p).
\]

For coordinate-sparse Agent-1 probes, \(G_p\) is diagonal with very few nonzeros and \(h_p\) is sparse or low-dimensional.  Group by shared \(u\), shared \(v\), and coordinate pair so that squares, cross-products, and linear projections are reused.

### Direct local-source evaluation

Writing centered \(y=Ax+r\), define

\[
a=Ax.
\]

The exact source integrand is

\[
\begin{aligned}
&(y^TPy)(q^Ty)-(a^TPa)(q^Ta)\\
={}&(a^TPa)(q^Tr)
+2(a^TPr)(q^Ta)
+2(a^TPr)(q^Tr)\\
&+(r^TPr)(q^Ta)
+(r^TPr)(q^Tr).
\end{aligned}
\]

All five terms use only projected \(a\) and \(r\) in the shared \(U,V\) bases.  No \(256^3\) tensor is formed.

---

## 3. Agent-1 centering correction must be part of the estimator

Agent 1’s fixed-radius feature has anchor

\[
A_{u,v}(m)=\frac1{d+1}\left[
K^C_{u,v}
-(u^Tq)(v^T\delta)
-2(u\odot\delta)^TMv
+4(u^T(\mu\odot\delta))(v^T\mu)
+2(u^T\delta^{\odot2})(v^T\mu)
\right],
\]

where \(m=\mu+\delta\), \(q=\operatorname{diag}M\), and \(K^C_{u,v}=u^TC_{21}v\).

The first-order map for all probes is

\[
B_p=\frac1{d+1}\left[-(u_p^Tq)v_p
-2\operatorname{diag}(u_p)Mv_p
+4(v_p^T\mu)(u_p\odot\mu)\right].
\]

Thus

\[
e_{\rm center}=B\delta+
\frac{2}{d+1}(V\mu)\odot U(\delta^{\odot2}).
\]

This is a shared mean-defect calculation, not 128 independent anchors.  The required batched operations are:

1. \(Mv_p\) for all probes: one \(256\times256\) by \(256\times128\) GEMM;
2. \(Uq\), \(V\delta\), \(U(\mu\odot\delta)\), and \(U(\delta^2)\): sparse matrix-vector products;
3. one 128-by-256 center Jacobian if an explicit map is desired.

The synthetic Agent-1 test found center-map rank 19 in dimension 20, with 90% energy in 6 modes and 99% in 9.  This supports a separate low-rank center channel, but the real frozen arrays are required before fixing a rank.

---

## 4. FLOP counts

All counts use multiply-add = 2 FLOPs, width \(n=256\), 128 probes, and 30 pullbacks.

### Dense pullback

Per probe/layer:

\[
A^TPA:4n^3,\qquad A^Tq:2n^2.
\]

For 128 probes and 30 layers:

\[
258.20\text{B FLOPs}.
\]

Reject.

### Independent signed rank-one pullbacks

For rank-one \(P=rr^T\), propagate \(r\) and \(q\):

\[
2n^2(1+1)
\]

per probe/layer, or \(1.007\)B total.  This is algebraically affordable, but source projections remain duplicated and dominate.

### Shared exact basis pullbacks

\[
C_{\rm pull}=2Ln^2(r_U+r_V).
\]

Examples:

| \(r_U+r_V\) | Full 30-layer pullback |
|---:|---:|
| 64 | 0.252B |
| 128 | 0.503B |
| 192 | 0.755B |
| 256 | 1.007B |

### Source rollout and implicit contractions

For \(H\) source rows and suffix depth \(d\):

\[
C_{\rm source\ forward}=2Hdn^2,
\]

\[
C_{\rm projection}=2Hn(d+1)(r_U+r_V).
\]

Sparse layer-31 replay for \(K\) corrected coordinates costs

\[
C_{\rm replay}=2N_KKn,
\qquad N_K=66{,}048.
\]

This is 0.270B, 0.541B, and 1.082B for \(K=8,16,32\).  It is much cheaper than replaying the complete 256-by-256 final layer.

### Total 4,096-row cost estimates

The following include shared pullback, source forward, source projections, conservative sparse probe assembly, and sparse layer-31 replay.

| Shared dimension \(r_U+r_V\) | Replay K | d=4 | d=6 | d=8 | d=12 | d=16 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 3.15B | 4.53B | 5.90B | 8.64B | 11.38B |
| 64 | 32 | 3.97B | 5.34B | 6.71B | 9.45B | 12.19B |
| 128 | 16 | 4.13B | 5.79B | 7.44B | 10.76B | 14.07B |
| 192 | 32 | 5.38B | 7.32B | 9.26B | 13.14B | 17.03B |
| 256 | 8 | 5.27B | 7.50B | 9.72B | 14.18B | 18.63B |
| 256 | 16 | 5.54B | 7.77B | 9.99B | 14.45B | 18.90B |
| 256 | 32 | 6.08B | 8.31B | 10.53B | 14.99B | 19.44B |

Consequences:

- If the frozen probe span is approximately 32+32, d=12 is under 10B.
- If it is approximately 64+64, d=8 is comfortably under 10B; d=12 is around 11B.
- In the worst 128+128 case, d=6 is under 10B for K=32; d=8 is under 10B only with K<=16.
- d=12 with a full 128+128 span exceeds the 14B meaningful-gain ceiling.
- Full-depth 4,096-row evaluation is rejected.
- A 1,536-row full-depth source stream is roughly 8.9B for a 32+32 span, but accuracy must be validated; row reduction is not an algebraic guarantee.

---

## 5. Exact oracle source-localization experiment

Use cumulative suffix depths

\[
d\in\{1,2,3,4,6,8,12,16,30\}.
\]

For each protected network:

1. Load the exact frozen \(U,V\), layer, Kerdock rotation, folds, ridge, coefficients, and shrinkage.
2. Use one fixed sequence of local maps \(A_\ell\) for both truth and Kerdock decompositions.
3. Build all shared dual states \(U_\ell,V_\ell\).
4. In true centered coordinates, compute probe contractions and source terms for the high-precision reference:
   \[
   s_\ell^*=T_{\ell+1}^*-T_\ell^*(A_\ell^TPA_\ell,A_\ell^Tq).
   \]
5. Compute the identical quantities on the protected Kerdock cloud, including the exact Agent-1 lower-order recentering:
   \[
   s_\ell^K.
   \]
6. Form source defects \(\Delta s_\ell=s_\ell^*-s_\ell^K\).
7. For every suffix depth, construct
   \[
   \hat T_L^{(d)}=T_L^K+\Delta T_0+
   \sum_{\ell=L-d}^{L-1}\Delta s_\ell,
   \]
   where \(\Delta T_0\) is zero for an input rule exact on cubics, but should still be checked numerically.
8. Insert \(\hat T_L^{(d)}\) into the unchanged sparse-control harness.
9. Replay exactly the frozen top-8/12/16/32 layer-31 coordinate sets using the sparse \(N_K\times K\) by \(K\times256\) update.

Report for each d and K:

- scalar-anchor residual norm divided by the full true-minus-Kerdock defect;
- correction cosine;
- signed cumulative source contribution, not absolute source energy;
- raw final-layer MSE ratio;
- fraction of exact sparse-control oracle gain retained;
- wins, median, bootstrap interval, and worst network;
- measured FLOPs and residual wall time.

Continuation gates:

- exact d=30 source sum agrees with direct anchor below 1e-8 relative error;
- d<=12 retains >=90% of exact sparse-control gain;
- added compute <10B preferred, <14B absolute ceiling;
- no bad worst-network tail after frozen shrinkage.

Agent 1’s independent layer-channel data shows that the layer-31 mean channel is strong, but it does **not** prove cubic source localization. The improvement rises from 62.69% removed at layer 24 to 75.45% at layer 28 and 82.69% at layer 31. This supports checkpoint-plus-suffix testing, while warning that a 3-layer suffix alone may omit substantial inherited defect.

---

## 6. Checkpoint consistency

A checkpoint estimator is exactly consistent when

\[
\hat T_L=
\hat T_c(P_c,q_c)+\sum_{\ell=c}^{L-1}\hat s_\ell
\]

uses:

- the same centered coordinates at every layer;
- the same local maps \(A_\ell\);
- source terms defined as residuals relative to those maps;
- a checkpoint contraction and source estimator targeting the same statistical process.

If \(T_c^K\) is inherited from the Kerdock cloud, the true correction is

\[
T_L^*-T_L^K=
(T_c^*-T_c^K)+\sum_{\ell=c}^{L-1}(s_\ell^*-s_\ell^K).
\]

Therefore “sample checkpoint + regenerated final sources” is a mathematically valid **hybrid** but is not an exact true anchor unless the checkpoint defect is negligible or separately estimated.  Same-cloud empirical sources telescope back to the sample anchor and provide no external correction.

---

## 7. Compression recommendation

### Dense
Reject.  Approximately 258B just for pullbacks.

### Independent signed low-rank
Exact for rank-one probes and only about 1B for pullbacks, but it duplicates source projections.  Use only as a reference implementation.

### Common basis
**Primary recommendation.**  It is exact, exploits the actual sparse \(u\)-support and \(v\)-span, and makes pullback cost negligible relative to source generation.

### Implicit source evaluation
**Mandatory.**  Project source rows into shared bases and evaluate grouped coordinate squares/cross-products.  Never form dense \(P\) matrices or \(K_3\).

### Randomized/common low-rank truncation
Do not use initially.  The sandbox prototype found that truncation can produce moderate scalar-anchor error while catastrophically worsening downstream control MSE.  Rank must be selected on scalar-anchor and final-control error, not Frobenius energy.  The only low-rank object with current positive evidence is the separate center-error map B, whose synthetic rank90 was 6.

### Suffix-local
**Required at H=4096 unless the frozen shared span is very small.**  Start with d=4,6,8,12.  Run d=16 only if localization shows a large incremental gain.  Reject full depth at 4,096 rows.

---

## 8. Numerical validation completed in the sandbox

The standalone prototype verifies:

- explicit tensor and implicit contraction: exact to displayed precision;
- batched adjoint/source reconstruction: error below 1.4e-17 in the synthetic tests;
- center-shift identity: max absolute error 1.1e-15;
- exact Agent-1 terminal \(P=\operatorname{diag}(u),q=v\): max relative error 1.5e-13;
- 30 float32 pullbacks versus float64: relative basis error about 1.35e-6 in a stress test;
- exact full-rank common-basis reconstruction survives 30 layers;
- aggressive basis truncation can destroy downstream control despite apparently moderate scalar error.

Use float64 for dual bases and source accumulation.  Stabilize by column-wise rescaling, which preserves sparse diagonal \(G_p\).  Avoid frequent QR if it densifies every \(G_p\); if QR/SVD is used, transform coefficients exactly and charge the resulting dense probe-evaluation cost.

---

## 9. Implementation specification

1. **Manifest loader**
   - load frozen `U[128,256]`, `V[128,256]`;
   - verify hashes and exact probe order;
   - compute `support_U`, rank-revealing QR/SVD of `V.T`, and report `r_U,r_V`.

2. **Terminal builder**
   - `Bp = I[:, support_U]`;
   - `G[p] = diag(U[p,support_U])`;
   - `Bq = orthonormal_basis(span(V.T))`;
   - `h[p] = V[p] @ Bq`.

3. **Backward pass**
   - batched GEMM `Bp = A.T @ Bp`, `Bq = A.T @ Bq`;
   - float64 accumulation;
   - optional column norm balancing with exact diagonal coefficient rescaling.

4. **Source pass**
   - for each retained suffix layer, project local source rows once into `Bp,Bq`;
   - group probes by nonzero U coordinate and V coefficient pattern;
   - evaluate all source contractions in one pass;
   - store per-layer signed 128-vector contributions.

5. **Center correction**
   - compute cubic adjoint vector `Kc[128]`;
   - compute `B @ delta` and the exact quadratic delta term;
   - divide cubic term by `257` at width 256;
   - reproduce Agent-1 raw-versus-connected identity before scoring.

6. **Control/replay**
   - do not refit coefficients or shrinkage;
   - apply the frozen sparse control;
   - replay only the frozen 8/12/16/32 layer-31 coordinates with a reduced GEMM.

7. **Diagnostics**
   - direct-versus-adjoint anchor agreement;
   - center correction on/off;
   - source depth curves;
   - scalar-anchor error and downstream final MSE;
   - measured FLOPs and wall time;
   - per-network worst case.

---

## Final gate status

- **Exact algebra:** passed in sandbox.
- **128 probes require independent adjoints:** no; disproved structurally.
- **Numerical stability over 30 layers:** passed in float64; float32 is not recommended.
- **Under 10–14B feasibility:** conditionally passed for suffix depth 6–12, depending on the measured frozen \(r_U+r_V\) and replay K.
- **Retains 90% of real exact sparse-control gain:** not yet measured because the actual frozen arrays and protected corpus are absent from the shared artifacts.
- **Branch verdict:** continue only to the single frozen source-localization experiment above.  Close immediately if d<=12 fails the 90%-gain gate or if the measured shared span pushes the winning depth above 14B.
