# M189 Kerdock-Index QTT Falsifier

## Final verdict

# **FAIL**

M189 does not produce a promotable competition candidate.

This verdict has two different confidence levels:

1. **Shared or layerwise Kerdock QTT is structurally closed.** The first-layer tensor already has an exact 256-dimensional neuron interface, and the first nonlinear even interaction becomes almost maximally dense in Kerdock-Walsh space.
2. **Direct final-output scalar QTT remains mathematically unresolved, but fails the bounded experiment operationally.** The retained artifacts do not contain the nodewise final-output tensors needed to measure its ranks, singular spectra, common pivots, or legal reconstruction errors. The canonical v31 ledger itself records M189 as “not run.”

The prompt requires a PASS only if low ranks, a held-out common query rule, a small query union, signed mean preservation, and a winning adjusted score are all demonstrated. It also requires closure when constructing or validating the tensor needs the complete output array. None of the promotion conditions has been established.

This is therefore a **competition-path FAIL**, not a universal theorem that a fortuitous low-rank final-output tensor can never occur.

---

## 1. Artifact audit

I searched the v31 ledger, current-state memo, Agent 5 materials, Kerdock estimator packages, experiment-launch archives, T0 instrumentation bundles, and the large nested partial-MUB archive.

### Retained artifacts actually available

| Artifact | Contents relevant to M189 | Sufficient for QTT audit? |
|---|---|---:|
| `kerdock_mub5_seed3.npz` | `chirps` of shape `128×256`; rotation of shape `256×256` | Only for defining the nodes |
| T0 evidence arrays | Aggregated estimator outputs of shape `256` | No |
| Nested partial-MUB network vectors | Final means, reference means, fitted corrections, and block summaries such as `ext_block_means` of shape `17×256` | No |
| Existing per-basis results | Basis or block means | No |
| Nodewise chirp outputs | Required shape `65,536×256` per network/layer | **Not present** |
| Legal TT-cross query transcripts | Queried indices, sweeps, pivots, and reconstructions | **Not present** |

The prior experiment report refers to “stored per-basis outputs,” but basis means are not per-node outputs and cannot recover dependence on the eight within-basis row bits. The v31 ledger consequently and correctly still labels M189 an unrun existing-array audit.

This artifact distinction is decisive:

\[
\text{per-basis mean}
\neq
\text{the }256\text{ within-basis output rows}.
\]

A tensor over basis means has only seven binary modes and cannot answer whether the original Kerdock function has low QTT rank or can be queried cheaply.

---

## 2. Exact tensor definition

Let

\[
a\in\{0,\ldots,127\},\qquad
b\in\{0,\ldots,255\},\qquad
s\in\{0,1\},
\]

where:

- \(a\) is the seven-bit Kerdock basis or chirp index;
- \(b\) is the eight-bit Walsh row index within the basis;
- \(s\) selects the antipodal sign;
- \(j\in\{1,\ldots,256\}\) is the output coordinate.

The estimator’s flattening convention is

\[
i=((256a+b)2+s),
\]

so \(s\) is the fastest-varying index, followed by the row index \(b\), then the basis index \(a\). The corresponding natural most-significant-to-least-significant QTT ordering is

\[
(a_6,\ldots,a_0,b_7,\ldots,b_0,s).
\]

Define the raw final-output tensor

\[
F_W(a,b,s,j)
=
h_{32}(x_{a,b,s})_j
\in
\mathbb R^{2^7\times2^8\times2\times256}.
\]

The main chirp component therefore has exactly

\[
2^{7+8+1}=65{,}536
\]

nodes. The complete 129-basis design also contains 512 signed coordinate nodes, which are not naturally part of this \(2^{16}\) tensor.

### Mean-relevant antipodal tensor

The preferable mean-relevant object is

\[
E_W(a,b,j)
=
\frac{
F_W(a,b,0,j)+F_W(a,b,1,j)
}{2},
\]

a 15-bit tensor of shape

\[
2^7\times2^8\times256.
\]

Its contraction gives the chirp contribution to the estimator mean:

\[
\overline E_W(j)
=
\frac1{2^{15}}
\sum_{a,b}E_W(a,b,j).
\]

Antipodal averaging is exact for the integration problem, not an approximation.

### Objects that do not qualify

| Object | Reason it does not establish a viable M189 estimator |
|---|---|
| Basis mean \(B(a,j)=2^{-8}\sum_b E(a,b,j)\) | Computing one entry already requires reading all within-basis rows |
| Constant tensor equal to the full Kerdock mean | Rank one, but requires knowing the answer |
| Residual after subtracting the true target mean | Target leakage |
| Target-fitted signed baseline-error proxy | Cannot be queried on a new network |
| TT-SVD adjusted to preserve the full-array mean | Reads the complete tensor and can leak its constant component |

A constant rank-one tensor can reproduce the mean perfectly after the full mean is known, which is why full-array mean error alone is not a legal representation test.

---

## 3. Exact structural rank results

Although the final-output rank audit could not be run, several exact facts already constrain the branch.

### 3.1 Shared neuron/output representation has rank 256

Let

\[
X_K\in\mathbb R^{65{,}536\times256}
\]

be the Kerdock-node matrix. It spans the full input space:

\[
\operatorname{rank}(X_K)=256.
\]

For first-layer weights \(W_0\),

\[
Z_1=X_KW_0.
\]

A continuously distributed square \(W_0\) is nonsingular almost surely, so

\[
\boxed{\operatorname{rank}(Z_1)=256\quad\text{almost surely}.}
\]

Consequently, any shared tensor network with an edge separating all index modes from the neuron mode needs bond rank at least 256.

This rules out:

- one rank-16 shared TT for all first-layer neurons;
- a low-rank shared output interface obtained merely because every query returns all 256 outputs;
- layerwise propagation that treats the output/neuron mode as a cheap small core.

Separate scalar TTs avoid this exact bond, but they must then demonstrate that their pivot sets substantially overlap.

### 3.2 Linear Kerdock preactivation is sparse in Walsh space

For a scalar first-layer preactivation,

\[
z(a,b,s)
=
2^{-4}
\sum_{t\in\mathbb F_2^8}
w_t(-1)^{s+a\cdot q(t)+b\cdot t},
\]

the Walsh transform has exactly 256 atoms, almost surely.

This is the strongest favorable structural fact for M189—but it does not survive the first nonlinear even interaction.

### 3.3 The quadratic even component is almost maximally dense

The Kerdock derivative property implies

\[
\operatorname{supp}\widehat{z^2}
=
\{(0,0,0)\}
\cup
\{(v,u,0):u\neq0,\;v\in\mathbb F_2^7\}.
\]

Therefore,

\[
\left|
\operatorname{supp}\widehat{z^2}
\right|
=
1+255\cdot128
=
32{,}641.
\]

Only \(32{,}768\) sign-even frequencies exist, so

\[
\boxed{
\frac{32{,}641}{32{,}768}
\approx99.61\%.
}
\]

For generic continuous weights, all these coefficients are nonzero.

This does not by itself prove high TT rank—dense tensors can be separable—but it disproves the hoped-for preservation of sparse Kerdock phase support through ReLU-like nonlinear interactions.

### 3.4 Unfinished first-layer rank certificate

The natural sharper object is the matrix

\[
C_w(v,u)=2w_tw_{t+u},
\qquad
D_uq(t)=v,
\]

of shape \(128\times255\). If

\[
\operatorname{rank}(C_w)=128,
\]

then the natural \(7\mid8\) QTT cut for \(z^2\) has rank at least 128. The previous report proposed this test but explicitly did not complete either a symbolic nonvanishing-minor proof or the empirical rank calculation.

It must therefore remain **unproven**, not silently promoted to a theorem.

---

## 4. Requested rank table

No honest numerical final-output rank table can be populated from the retained artifacts.

| Rank cap | Exact final-output rank/singular decay | Frobenius reconstruction error | Mean reconstruction error | Status |
|---:|---:|---:|---:|---|
| 4 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |
| 8 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |
| 12 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |
| 16 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |
| 24 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |
| 32 | Not measurable | Not measurable | Not measurable | Missing nodewise tensor |

Reporting invented or surrogate ranks here would violate the prompt. The required tests concern the mean-relevant Kerdock-index function and must not be replaced by ranks of basis means, activation covariance, or another object that cannot compute the required mean more cheaply.

### Representation sizes

For reference, a 15-bit scalar TT whose bond ranks are capped at \(r\) has the following maximal core parameter counts after respecting the small dimensions near the two ends:

| Rank cap \(r\) | Core parameters | TT-manifold dimension after gauge |
|---:|---:|---:|
| 4 | 392 | 192 |
| 8 | 1,320 | 640 |
| 12 | 2,568 | 1,248 |
| 16 | 4,264 | 2,048 |
| 24 | 7,976 | 3,840 |
| 32 | 12,968 | 6,144 |

These are representation dimensions, not sufficient query counts. Stable cross approximation normally needs oversampling, multiple sweeps, pivot conditioning, and a stopping test.

---

## 5. Pivot-overlap audit

No cross-approximation run or saved pivot set was found.

| Sharing question | Measured result |
|---|---|
| Across output neurons | Not run |
| Across networks | Not run |
| Across layers | Not run |
| Held-out common pivot rule | Not defined |
| Union of queried entries | Not measured |
| Pivot Jaccard overlap | Not measured |
| Frozen stopping rule | Not defined |

The exact rank-256 first-layer row-by-neuron interface shows why “one query returns all outputs” is insufficient. Values for all outputs are returned, but the outputs need not admit the same interpolation subspace or pivot sequence.

A separate rank-16 scalar cross for every output is not a legal \(Q\)-query method unless the **union** of all requested Kerdock nodes remains near \(Q\).

---

## 6. Query and FLOP accounting

The existing production Kerdock computation costs approximately

\[
175.62\text{ billion effective FLOPs}
\]

over 66,048 rows. This gives an empirical amortized cost of

\[
c_{\rm row}
=
\frac{175.62\times10^9}{66{,}048}
\approx
2.659\times10^6
\]

effective FLOPs per full-depth input.

The winning-scale compute gate is

\[
C\le27.2\text{ billion effective FLOPs},
\]

with raw MSE at most

\[
2.962\times10^{-7}.
\]

### If the 512 coordinate nodes are retained exactly

The coordinate component alone costs

\[
512c_{\rm row}
\approx1.3614\text{ billion FLOPs}.
\]

The chirp-query rungs then cost:

| Queried chirp nodes | Chirp propagation | Plus 512 coordinate nodes | Remaining under 27.2B |
|---:|---:|---:|---:|
| 1,024 | 2.723B | 4.084B | 23.116B |
| 2,048 | 5.446B | 6.807B | 20.393B |
| 4,096 | 10.891B | 12.253B | 14.947B |
| 6,144 | 16.337B | 17.698B | 9.502B |
| 8,192 | 21.782B | 23.144B | 4.056B |
| 10,240 | 27.228B | 28.589B | **over budget by 1.389B** |

Thus the previously proposed 10,240-query rung cannot retain the coordinate nodes and remain under the compute floor even before charging:

- TT fitting;
- pivot selection;
- multiple sweeps;
- first-layer diagnostics;
- mean contraction;
- memory management;
- residual wall time.

### Antipodal pairing tightens the sampling budget

A direct query of the paired tensor

\[
E(a,b)=\frac{F(a,b,0)+F(a,b,1)}2
\]

requires two full-network evaluations. After retaining the 512 coordinate nodes, the absolute maximum is therefore only

\[
\left\lfloor
\frac{27.2\text{B}-1.3614\text{B}}
{2c_{\rm row}}
\right\rfloor
=
4{,}858
\]

paired tensor entries, with zero allowance for fitting overhead.

This does not prove rank 16 impossible, but it leaves very little room for robust common-pivot discovery across 256 outputs.

### Complete-array TT-SVD is noncompetitive

Reading all 65,536 chirp nodes costs approximately

\[
65{,}536c_{\rm row}
\approx174.26\text{ billion FLOPs},
\]

before the coordinate nodes or TT-SVD itself.

Therefore a low-rank tensor discovered after reading the full array is:

- an oracle representation result;
- roughly \(6.4\times\) above the 27.2B compute gate in network evaluations alone;
- not evidence of a legal cheap estimator.

---

## 7. Mean-error and held-out curves

The required held-out curves cannot be calculated because neither the full node tensors nor a frozen legal cross transcript exists.

| Required curve | Result |
|---|---|
| Rank versus Frobenius error | Unavailable |
| Rank versus Kerdock-mean error | Unavailable |
| Query count versus mean error | Unavailable |
| Query count versus independent-truth MSE | Unavailable |
| Network-wise signed correction preservation | Unavailable |
| Worst-network and upper-tail behavior | Unavailable |
| Adjusted score versus query count | No candidate reconstruction exists |

This absence is itself consequential. The deterministic approximation error obeys

\[
\widetilde K-\mu=e+\delta,
\]

so

\[
\operatorname{MSE}(\widetilde K)
=
M_K
+
2\mathbb E\langle e,\delta\rangle
+
\mathbb E\|\delta\|^2.
\]

Without measuring the signed cross term, small tensor reconstruction error does not imply a better estimator. Under the less favorable baseline reconciliation, the conservative additional squared-error allowance may be only about \(1.94\%\) of Kerdock’s raw MSE.

A block-residual correction does remove the fixed cross term, but it would need the surrogate to explain approximately 96%–99.5% of complete-basis residual variance, depending on the number of correction blocks. No stored artifact demonstrates this.

---

## 8. Promotion-gate decision

| Promotion requirement | Evidence | Decision |
|---|---|---:|
| Mean-relevant rank roughly 12–16 or lower | No final-output ranks | Fail |
| One legal held-out pivot rule | No rule or transcript | Fail |
| Query union much smaller than current design | No pivot union | Fail |
| Signed mean corrections and tails preserved | No reconstruction | Fail |
| Complete adjusted score beats baseline | No estimator | Fail |

Because the gate is conjunctive, failure to establish any one item prevents promotion. Here all five are absent.

---

## 9. Kill-condition decision

| Kill condition | Finding |
|---|---|
| Relevant ranks materially exceed 16 | Not established for final outputs; structural rank 256 established for the shared first-layer interface |
| Low rank only after unavailable target subtraction | No such candidate accepted |
| Pivots vary strongly by network | Not measurable |
| Pivot union approaches original design | Not measurable |
| Mean error worse than Frobenius suggests | Not measurable |
| Tensor construction requires reading complete output array | The only proposed TT-SVD diagnostic does; it is oracle-only |
| Required stored node array unavailable | Yes |

The last two conditions are enough for an operational closure. They do not constitute a universal final-output rank theorem.

---

## 10. Contradiction and overstatement audit

The prior research record contains two incompatible descriptions:

1. Agent 5 proposed running the audit on “already stored complete Kerdock activation/output arrays.”
2. The v31 canonical ledger records M189 as not run, with no rank, pivot, or query result.

Direct archive inspection supports the ledger, not the stronger artifact claim. The retained bundles contain the Kerdock node generator, final aggregate estimates, and basis/block summaries, but not the complete node-by-output tensors.

The phrase **“stored full-Kerdock arrays”** should therefore be corrected to:

> Stored Kerdock design assets and aggregate/per-basis outputs; complete nodewise final-output arrays were not located in the retained shared artifacts.

---

## 11. Proposed canonical ledger patch

**ID:** M189  
**Branch:** Tensor integration  
**Experiment:** Kerdock-index QTT audit  
**Environment:** Retained v31 shared artifacts; no new network evaluations  

**Canonical result:**  
Operational FAIL. The complete `65,536×256` nodewise final-output tensors required for exact QTT ranks, singular-value curves, common-pivot simulation, and held-out mean reconstruction were not present in the retained artifacts. Available files contain the Kerdock design asset, aggregate output vectors, and basis/block summaries only. Consequently, no legal query rule or adjusted-score candidate was produced. Exact structural evidence closes shared/layerwise QTT: the first preactivation row-by-neuron matrix has rank 256 almost surely, while a scalar preactivation’s square has 32,641 of 32,768 sign-even Walsh frequencies. Direct final-output scalar QTT remains mathematically unresolved but has no evidence passing the rank, pivot-union, query, or score gates.

**Compute status:**  
Full-array TT-SVD would require approximately 174.26B chirp-node propagation FLOPs and is oracle-only. With 512 coordinate nodes retained, 10,240 chirp queries already cost approximately 28.59B before TT overhead and exceed the 27.2B gate.

**Status:**  
Closed operationally; direct-final-output rank theorem absent.

**Closure confidence:**  
High for shared/layerwise QTT; moderate for direct final-output QTT as a competition-path closure.

**Next action:**  
No further competition effort. Reopen only if an immutable nodewise archive with exact index mapping, network identities, layers, and hashes is restored; any restored audit must begin with a frozen common-pivot protocol and may not generate a full-array TT-SVD candidate first.

**Protected data:**  
Remain sealed.

---

## Bottom line

The experiment did not uncover a low-rank QTT candidate. More importantly, the numerical experiment described by M189 was never preserved in runnable form and cannot be reconstructed under the prohibition on new network evaluations.

The mathematically strongest surviving statement is:

\[
\boxed{
\text{Direct final-output QTT is not universally disproved, but M189 fails as an evidence-backed winning path.}
}
\]

The branch should be removed from the active competition priority list rather than retained as though a cheap existing-array test were still immediately available.
