# Prompt 2 — Final M192 Tied/Shared-Covariance Exception

**Date:** 2026-07-31  
**Canonical context:** WHestBench v31 final local write-up  
**Protected evaluation opened:** No

## Executive verdict

# **REPRESENTATION ONLY**

A common-covariance location-latent representation has reported oracle closure below the \(3\times10^{-3}\) gate, but the passing construction uses an oracle/particle latent state rather than a compact, legally initialized \(K\le64\) mixture. No legal 32-layer rollout or complete benchmark cost exists.

The strict tied-covariance recurrence is mathematically valid and substantially cheaper than component-specific covariance propagation. However:

1. sharing the pre-ReLU covariance does **not** make post-ReLU covariances component-independent;
2. at \(K=64\), the recurrence still requires about \(64.76\) million distinct noncentral bivariate ReLU pair evaluations over 31 layers;
3. no exact finite collection of component-independent correlation transforms can eliminate this component dependence;
4. fixed-rank shared covariance modulation is not closed under ReLU—the modulation generically becomes full-rank after one layer;
5. the available library does not contain the tied-\(K\) oracle outputs, fitting arrays, or submission-compatible bivariate-CDF benchmark required to establish representation and cost simultaneously.

The broader v31 record likewise describes tied/shared covariance as the one remaining unisolated M192 exception, not as an already measured candidate.

**Operational recommendation:** close M192 as a competition-development branch. Preserve the tied-mixture recurrence as a valid representation theorem, but do not promote it to M188 or run a 32-layer rollout without a separately reproduced compact \(K\le64\) oracle pass and a measured FlopScope implementation.

---

## 1. Exact tied-covariance recurrence

Use the column-vector convention

\[
Z_\ell\sim q_\ell
=
\sum_{k=1}^{K}\pi_k
\mathcal N(\mu_{\ell k},R_\ell),
\qquad
\sum_k\pi_k=1,
\]

and let

\[
A_\ell=\operatorname{ReLU}(Z_\ell),
\qquad
Z_{\ell+1}=W_{\ell+1}A_\ell.
\]

For the shared covariance \(R_\ell\), define

\[
\sigma_i=\sqrt{(R_\ell)_{ii}},
\qquad
a_{ki}=\frac{\mu_{\ell k,i}}{\sigma_i},
\qquad
\rho_{ij}=
\frac{(R_\ell)_{ij}}{\sigma_i\sigma_j}.
\]

Thus the marginal scales and correlation matrix are shared, but the standardized thresholds \(a_{ki}\) are component-specific.

### 1.1 Componentwise ReLU means

For every component and coordinate,

\[
m_{ki}
=
\mathbb E[A_{\ell i}\mid k]
=
\mu_{ki}\Phi(a_{ki})
+
\sigma_i\phi(a_{ki}).
\]

### 1.2 Componentwise diagonal second moments

\[
S_{k,ii}
=
\mathbb E[A_{\ell i}^{2}\mid k]
=
(\mu_{ki}^{2}+\sigma_i^{2})\Phi(a_{ki})
+
\mu_{ki}\sigma_i\phi(a_{ki}).
\]

### 1.3 Componentwise off-diagonal second moments

For \(i\neq j\), put

\[
a=a_{ki},\qquad b=a_{kj},\qquad
\rho=\rho_{ij},\qquad
s=\sqrt{1-\rho^{2}},
\]

\[
P=\Phi_2(a,b;\rho),
\]

where \(\Phi_2\) is the standard bivariate-normal CDF, and

\[
D=\phi_2(a,b;\rho).
\]

Then

\[
\begin{aligned}
S_{k,ij}
={}&
\mu_{ki}\mu_{kj}P\\
&+\mu_{ki}\sigma_j\phi(b)
  \Phi\!\left(\frac{a-\rho b}{s}\right)\\
&+\mu_{kj}\sigma_i\phi(a)
  \Phi\!\left(\frac{b-\rho a}{s}\right)\\
&+\sigma_i\sigma_j
 \left[
 \rho P+(1-\rho^2)D
 \right].
\end{aligned}
\]

The component covariance is

\[
C_k=S_k-m_km_k^\top.
\]

### 1.4 Within- and between-component covariance

Define

\[
\bar m=\sum_k\pi_km_k,
\]

\[
\bar C_{\mathrm{within}}
=
\sum_k\pi_k C_k,
\]

and

\[
C_{\mathrm{between}}
=
\sum_k\pi_k
(m_k-\bar m)(m_k-\bar m)^\top.
\]

Therefore the global post-ReLU covariance under \(q_\ell\) is exactly

\[
\operatorname{Cov}_{q_\ell}(A_\ell)
=
\bar C_{\mathrm{within}}
+
C_{\mathrm{between}}.
\]

### 1.5 Next component means and shared covariance

Set

\[
\mu_{\ell+1,k}=W_{\ell+1}m_k
\]

and

\[
R_{\ell+1}
=
W_{\ell+1}
\bar C_{\mathrm{within}}
W_{\ell+1}^{\top}.
\]

The next tied mixture is

\[
q_{\ell+1}
=
\sum_k\pi_k
\mathcal N(\mu_{\ell+1,k},R_{\ell+1}).
\]

---

## 2. Claimed properties: proof audit

| Claim | Result | Qualification |
|---|---|---|
| Global mean preserved | **True** | Exactly under the current approximate mixture law |
| Global covariance preserved | **True** | Exactly under the current approximate mixture law |
| PSD and realizability | **True** | In exact arithmetic |
| \(K=1\) equals Gaussian closure | **True** | No non-Gaussian state can arise spontaneously |
| One full covariance transform | **True** | But only after \(K\) dense pair-moment matrices have been evaluated and pooled |
| Mean propagation \(O(Kn^2)\) | **True** | Dense component mean transforms cost \(2Kn^2\) per layer |
| No hidden per-component \(n^3\) calculation | **True** | The remaining hidden cost is \(O(Kn^2)\) nonlinear pair evaluation, not \(O(Kn^3)\) |

### 2.1 Mean preservation

\[
\mathbb E_{q_{\ell+1}}[Z_{\ell+1}]
=
\sum_k\pi_kWm_k
=
W\bar m
=
\mathbb E_{q_\ell}[W\operatorname{ReLU}(Z_\ell)].
\]

### 2.2 Covariance preservation

The covariance of the projected tied mixture is

\[
\begin{aligned}
\operatorname{Cov}_{q_{\ell+1}}(Z_{\ell+1})
&=
R_{\ell+1}
+
\operatorname{Cov}_k(Wm_k)\\
&=
W\bar C_{\mathrm{within}}W^\top
+
WC_{\mathrm{between}}W^\top\\
&=
W\operatorname{Cov}_{q_\ell}(A_\ell)W^\top.
\end{aligned}
\]

Therefore it preserves the complete mean and covariance produced by applying ReLU and the linear map to the **approximate input law**. It does not claim equality to the unknown true network distribution.

### 2.3 PSD and realizability

Each \(C_k\) is the covariance of an actual rectified Gaussian vector, hence

\[
C_k\succeq0.
\]

Consequently,

\[
\bar C_{\mathrm{within}}\succeq0
\quad\Longrightarrow\quad
R_{\ell+1}
=
W\bar C_{\mathrm{within}}W^\top
\succeq0.
\]

Positive weights and Gaussian components then define a genuine probability law. Numerical CDF error may introduce small negative eigenvalues in an implementation, so symmetrization and PSD diagnostics remain necessary, but no PSD projection is required by the exact mathematics.

### 2.4 \(K=1\)

For \(K=1\),

\[
C_{\mathrm{between}}=0,
\]

and the recurrence becomes ordinary moment-matched Gaussian propagation. Starting from the exactly Gaussian first preactivation, duplicate components remain duplicates unless a nontrivial legal split is introduced.

### 2.5 Legal initialization

A nontrivial tied mixture can be initialized without activation samples. Choose a decomposition

\[
\Sigma=BB^\top+R,\qquad R\succeq0,
\]

replace the latent Gaussian in \(B\xi\) by a positive quadrature rule whose nodes reproduce its first two moments, and use

\[
\mu_k=\mu+B\xi_k.
\]

Alternatively, after obtaining candidate offsets \(d_k\), scale them by the maximum \(\alpha\) satisfying

\[
R=V-\alpha^2
\sum_k\pi_kd_kd_k^\top
\succeq0.
\]

This gives exact mean/covariance matching and a legal positive mixture.

What is not established is that a direction rule derived only from weights and the prior state selects the **oracle-useful** latent orientation.

---

## 3. What is actually reusable

The shared \(R_\ell\) permits reuse of:

\[
\sigma_i,\qquad
\rho_{ij},\qquad
\sqrt{1-\rho_{ij}^{2}},
\]

together with correlation-only constants and quadrature nodes.

It does **not** permit reuse of:

\[
\Phi_2(a_{ki},a_{kj};\rho_{ij}),
\]

\[
\Phi\!\left(
\frac{a_{ki}-\rho_{ij}a_{kj}}
{\sqrt{1-\rho_{ij}^{2}}}
\right),
\]

or

\[
\Phi\!\left(
\frac{a_{kj}-\rho_{ij}a_{ki}}
{\sqrt{1-\rho_{ij}^{2}}}
\right),
\]

because these depend on the component-specific pair of thresholds.

Thus

\[
C_k\neq C_{k'}
\]

generically even though both components have the same pre-ReLU covariance.

### 3.1 Infinite-rank threshold obstruction

Let \(X,Y\) be standard Gaussians with correlation \(\rho\), and define the centered shifted-ReLU function

\[
f_a(x)=(x+a)_+-\mathbb E[(X+a)_+].
\]

Mehler’s identity gives

\[
\operatorname{Cov}(f_a(X),f_b(Y))
=
\sum_{r=1}^{\infty}
\frac{\rho^r}{r!}
h_r(a)h_r(b),
\]

where

\[
h_1(a)=\Phi(a)
\]

and, for \(r\ge2\),

\[
h_r(a)
=
(-1)^{r-2}
H_{r-2}(a)\phi(a).
\]

The functions

\[
\Phi(a),\quad
\phi(a),\quad
a\phi(a),\quad
H_2(a)\phi(a),\ldots
\]

are linearly independent. Therefore, for generic \(0<|\rho|<1\), the kernel as a function of \((a,b)\) has infinite separation rank.

Consequently, there is no identity of the form

\[
C_k
=
\sum_{r=1}^{s}
\alpha_r(\mu_k)M_r(R)
\]

with a fixed finite \(s\) that is exact for arbitrary component means.

This does **not** rule out a controlled approximation on a restricted threshold range. It does rule out the stronger hope that one or a few exact correlation-only matrix functions can supply all component pair moments.

Batched kernels improve wall time and memory locality. They do not reduce the benchmark FLOP count.

---

## 4. Minimum state required by the recurrence

At a single layer, selected next variances require only

\[
\operatorname{diag}
\left(
W\bar C_{\mathrm{within}}W^\top
\right)
\]

plus the between-component contribution.

For recursive propagation, however, the next layer requires the complete correlation matrix. Therefore the state must retain

\[
R_{\ell+1}\in\mathbb S_+^n,
\]

not just its diagonal.

For a nonsingular square \(W\), the map

\[
C\mapsto WCW^\top
\]

is injective:

\[
C=W^{-1}(WCW^\top)W^{-\top}.
\]

Hence the exact next shared covariance contains all

\[
\frac{n(n+1)}2
\]

degrees of freedom of the pooled within covariance. Diagonal-only output is not a recursively closed state.

The tied construction saves \(K-1\) cubic congruence transforms. It does not remove the dense pairwise covariance object.

---

## 5. Exact operation and call counts

Let

\[
P=\frac{n(n-1)}2,
\qquad
Q=\frac{n(n+1)}2,
\]

with \(n=256\) and \(L=31\).

### 5.1 Dense linear algebra

The one shared covariance transform costs

\[
2Ln^3
=
1{,}040{,}187{,}392
\]

dense multiply-add FLOPs.

Component-mean transforms cost

\[
2LKn^2.
\]

| \(K\) | Mean-transform FLOPs |
|---:|---:|
| 1 | 4,063,232 |
| 2 | 8,126,464 |
| 4 | 16,252,928 |
| 8 | 32,505,856 |
| 16 | 65,011,712 |
| 32 | 130,023,424 |
| 64 | 260,046,848 |

### 5.2 Pair-moment call counts

The total number of component/pair entries is

\[
LKQ.
\]

The number of off-diagonal bivariate-CDF calls is

\[
LKP.
\]

| \(K\) | All pair entries | Off-diagonal \(\Phi_2\) calls |
|---:|---:|---:|
| 1 | 1,019,776 | 1,011,840 |
| 2 | 2,039,552 | 2,023,680 |
| 4 | 4,079,104 | 4,047,360 |
| 8 | 8,158,208 | 8,094,720 |
| 16 | 16,316,416 | 16,189,440 |
| 32 | 32,632,832 | 32,378,880 |
| 64 | **65,265,664** | **64,757,760** |

At \(K=64\), the off-diagonal formula also calls approximately

\[
129{,}515{,}520
\]

component/pair-specific conditional univariate CDFs, unless a particular bivariate routine returns those derivatives jointly.

### 5.3 Why a final benchmark FLOP number cannot be supplied

The exact charged count depends on the actual submission-compatible implementations of:

- \(\Phi\);
- \(\Phi_2\);
- exponentials and square roots;
- clipping and correlation-edge fallbacks;
- PSD checks;
- array construction;
- residual wall time.

The available research routine uses Gauss–Legendre integration with a default of 16 nodes per bivariate CDF.

At \(K=64\), that particular rule would perform

\[
16\times64{,}757{,}760
=
1{,}036{,}124{,}160
\]

bivariate-density node evaluations over the rollout, before the conditional CDFs, covariance assembly, matrix products, edge handling, and residual-time charge.

The benchmark currency is

\[
C_{\mathrm{effective}}
=
C_{\mathrm{tracked}}
+
10^{11}t_{\mathrm{residual}},
\]

so each 10 milliseconds of residual runtime costs another \(10^9\) effective FLOPs.

Every CDF, factor construction, PSD check, fallback and wall-time residual must be counted.

The v31 artifacts state that the raw scripts and result JSON needed for independent regeneration were referenced but not attached. Consequently, any single numerical “exact FLOP” total beyond the algebra and call counts above would be invented.

---

## 6. Oracle representation evidence

### 6.1 Requested finite-\(K\) tied ladder

No independently inspectable results were available for

\[
K=1,2,4,8,16,32,64
\]

under the strict tied-covariance family.

Agent 4’s own final assessment lists the tied \(K=32/64\) oracle curve, legal rollout and actual bivariate-CDF cost as unresolved.

Therefore the following requested quantities cannot honestly be reported:

- layerwise mean, median and worst tied-\(K\) errors;
- tied-\(K\) results at layers 16 and 29;
- initialization sensitivity;
- PSD violation frequency;
- final analytic-mean error.

### 6.2 Nearby positive oracle result

A richer **location-latent, common-covariance** representation reportedly obtained:

\[
e_{\sigma,\ell=29}(r=32)
\approx2.37\times10^{-3},
\]

\[
e_{\sigma,\ell=29}(r=64)
\approx8.6\times10^{-4},
\]

and

\[
\operatorname{mean}_\ell e_\sigma(r=64)
\approx1.96\times10^{-3}.
\]

This passes the representation threshold. But it was purchased using forward particles and does not supply a compact legal analytic recurrence.

The underlying local notes describe this as a location-only latent model with fixed conditional covariance, but requiring roughly 64–128 latent dimensions.

Thus the evidence supports:

> A common-covariance latent representation can contain the missing joint information.

It does **not** support:

> A finite tied mixture with \(K\le64\), legally initialized from weights, reaches the same closure error.

### 6.3 Full-covariance comparison

The full component-covariance \(K=32\) oracle reached approximately

\[
5.33\times10^{-3}
\]

at layer 16 and

\[
3.58\times10^{-3}
\]

at layer 29, with mean error around

\[
4.8\times10^{-3}.
\]

It therefore missed the \(3\times10^{-3}\) gate despite using a more expressive covariance family.

This does not prove tied covariance fails, because the fitting procedures and latent parameterizations differ. It does make an unmeasured tied-\(K\) pass unsafe to assume.

---

## 7. Shared low-rank covariance modulation

Consider

\[
R_k=R+UD_kU^\top,
\qquad
\operatorname{rank}(U)=s.
\]

This provides a low-rank pre-ReLU covariance difference, but it is not closed under ReLU.

### 7.1 Rank-growth proof

For a Gaussian vector, Price’s theorem gives, schematically,

\[
\frac{\partial}{\partial \Sigma_{ij}}
\mathbb E[\rho(Z_i)\rho(Z_j)]
=
\Pr(Z_i>0,Z_j>0)
\]

for off-diagonal perturbations.

Therefore, to first order,

\[
\Delta C_k
=
H_k\circ(UD_kU^\top)
+
\text{diagonal/threshold terms},
\]

where \(H_k\) is a dense matrix of gate-coactivation probabilities and \(\circ\) denotes the Hadamard product.

Even for \(s=1\),

\[
UD_kU^\top=d_k uu^\top,
\]

so

\[
H_k\circ(uu^\top)
=
\operatorname{diag}(u)\,
H_k\,
\operatorname{diag}(u).
\]

For generic nonzero \(u\) and generic dense \(H_k\),

\[
\operatorname{rank}
\left(
H_k\circ(uu^\top)
\right)
=
\operatorname{rank}(H_k)
=
n.
\]

Thus a rank-one covariance perturbation generically produces a full-rank post-ReLU covariance perturbation.

After the linear map,

\[
W\Delta C_kW^\top
\]

remains generically full-rank. A fixed-rank family

\[
R_{\ell+1,k}
=
R_{\ell+1}
+
U_{\ell+1}D_{\ell+1,k}U_{\ell+1}^\top
\]

therefore requires a projection after every layer. Such a projection is an approximation, not an exact recurrence, and there is no theorem preventing the required rank from growing to \(n\).

### 7.2 Consequence

Shared \(U\) reduces the number of pre-ReLU covariance parameters. It does not produce an exact reusable post-ReLU pair transform, and fixed ranks

\[
s=1,2,4,8
\]

are not invariant under the map.

Because strict tying has no compact oracle result and the low-rank extension is not exactly closed, Part IV does not supply a promotion rung.

---

## 8. Legal 32-layer rollout

The rollout gate was not reached.

A valid rollout would require all of the following:

- a compact \(K\le64\) state passing the oracle closure gate;
- a frozen weight-derived initialization;
- no empirical clustering or per-layer recalibration;
- measured \(\Phi_2\) implementation cost;
- full PSD diagnostics;
- raw MSE no greater than \(2.962\times10^{-7}\);
- complete adjusted-score and tail accounting.

The existing passing common-covariance result is an oracle particle representation, while the strict finite-\(K\) tied state remains unmeasured. M188 should therefore remain unrun.

---

## 9. Final decision table

| Requirement | Status |
|---|---|
| Exact tied recurrence | **Passed mathematically** |
| Exact global moment preservation under approximate law | **Passed** |
| PSD/realizability | **Passed** |
| One dense covariance congruence per layer | **Passed** |
| Component-independent post-ReLU covariance | **Failed** |
| Exact finite shared pair-kernel basis | **Theoretically impossible** |
| Fixed-rank covariance modulation closure | **Failed theoretically** |
| Common-covariance oracle representation | **Passed only in richer particle/continuous-latent form** |
| Strict tied \(K\le64\) oracle gate | **Not reproduced / artifacts absent** |
| Legal initialization yielding oracle-useful orientation | **Not established** |
| Exact benchmark cost | **Not established** |
| Legal 32-layer rollout | **Not run** |
| Deployable candidate | **No** |

# **Required conclusion: REPRESENTATION ONLY**

The missing joint law can be represented by a common-covariance latent state, but the available passing state is oracle- and particle-dependent. The compact tied mixture has a valid moment recurrence, yet no reproduced \(K\le64\) closure result, no demonstrated legal orientation, and no complete benchmark cost. Fixed shared-low-rank covariance modulation does not remain low-rank after ReLU.

## Proposed canonical ledger patch

**ID:** M192  
**Evidence level:** Exact recurrence theorem plus oracle representation ceiling; compact tied experiment unavailable  
**Family:** Conditional Gaussian mixture / common-covariance latent state  
**Experiment:** Final tied/shared-covariance exception  
**Result:** A positive tied-covariance mixture preserves the complete global mean and covariance of its approximate law, remains PSD, and uses one dense covariance congruence per layer. It nevertheless requires \(LK\,n(n-1)/2\) component-specific noncentral bivariate pair evaluations; at \(K=64,n=256,L=31\), this is 64,757,760 \(\Phi_2\) calls. No finite exact correlation-only matrix basis removes the threshold dependence. Fixed-rank shared covariance modulation is not closed under ReLU and generically becomes full-rank. A richer common-covariance location-latent particle state reports mean sigma error approximately \(1.96\times10^{-3}\), but no compact legal \(K\le64\) realization or rollout is available.  
**Verdict:** **REPRESENTATION ONLY; close operational M192 development.**  
**Status:** Theorem retained / competition branch closed  
**Scope limit:** This does not prove that every conceivable controlled approximation to the tied pair kernel is impossible. It establishes that no passing, legally generated and completely costed compact state currently exists, and that fixed exact reuse or fixed-rank modulation cannot provide one.

---

## Source files consulted

- `Pasted markdown(13).md`
- `WHestBench_Current_State_v31_20260731.md`
- `whestbench_canonical_research_ledger_20260731_reconciled_v31_final_local_writeup.xlsx`
- `Agent 4 - Deterministic Gaussian Mixture and Transport Propagation.md`
- `agent_4_deterministic_gaussian_mixture_transport_propagation.md`
- `Pasted text(67).txt`
- `Pasted text(68).txt`
- `WHestBench_Current_State_v30_20260731.md`
- `WHestBench_Agent1_Latent_Variable_Copula_Closure_20260730.md`
- `experiment_suite.py`
