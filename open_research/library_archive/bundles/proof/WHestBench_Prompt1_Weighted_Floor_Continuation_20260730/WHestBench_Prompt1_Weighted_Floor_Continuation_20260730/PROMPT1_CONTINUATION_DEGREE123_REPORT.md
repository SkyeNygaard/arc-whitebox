# Prompt 1 continuation: a certified 1.10× signed-static barrier

**Date:** 2026-07-30  
**Population:** dimension 256, depth-32 normalized infinite-width ReLU kernel  
**Node budget:** 66,048  
**Rule class:** static, network-independent, mass-one linear cubature with arbitrary spherical nodes and arbitrary real weights

## Executive result

The higher-cutoff weighted-rank path does not stop at the degree-47 floor. A frozen degree-123 comparison kernel gives the computer-assisted theorem

\[
R_K(Q)\ge
P_{123}
=2.2132585675729752132148303220517610018704E-7.
\]

The released complete-Kerdock proof package certifies

\[
2.4336603575430029389091338017406054668573E-7
\le R_K(Q_{\rm Ker})\le
2.4336603575430052276094665026697645914811E-7.
\]

Therefore, using the required upper endpoint in the denominator,

\[
\boxed{
R_K(Q)\ge
0.9094360931315226500472941259833592569805\,R_K(Q_{\rm Ker})
}
\]

and every rule in the theorem class has same-cost improvement at most

\[
\boxed{
R_K(Q_{\rm Ker})/R_K(Q)
\le 1.099582485842004070885790488053.
}
\]

This rigorously clears the Prompt-1 target

\[
10/11=0.9090909090\ldots,
\]

so **every 1.10× same-cost gain is ruled out in the limiting-kernel static signed class**.

A separate exact-rational dual bounds the best floor obtainable by the entire degree-123 completely-positive comparison family:

\[
\boxed{
0.909436093131522650047294125983
\le
\frac{\beta_{123}^{\rm CP}}{R_K(Q_{\rm Ker})}
\le
0.944932965937273425204533873727.
}
\]

The upper endpoint uses the certified *lower* Kerdock endpoint, as required for an upper ratio. It is below

\[
20/21=0.9523809523\ldots,
\]

so no degree-123 comparison in this class can prove a 1.05× same-cost barrier. D123 already attains at least

\[
0.962434506906486449533384451744
\]

of the optimal absolute floor in the declared class; its remaining proof-method improvement factor is at most

\[
1.039031739639363893902009690004.
\]

## 1. Exact optimization problem

Let \(G_\ell\) denote normalized Gegenbauer polynomials, \(G_\ell(1)=1\), and let \(d_\ell\) be the multiplicity of degree-\(\ell\) spherical harmonics. For nonnegative harmonic eigenvalues \(c_\ell\), set

\[
L_c(t)=\sum_{\ell=0}^L c_\ell d_\ell G_\ell(t),
\qquad
A(c)=\bigoplus_{\ell=0}^L c_\ell I_{d_\ell}.
\]

Write

\[
L_c(t)^2=\sum_{r=0}^{2L} b_r(c)G_r(t),
\qquad b_r(c)=c^\top C_r c,
\]

where every Gegenbauer linearization matrix \(C_r\) is entrywise nonnegative. If the target kernel has coefficients \(k_r\), the homogeneous normalized primal is

\[
\beta_L=
\sup_{c\ge0}F_N(A(c))
\quad\text{subject to}\quad
c^\top C_r c\le k_r,
\quad r=1,\ldots,2L.
\]

The exact all-rank obstruction is

\[
F_N(A)=
\sum_{j>N}\lambda_j^2+
\frac1N\left(\sum_{j>N}\lambda_j\right)^2.
\]

It is valid even when the signed moment matrix is indefinite. For every rank \(r<N\), the corresponding error is weakly larger, so cancellation-induced rank loss cannot escape the theorem.

## 2. The degree-123 primal certificate

The frozen certificate uses degrees \(0,\ldots,123\). Its parameterization is

\[
c_0=c_1=c_2=c_3=s,
\qquad z_\ell=d_\ell c_\ell\quad(\ell\ge4).
\]

Every higher eigenvalue satisfies \(c_\ell<c_3\). Thus the rank chamber is exact: retain

- all 1 degree-0 direction;
- all 256 degree-1 directions;
- all 32,895 degree-2 directions;
- 32,896 degree-3 directions;
- no direction of degree 4 or higher.

The exact rank defect is stored as a rational number in `SIGNED_RANK_DEGREE123_CERTIFICATE.json`; numerically,

\[
F_N(A)=
2.2128159379920508465698729584853975069246E-7.
\]

All 246 nonconstant comparison coefficients are positive and are checked. The certified minimum target-to-comparison ratio is

\[
\gamma\ge
1.0002000300039984422392341681106053358599,
\]

binding at degree **106**. Multiplying \(F_N(A)\gamma\) gives \(P_{123}\).

### Exact safety scaling

The numerical candidate was uniformly contracted by \(0.9999\) before rational freezing. This creates coefficient slack but does not reduce the theorem value. Under \(c\mapsto tc\),

\[
b_r\mapsto t^2b_r,
\qquad
F_N(A)\mapsto t^2F_N(A),
\qquad
\gamma\mapsto t^{-2}\gamma.
\]

Hence \(\gamma F_N(A)\) is exactly scale invariant.

## 3. Kernel-coefficient proof

The direct-C implementation propagates an MPFR Taylor jet through all 32 ReLU-kernel compositions to order 511 with explicit directed rounding. Its Maclaurin coefficients are nonnegative.

For normalized dimension-256 Gegenbauer projections,

\[
\frac{P_{p+2,r}}{P_{p,r}}
=
\frac{(p+2)(p+1)}{(p+2-r)(256+p+r)}.
\]

The primary lower verifier uses a nonnegative path recurrence truncated at degree 246. Truncation can only remove nonnegative paths, so it gives conservative lower bounds for every target coefficient \(k_r\), \(1\le r\le246\). The omitted Taylor tail above order 511 is also nonnegative and may safely be omitted for a lower theorem.

The comparison square has exact degree 246, so it has no coefficient tail beyond the checked range.

## 4. Independent exact cross-check

A second implementation does **not** truncate the Gegenbauer state space. It evaluates every monomial projection with the closed recurrence above, recomputes all target coefficients through Taylor order 511, rebuilds the comparison square, and independently recomputes the rank defect.

It verifies:

- the same binding degree 106;
- every full projection coefficient is at least the conservative primary coefficient;
- the exact rank-defect fraction matches;
- the full minimum ratio is slightly stronger than the conservative one.

Thus the primary theorem does not depend on the truncated-path implementation being exact; only on its proven one-sidedness.

## 5. Higher-cutoff progression

The finite-cutoff continuation was monotone in discovery and then frozen at useful thresholds.

| Cutoff | Kerdock risk retained | Same-cost cap | Status |
|---:|---:|---:|---|
| 47 | 79.101228910% | 1.264203× | certified |
| 63 | 84.079061367% | 1.189357× | certified |
| 72 | 85.8464244% | 1.164871× | numerical discovery |
| 88 | 88.1264347% | 1.134734× | numerical discovery |
| 100 | 89.3565289% | 1.119111× | numerical discovery |
| 112 | 90.3087618% | 1.107312× | numerical discovery |
| 120 | 90.7889073% | 1.101455× | numerical discovery |
| **123** | **90.943609313%** | **1.099582×** | **certified** |

The active system nearly equioscillates across a long contiguous degree range. This is evidence for a limiting factorization, but the intermediate floating-point rows are not themselves theorem statements.

## 6. Corrected degree-47 dual

The first Prompt-1 report used a high-order projection recurrence truncated at degree 94 while constructing **upper** target coefficients. That shortcut was invalid: states above degree 94 can later return to a lower degree.

The corrected upper construction uses exact untruncated closed projections for every Taylor power through 511 plus a rigorous positive-tail projection bound. All 1,035 exact rational inequalities pass. The corrected degree-47 class ceiling is

\[
\frac{\beta_{47}^{\rm CP}}{R_K(Q_{\rm Ker})}
\le 0.876710895091625562478681269745.
\]

The earlier qualitative conclusion survives, but the corrected files supersede the original upper-dual files.

Why this does not harm D47 or D123 lower theorems: truncating a nonnegative path expansion is conservative for a lower coefficient bound. It is unsafe only when used as an upper coefficient bound.

## 7. Degree-123 CP-mixture dual

Let

\[
J(t)=\sum_s L_{c^{(s)}}(t)^2,
\qquad
X=\sum_s c^{(s)}c^{(s)\top}.
\]

Then \(X\) is completely positive and entrywise nonnegative. Fix the concrete selection used above and let \(H_*\) be its rank-defect quadratic matrix. Since the true best rank approximation can only improve on a declared selection,

\[
\sum_sF_N(A(c^{(s)}))\le\langle H_*,X\rangle.
\]

The frozen nonnegative dual weights satisfy entrywise. The verifier works in the equivalent \(z_\ell=d_\ell c_\ell\) basis; positive diagonal rescaling preserves entrywise domination.

\[
H_*
\le
P_{123}\sum_{r=1}^{246}
\frac{y_r}{\bar k_r}C_r,
\]

where \(\bar k_r\) are rigorous upper target coefficients. Pairing this inequality with \(X\ge0\) entrywise and using feasibility gives

\[
\sum_sF_N(A(c^{(s)}))
\le P_{123}\sum_r y_r.
\]

A clean split rerun regenerated the MPFR jet byte-for-byte, reproduced both primal checks, and verified all **7,381** positive upper-triangular dual inequalities. The smallest margin is

\[
1.51450507208424270000E-8
\]

at pair \((101, 123)\). The objective factor is

\[
\sum_r y_r
\le 1.039031739639363893902009690004.
\]

Therefore

\[
\beta_{123}^{\rm CP}
\le
2.2996458997370750611659425984600977386205E-7
\le
0.944932965937273425204533873727R_K(Q_{\rm Ker}).
\]

This is an upper bound on the *power of the comparison proof family*, not an upper bound on cubature risk.

## 8. Hostile audit: claims attempted and broken

### Broken shortcut 1 — frozen rank chamber

Optimizing a fixed eigenvalue-order chamber can produce a false improvement near the degree-0–3 tie. The theorem computes the sorted rank obstruction exactly and freezes a candidate that lies strictly in the stated chamber above degree 3.

### Broken shortcut 2 — truncated recurrence used as an upper bound

A truncated nonnegative recurrence is a lower bound, not an upper bound. The corrected dual uses untruncated projections and supersedes the earlier degree-47 upper files.

### Broken shortcut 3 — wrong Kerdock denominator endpoint

For a statement \(P/R_{\rm Ker}\ge f\), one must divide by the certified **upper** endpoint of \(R_{\rm Ker}\). For a statement \(U/R_{\rm Ker}\le u\), one must divide by the certified **lower** endpoint. Both final normalized bounds use the correct direction.

### Broken shortcut 4 — numerical optimizer as proof

The optimizer only discovers weights. The theorem interprets frozen decimal strings as exact rationals and rechecks every inequality, rank count, multiplicity, odd degree, even degree, and coefficient tail independently.

### Broken shortcut 5 — treating the dual ceiling as a risk floor

The CP ceiling does not prove \(R(Q)\ge0.9449R_K\). It proves that this degree-123 proof architecture cannot certify a floor above that number.

## 9. Competition implications

For a certified floor \(R(Q)\ge fR_K\), the same-cost raw gain is at most \(1/f\). If the recorded adjusted gap is 4.34×, a hypothetical rule combining MSE improvement with lower evaluation cost must use at most

\[
\frac1{4.34f}
\]

of baseline evaluation compute.

| Certified case | Floor fraction | Same-cost cap | Max baseline compute compatible with 4.34× |
|---|---:|---:|---:|
| v18 T47 | 0.5051771255 | 1.979504× | 45.61% |
| D47 | 0.7910122891 | 1.264203× | 29.13% |
| D63 | 0.8407906137 | 1.189357× | 27.40% |
| **D123** | **0.9094360931** | **1.099582×** | **25.34%** |
| Kerdock parity | 1.0000000000 | 1.000000× | 23.04% |

Thus static signed geometry at equal cost is now closed even at the 1.10× scale. To explain a 4.34× adjusted gap while remaining in the theorem's statistical class, compute must fall below about **25.34%** of baseline.

## 10. Mathematical endpoint and next route

The most informative continuation is no longer another blind cutoff sweep.

The long active run from degree 6 upward suggests an infinite factorization of the form

\[
L(t)^2=K(t)-q(t),
\]

where \(q\) has only low harmonic degrees, plausibly at most 5, and \(L\) has nonnegative Gegenbauer coefficients. A proof would require:

1. construction of the positive-definite square root \(L\);
2. proof that every Gegenbauer coefficient of \(L\) is nonnegative;
3. absolute convergence and exact coefficient domination in all degrees;
4. evaluation of the infinite rank tail;
5. a copositive or entrywise dual controlling the optimum.

For a finite theorem stronger than 1.05×, the degree-123 dual proves that the cutoff must increase or the comparison class must change. For publication of D123 itself, the remaining gates are an Arb/FLINT or comparable third interval engine and named human mathematical review.

## 11. Reproducibility map

- `SIGNED_RANK_DEGREE123_CERTIFICATE.json` — exact frozen primal certificate.
- `SIGNED_RANK_DEGREE123_CLOSED_PROJECTION_CROSSCHECK.json` — independent exact projection/rank cross-check.
- `DEGREE123_ENTRYWISE_DUAL_CERTIFICATE.json` — exact degree-123 CP dual.
- `DEGREE123_DUAL_CHUNK_*.json` — six exact verification chunks.
- `KERNEL_FULL_UPPER511_DEGREE246.json` — full upper target coefficients and tail bounds.
- `KERDOCK_MSE_CERTIFIED_INTERVAL.json` — certified normalization endpoints and direction rule.
- `HIGHER_CUTOFF_PROGRESSION.csv` — certified and exploratory cutoff progression.
- `COMPETITION_THRESHOLD_TABLE_CONTINUED.csv` — competition thresholds.
- `run_checks.sh` — reruns the primal, independent cross-check, and all exact dual chunks.
