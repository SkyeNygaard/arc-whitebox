# Prompt 1 result: weighted signed-floor optimization

**Date:** 2026-07-30  
**Problem:** Strengthen T47 by replacing open-ended harmonic-weight search with a principled primal/dual optimization and hostile audit.  
**Bottom line:** The existing degree-47 lower certificate already clears the requested 1.30× threshold. A new directed dual proves that, inside a substantially broadened degree-47 comparison class, no weighted-rank argument can reach the requested 1.10× threshold.

## 1. Main result

Let \(R_K\) denote complete-Kerdock limiting-kernel MSE for the dimension-256, depth-32 normalized ReLU ensemble, and let the node budget be

\[
N=66{,}048.
\]

The frozen D47 certificate gives

\[
R(Q)\ge P_{47}
=1.9250552503122307789\times10^{-7}
=0.79101228910008758459\,R_K
\]

for every static, network-independent, mass-one signed linear rule with at most \(N\) arbitrary spherical nodes. Its same-cost raw-improvement cap is

\[
R_K/P_{47}=1.2642028623065665.
\]

The new result concerns the *power of the proof method*. Define \(\beta_{47}^{\mathrm{CP}}\) as the best lower floor obtainable from any finite or countable sum of squared, rotation-invariant harmonic comparison kernels supported on degrees \(0,\ldots,47\), with nonnegative harmonic weights. Equivalently, this allows arbitrary completely-positive mixtures of degree-47 weight vectors, not only one square \(L_a^2\).

### Theorem A — Degree-47 primal/dual window

\[
\boxed{
0.79101228910008758459\,R_K
\le \beta_{47}^{\mathrm{CP}}
\le 0.87671080553396886557\,R_K.
}
\]

Consequences:

1. The Prompt-1 target \(0.769231R_K\), sufficient to rule out a 1.30× same-cost signed gain, is already achieved.
2. The target \(0.909091R_K\), sufficient to rule out a 1.10× same-cost signed gain, is **impossible within this entire degree-47 CP comparison class**.
3. D47 attains at least
   \[
   \frac{0.7910122891000876}{0.8767108055339689}
   =0.9022499598579878
   \]
   of the best floor available in the declared class. Its remaining multiplicative improvement room is at most 1.108340309771139.
4. This is an upper bound on what this proof class can certify. It is not an upper bound on the true risk of arbitrary cubature rules, and it does not close higher cutoffs or infinite comparison kernels.

The directed verifier checks 1,035 nonzero upper-triangular dual inequalities. The smallest directed margin is

\[
6.3126914675002077\times10^{-10},
\]

at harmonic pair \((3,3)\).

---

## 2. Exact rank obstruction, including every hostile case

The weighted-rank argument needs an exact answer to the following relaxation.

Let \(A\succeq0\) be a finite-dimensional positive semidefinite matrix with eigenvalues

\[
\lambda_1\ge\lambda_2\ge\cdots\ge0,
\qquad T=\operatorname{tr}A.
\]

For \(1\le r\), define

\[
E_r(A)=\inf\left\{
\|A-M\|_F^2:
M=M^\top,\ \operatorname{rank}M\le r,\ \operatorname{tr}M=T
\right\}.
\]

### Theorem B — Exact all-rank, trace-preserving approximation

\[
\boxed{
E_r(A)=
\sum_{j>r}\lambda_j^2+
\frac1r\left(\sum_{j>r}\lambda_j\right)^2.
}
\]

An optimizer is

\[
M_r
=\sum_{j=1}^r
\left(\lambda_j+\frac{T_r}{r}\right)u_ju_j^\top,
\qquad
T_r=\sum_{j>r}\lambda_j.
\]

It is positive semidefinite. Thus allowing indefinite matrices does not improve the relaxation.

### Proof

Let \(S=\operatorname{range}M\), \(s=\dim S\le r\), and let \(P\) be the orthogonal projector onto \(S\). The Frobenius decomposition

\[
A-M=(A-PAP)+(PAP-M)
\]

is orthogonal because \(P(A-PAP)P=0\). Therefore

\[
\|A-M\|_F^2
=\|A-PAP\|_F^2+\|PAP-M\|_F^2.
\]

For fixed \(S\), write

\[
\delta=T-\operatorname{tr}(PAP)=\operatorname{tr}((I-P)A)\ge0.
\]

Among symmetric matrices supported on \(S\) with trace \(T\), the second term is minimized by the orthogonal trace correction

\[
M=PAP+\frac{\delta}{s}P,
\]

with minimum \(\delta^2/s\). This optimizer is positive semidefinite. The total fixed-subspace error is

\[
\|A\|_F^2-\|PAP\|_F^2+
\frac{(T-\operatorname{tr}PAP)^2}{s}.
\]

This decreases when either \(\|PAP\|_F^2\) or \(\operatorname{tr}(PAP)\) increases. Ky Fan inequalities maximize both on a top-\(s\) eigenspace. Substitution gives the displayed formula.

Finally, writing \(T_{s+1}=\sum_{j>s+1}\lambda_j\),

\[
E_s(A)-E_{s+1}(A)
=
\frac{\big((s+1)\lambda_{s+1}+T_{s+1}\big)^2}{s(s+1)}
\ge0.
\]

Hence the weakest lower bound among all ranks \(s\le N\) occurs at \(s=N\). Accidental rank loss from signed cancellation only strengthens the obstruction. Eigenvalue multiplicities create nonuniqueness of the top eigenspace but do not alter the value.

### Why the relaxation is safe

For a signed mass-one rule

\[
Q=\sum_{i=1}^m w_i\delta_{x_i},
\qquad \sum_iw_i=1,
\]

and a harmonic feature map \(v_a\), its moment matrix

\[
M_Q=\sum_iw_iv_a(x_i)v_a(x_i)^\top
\]

is symmetric and has rank at most \(m\le N\), even when the weights are signed. Its trace equals the population trace because \(\|v_a(x)\|^2\) is constant on the sphere. The matrix optimizer above need not be realizable by nodes. That only enlarges the feasible matrix class, so its optimum remains a valid lower bound for every realizable rule.

---

## 3. The exact finite-cutoff primal

Let \(G_\ell\) be normalized Gegenbauer polynomials with \(G_\ell(1)=1\), and let \(d_\ell\) be the dimension of the degree-\(\ell\) harmonic space. For a nonnegative weight vector \(a=(a_0,\ldots,a_L)\), define

\[
L_a(t)=\sum_{\ell=0}^L a_\ell d_\ell G_\ell(t),
\qquad
A(a)=\bigoplus_{\ell=0}^L a_\ell I_{d_\ell}.
\]

Write

\[
L_a(t)^2=\sum_{r=0}^{2L} b_r(a)G_r(t),
\qquad
b_r(a)=a^\top C_r a,
\]

where every linearization matrix \(C_r\) is entrywise nonnegative. Let the target ReLU kernel have coefficients \(k_r\ge0\).

For mass-one discrepancy, degree zero cancels. Thus

\[
\gamma(a)=\min_{1\le r\le2L:b_r(a)>0}\frac{k_r}{b_r(a)}
\]

and

\[
R_K(Q)\ge\gamma(a)F_N(A(a)).
\]

The objective is homogeneous. With \(c=\sqrt{\gamma(a)}a\), the exact normalized primal is

\[
\boxed{
\beta_L=
\sup_{c\ge0}
F_N(A(c))
\quad\text{subject to}\quad
c^\top C_rc\le k_r,
\quad r=1,\ldots,2L.
}
\]

This formulation includes every odd and even active degree. There is no comparison tail beyond \(2L\), because \(L_c^2\) is exactly degree \(2L\). The target-kernel coefficient tail matters when one constructs *upper* bounds on \(k_r\); Section 7 handles it rigorously.

### Piecewise-quadratic rank chambers

The eigenvalues of \(A(c)\) are the scalars \(c_\ell\), each repeated \(d_\ell\) times. For any declared selection counts \(q_\ell\in\{0,\ldots,d_\ell\}\) with \(\sum q_\ell=N\), define tail counts \(t_\ell=d_\ell-q_\ell\) and

\[
H_q=\operatorname{diag}(t)+\frac1Ntt^\top.
\]

The error obtained by retaining that declared set of \(N\) eigenvectors is

\[
c^\top H_qc.
\]

The exact best-rank error is the minimum over valid selections:

\[
F_N(A(c))=\min_q c^\top H_qc.
\]

This minimum is nonsmooth at eigenvalue ties. Any optimization or dual that silently freezes one chamber can be false.

---

## 4. Hostile audit of the D47 numerical optimum

The D47 weight vector begins

\[
a_0=a_1=a_2=a_3=1.
\]

This is an exact tie across harmonic blocks whose combined multiplicity far exceeds \(N\). The tie is structurally important.

### Attempted disproof: frozen-chamber improvement

A nonlinear search using the chamber that retains all degrees 0–2 and 32,896 directions from degree 3 found an apparent 0.4865% objective improvement. The candidate drove degrees 1 and 2 to zero and raised degree 3 slightly.

Recomputing the exact sorted-eigenvalue obstruction changed the selected eigenspace to degree 3 alone. Its actual floor was slightly *below* D47. The apparent improvement was therefore a chamber error, not a better certificate.

### Balanced-slice local rigidity

Imposing the natural tie constraint

\[
a_0=a_1=a_2=a_3
\]

leaves 45 scalar variables. The nearly equioscillating D47 pattern has 45 near-active constraints: degrees 6–49 and degree 92. Twenty independent multistart constrained searches returned to D47 to numerical precision.

The active-gradient system is nonsingular in double precision, with condition number about \(3.35\times10^2\). Solving the KKT equations gives positive multipliers:

\[
0.00099576\lesssim y_r\lesssim0.08266363,
\qquad
\sum_r y_r\approx1.
\]

This is strong local evidence.

### Attempted proof: simple PSD Lagrangian dual

The natural quadratic dual slack is

\[
S=\sum_r y_rB_r-G.
\]

It is indefinite: in the balanced slice its minimum eigenvalue is about \(-0.00604\), with 21 materially negative eigenvalues in the normalized diagnostic. Thus the attractive claim “equioscillation plus positive multipliers proves global optimality” is false.

This failure is useful. It identifies the correct cone: the primal harmonic mixtures are nonnegative/completely positive, so entrywise nonnegative domination can certify a global bound even when PSD domination fails.

---

## 5. Broader comparison class: completely-positive mixtures

A single square is not the only natural comparison. Let

\[
J(t)=\sum_s L_{a^{(s)}}(t)^2,
\qquad a^{(s)}\ge0,
\]

where the sum may be finite or countable with convergent coefficients. Define

\[
X=\sum_s a^{(s)}a^{(s)\top}.
\]

Then \(X\) is completely positive and entrywise nonnegative, and

\[
[J]_r=\langle C_r,X\rangle.
\]

If \([J]_r\le k_r\) for every nonconstant degree, then

\[
R_K(Q)\ge R_J(Q)
=\sum_s R_{L_{a^{(s)}}^2}(Q)
\ge\sum_sF_N(A(a^{(s)})).
\]

For any fixed rank selection \(q\),

\[
\sum_sF_N(A(a^{(s)}))
\le\sum_s a^{(s)\top}H_qa^{(s)}
=\langle H_q,X\rangle.
\]

Thus a dual valid on entrywise nonnegative matrices automatically covers every such mixture, not merely one rank-one weight vector.

---

## 6. The successful entrywise dual

Use the concrete fixed selection

- all 1 degree-0 direction;
- all 256 degree-1 directions;
- all 32,895 degree-2 directions;
- 32,896 degree-3 directions;
- no higher-degree directions.

This totals \(N=66{,}048\). Let \(H_*\) be the corresponding fixed-selection matrix. Since the best rank-\(N\) approximation can only improve on a declared selection,

\[
F_N(A(c))\le c^\top H_*c
\]

for every nonnegative \(c\), regardless of its eigenvalue ordering.

Let

\[
P_{47}=1.9250552503122307789\times10^{-7}
\]

be the D47 floor. Directed coefficient upper bounds \(\bar k_r\ge k_r\) are constructed in Section 7.

The frozen dual supplies nonnegative numbers \(y_1,\ldots,y_{94}\) satisfying, entry by entry,

\[
\boxed{
H_*
\le
P_{47}\sum_{r=1}^{94}
\frac{y_r}{\bar k_r}C_r.
}
\]

Because \(X\) is entrywise nonnegative,

\[
\langle H_*,X\rangle
\le P_{47}\sum_r\frac{y_r}{\bar k_r}\langle C_r,X\rangle
\le P_{47}\sum_ry_r.
\]

The directed objective is

\[
U=\sum_ry_r
=1.1083403097711390436616238397.
\]

Therefore

\[
\beta_{47}^{\mathrm{CP}}
\le UP_{47}
=0.87671080553396886557R_K.
\]

This proves Theorem A.

### Why an entrywise dual is legitimate

The slack need not be positive semidefinite. It is paired only with completely-positive matrices

\[
X=\sum_s a^{(s)}a^{(s)\top},
\qquad a^{(s)}\ge0.
\]

Every entry of \(X\) is nonnegative. Therefore entrywise nonnegative slack is sufficient. This is a tractable inner cone of the copositive dual and is stronger here than the failed PSD route.

---

## 7. Coefficient-tail certificate

A first version of the dual used an order-95 Taylor jet as though it gave upper bounds on the full kernel coefficients. That was false and was quarantined.

### 7.1 Why truncation alone is one-sided

The normalized ReLU dual activation is

\[
\kappa(t)=\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi}
=\frac1\pi+\frac t2+
\sum_{n\ge1}
\frac{\binom{2n-2}{n-1}}
{\pi4^{n-1}(2n)(2n-1)}t^{2n}.
\]

Every Maclaurin coefficient is nonnegative. Composition preserves nonnegative coefficients, so the depth-32 kernel

\[
K(t)=\sum_{p\ge0}\alpha_pt^p
\]

has \(\alpha_p\ge0\). Also \(K(1)=1\), hence \(\sum_p\alpha_p=1\).

A truncated sum therefore gives a lower bound automatically, not an upper bound.

### 7.2 Directed order-511 jet

A direct-C implementation calls `libmpfr.so.6` with explicit downward/upward rounding and propagates a Taylor jet of order 511 through all 32 ReLU-dual compositions at 320-bit precision.

The resulting directed Maclaurin tail-mass bound is

\[
\sum_{p>511}\alpha_p
\le 2.0906068483914061547\times10^{-4}.
\]

That number alone is not small enough for high Gegenbauer degrees. The high-dimensional projection supplies the missing suppression.

### 7.3 Exact monomial-projection tail

Write

\[
t^p=\sum_rP_{p,r}G_r(t).
\]

The coefficients are nonnegative and sum to one. In dimension \(d\), for matching parity,

\[
\frac{P_{p+2,r}}{P_{p,r}}
=
\frac{(p+2)(p+1)}{(p+2-r)(d+p+r)}.
\]

The denominator minus the numerator is

\[
(d-1)p+(2-r)(d+r)-2,
\]

which is positive for \(d=256\), \(r\le94\), and every admissible \(p\ge512\). Thus \(P_{p,r}\) decreases throughout the omitted tail, and

\[
\sup_{p>511}P_{p,r}
\]

is exactly the first admissible coefficient at power 512 or 513. Across degrees 1–94, the largest such supremum is only

\[
5.2297905506021371\times10^{-37}.
\]

Consequently

\[
k_r
\le
\sum_{p\le511}\alpha_p^{\rm upper}P_{p,r}
+
\left(1-
\sum_{p\le511}\alpha_p^{\rm lower}
\right)
\sup_{p>511}P_{p,r}.
\]

These are the full directed upper bounds \(\bar k_r\) used by the dual. No coefficient tail is omitted.

---

## 8. Rotation invariance, mixed kernels, and what is covered

### Covered

- Arbitrary signed quadrature weights, provided they sum to one.
- Arbitrary spherical nodes, not restricted to Kerdock support.
- Every rank \(r\le N\), including rank loss caused by signed cancellation.
- Indefinite signed moment matrices.
- Every odd and even comparison degree 1–94.
- Finite or countable sums of squared zonal harmonic kernels through degree 47.
- Completely-positive mixtures, including mixed cross-degree products induced by those sums.
- Randomized static rules independent of the realized network, by conditioning.

### Why scalar rotation-invariant kernels are diagonal by degree

Each harmonic degree is an inequivalent irreducible representation of \(O(256)\). Schur's lemma forces an invariant covariance operator to be scalar on each degree and to have no cross-degree block. Thus a scalar rotation-invariant positive-definite comparison kernel is exactly parameterized by nonnegative harmonic weights \(a_\ell\).

### Not covered

- Harmonic cutoff above 47 or an infinite harmonic factorization.
- Non-zonal or non-rotation-invariant feature constructions.
- Comparison kernels that are not sums of squares of positive-definite zonal kernels.
- Signed harmonic feature weights for which the reproducing-kernel/rank representation fails.
- Finite-width-specific kernels.
- Network-dependent support or weights.
- Nonlinear aggregation of network evaluations.
- Transformed residual estimators, unless their residual kernel receives a new certificate.
- Free-total-mass rules.
- Lower-cost estimators whose adjusted score benefits from fewer evaluations.
- Unrestricted white-box use of the full network weights.

---

## 9. Competition thresholds

For a certified floor \(R(Q)\ge fR_K\), the same-cost raw-MSE improvement is at most \(1/f\). To explain an adjusted gap of 4.34× using both MSE and reduced evaluation compute, a hypothetical rule must use at most

\[
\frac{1}{4.34f}
\]

of baseline evaluation compute.

| Case | Floor fraction \(f\) | Same-cost raw cap \(1/f\) | Max compute fraction for 4.34× |
|---|---:|---:|---:|
| v18 T47 | 0.5051771255 | 1.9795037× | 45.61% |
| Prompt-1 target A | 0.7692307692 | 1.3000000× | 29.95% |
| **D47 theorem** | **0.7910122891** | **1.2642029×** | **29.13%** |
| Degree-47 CP ceiling* | 0.8767108055 | 1.1406270× | 26.28% |
| Prompt-1 target B | 0.9090909091 | 1.1000000× | 25.35% |
| Kerdock reference | 1.0000000000 | 1.0000000× | 23.04% |

\*The CP ceiling is an upper bound on the best *provable floor* in the declared method, not a certified lower bound on rule risk. It must not be used as though the 1.140627 cap were already proved for cubature.

---

## 10. What was proved, and what was disproved

### Proved

1. The exact rank-\(r\), trace-preserving Frobenius obstruction for every \(r\le N\), with indefinite matrices and multiplicities handled.
2. The homogeneous finite-cutoff primal.
3. A CP-mixture extension that strictly broadens the single-square weighted family.
4. A directed degree-47 entrywise dual ceiling:
   \[
   \beta_{47}^{\mathrm{CP}}
   \le0.87671080553396886557R_K.
   \]
5. D47 is within a factor 1.10834031 of optimal in that class.
6. The 1.10 target cannot be reached without leaving the declared degree-47 CP class.
7. A complete order-511 coefficient-tail upper bound.

### Disproved or quarantined

1. Freezing one eigenvalue chamber at the degree-0–3 tie can produce false improvements.
2. Positive KKT multipliers do not imply a PSD global dual.
3. An order-95 positive Taylor truncation is not a full coefficient upper bound.
4. The optimistic preliminary 0.85125567 ceiling based on that truncated upper bound is invalid.

---

## 11. Best next theorem

The new ceiling gives a clean fork.

### Route 1 — Increase the cutoff

Run the same pipeline at \(L=63\) or \(L=79\):

1. discover a lower certificate;
2. regenerate a sufficiently long directed MPFR jet;
3. build full coefficient upper bounds with the exact tail ratio;
4. solve the entrywise CP dual;
5. verify every matrix entry with exact Fraction algebra and directed endpoints.

A floor above 0.87671 would prove that degrees above 47 genuinely add proof power. Crossing 0.909091 would achieve the 1.10 target.

### Route 2 — Infinite-dimensional factorization

The degree progression suggests an infinite equioscillating profile. A publishable endpoint would identify asymptotic weights \(a_\ell\), prove coefficient domination for all degrees, control the rank tail, and produce an analytic copositive/entrywise dual majorant. The finite D47 primal and dual should be treated as discretizations of that problem.

### Route 3 — Prove a stronger upper ceiling

The entrywise cone is deliberately simple. A stronger copositive relaxation could lower the 0.87671 ceiling while remaining rigorous. However, lowering the method ceiling does not strengthen the cubature risk theorem; it only proves that this comparison architecture is exhausted sooner.

---

## 12. Reproducibility

Core files:

- `SIGNED_RANK_DEGREE47_CERTIFICATE.json` — frozen D47 lower certificate.
- `PROMPT1_DEGREE47_ENTRYWISE_DUAL_CERTIFICATE.json` — frozen nonnegative dual weights.
- `PROMPT1_DEGREE47_DUAL_VERIFICATION.json` — directed verifier output.
- `verify_prompt1_degree47_dual.py` — exact Fraction and interval verification.
- `mpfr_kernel_jet_511.c` — direct-C MPFR directed jet generator.
- `MPFR_KERNEL_JET_511.json` — frozen order-511 directed jet.
- `PROMPT1_KERNEL_FULL_UPPER511.json` — explicit full coefficient upper bounds and tail data.
- `COMPETITION_THRESHOLD_TABLE.csv` — threshold table.
- `PROPOSED_LEDGER_PATCH.md` — proposed canonical updates.
- `run_checks.sh` — regenerates the MPFR jet byte-for-byte and reruns the verifier.

The verifier result is `PASS`. Publication should still require a qualified human proof review and, ideally, an Arb/FLINT implementation independent of both the direct MPFR jet and the mpmath endpoint recombination.
