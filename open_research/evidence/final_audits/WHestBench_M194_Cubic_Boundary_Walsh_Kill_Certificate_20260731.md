# Prompt 3 — M194 Exact Cubic Boundary/Walsh Phase Identity

## Verdict

**KILL CERTIFICATE for the tested kernel.**

The most natural gauge-invariant cubic construction based on **boundary-normal energy weighted by downstream path adjoints** has no nontrivial Walsh phase. Every complete Kerdock/MUB basis gives exactly the same quadratic normal energy by Parseval, so the centered weight-derived phase field is identically zero. Consequently,

\[
B(W,Q)\equiv 0
\]

for every network, every rotation, and every output transcript.

Thus:

\[
\operatorname{Cov}(B,e)=0
\]

exactly, before examining any residual targets. No coefficient may be fitted.

This kills this algebraically specified kernel, but it does **not** prove that every sign-conditioned, all-ancestor boundary-current kernel in M194 is impossible.

---

## 1. Construction of the candidate

The baseline consists of 128 antipodally paired Kerdock/MUB basis groups, plus the coordinate-axis group. Each natural group contains 512 nodes, and the direct-output source is constructed from their output means.

Let

\[
G_a=\{\pm \rho b_{a,1},\ldots,\pm \rho b_{a,n}\},
\qquad a\in\mathbb F_2^7,
\]

where \(\{b_{a,u}\}_{u=1}^n\) is an orthonormal basis and \(\rho\) is the common node radius.

Write the rows of the first-layer matrix as

\[
W_1=
\begin{bmatrix}
w_1^\top\\
\vdots\\
w_n^\top
\end{bmatrix}.
\]

Each \(w_j\) is the normal to the first-layer activation boundary

\[
w_j^\top x=0.
\]

Let \(V_r\) be the frozen target-free direct-output source, and define the downstream linear path contraction

\[
D_1=W_LW_{L-1}\cdots W_2.
\]

The source-space downstream adjoint attached to first-layer unit \(j\) is

\[
a_j=V_r^\top D_1e_j,
\]

and its source-relevant energy is

\[
\alpha_j=\|a_j\|_2^2.
\]

Now define the basis-resolved boundary-normal energy

\[
q_a(W,Q)
=
\frac{1}{|G_a|}
\sum_{x\in G_a}
\sum_{j=1}^{n}
\alpha_j\,(w_j^\top x)^2.
\]

This is the minimal algebraic contraction combining:

- forward boundary normals \(w_j\);
- downstream adjoint/path energy \(\alpha_j\);
- the actual Kerdock basis geometry;
- no targets or residual-dependent choices.

Center it over the 128 basis labels:

\[
p_a=q_a-\frac1{128}\sum_bq_b.
\]

Let

\[
\widehat p_\chi
=
2^{-7}\sum_a(-1)^{\chi\cdot a}p_a
\]

be its Walsh transform.

For centered output-block vectors

\[
s_a=y_a-\frac1{128}\sum_by_b,
\qquad
\widehat s_\chi
=
2^{-7}\sum_a(-1)^{\chi\cdot a}s_a,
\]

define the cubic vector observable

\[
\boxed{
B(W,Q)=
\sum_{\chi,\psi}
\widehat p_\chi
\widehat p_\psi
\widehat p_{\chi+\psi}
\widehat s_\chi
\left\langle
\widehat s_\psi,
\widehat s_{\chi+\psi}
\right\rangle .
}
\]

This is a frozen algebraic kernel of the required schematic form. The character-selection pattern is the lowest-order cubic combination that is basis-translation invariant, output-equivariant, and odd under output sign reversal.

---

## 2. Symmetry audit

### Hidden-unit permutations

A permutation \(j\mapsto\pi(j)\) permutes \(w_j\), \(D_1e_j\), and \(\alpha_j\) together. Since \(q_a\) sums over \(j\), it is unchanged.

### Positive ReLU gauge transformations

For a positive first-layer gauge \(c_j>0\),

\[
w_j\mapsto c_jw_j,
\qquad
D_1e_j\mapsto c_j^{-1}D_1e_j.
\]

Therefore

\[
\alpha_j\mapsto c_j^{-2}\alpha_j,
\qquad
(w_j^\top x)^2\mapsto c_j^2(w_j^\top x)^2,
\]

so

\[
\alpha_j(w_j^\top x)^2
\]

is exactly invariant.

### Output rotations

Under \(y\mapsto Oy\), the source projector rotates as \(V_rV_r^\top\mapsto OV_rV_r^\top O^\top\). Hence \(\alpha_j\) is invariant, while

\[
B\mapsto OB.
\]

### Input rotations

Under simultaneous rotation of weights and input design,

\[
w_j\mapsto Rw_j,
\qquad
x\mapsto Rx,
\]

the inner products \(w_j^\top x\) are preserved. More strongly, the collapse below holds under any literal rotation because each basis group has the same isotropic second moment.

### Basis-label translations

For \(a\mapsto a+t\),

\[
\widehat p_\chi\mapsto
(-1)^{\chi\cdot t}\widehat p_\chi,
\qquad
\widehat s_\chi\mapsto
(-1)^{\chi\cdot t}\widehat s_\chi.
\]

Every cubic character triple has total character

\[
\chi+\psi+(\chi+\psi)=0,
\]

so all phases cancel.

### Output sign

Under \(s\mapsto-s\), the weight-derived \(p\) is unchanged while the transcript cubic changes sign:

\[
B\mapsto-B.
\]

Thus the formal candidate has the intended signed parity.

---

## 3. Exact algebraic collapse

For every complete antipodal basis group,

\[
\frac1{|G_a|}\sum_{x\in G_a}xx^\top
=
\frac{\rho^2}{n}I.
\]

Therefore,

\[
\begin{aligned}
q_a
&=
\sum_j
\alpha_j
w_j^\top
\left(
\frac1{|G_a|}\sum_{x\in G_a}xx^\top
\right)
w_j\\
&=
\frac{\rho^2}{n}
\sum_j\alpha_j\|w_j\|_2^2.
\end{aligned}
\]

The right-hand side contains no \(a\). Hence

\[
\boxed{q_a=q_0\quad\text{for every basis }a.}
\]

It follows that

\[
p_a=0,
\qquad
\widehat p_\chi=0
\quad\forall\chi,
\]

and therefore

\[
\boxed{B(W,Q)=0.}
\]

This is not an approximate Kerdock cancellation. It follows exactly from the common second-moment matrix of every complete orthonormal basis.

### Equivalent Walsh-convolution interpretation

The weighted cubic bispectrum can be converted by Fourier inversion into a local skew of a filtered block transcript:

\[
B\propto
\sum_a t_a\|t_a\|^2,
\qquad
t=p*s.
\]

Since \(p=0\),

\[
t=0,
\]

so this is the zero member of the filtered-skew class. A symmetric cubic kernel would collapse to ordinary output skew; this candidate collapses still further, to zero.

---

## 4. Antipodal and Kerdock tests

**Antipodal pairing alone does not kill the primitive.** Because the primitive is quadratic,

\[
(w_j^\top(-x))^2=(w_j^\top x)^2.
\]

Both members of the antipodal pair contribute equally.

**The complete-basis property kills it.** Summing over an entire basis replaces every quadratic form by its trace. All nontrivial basis-index phase is erased before the Walsh transform.

Consequently:

- nonzero Walsh characters vanish;
- character scrambling changes nothing;
- basis translations change nothing;
- literal Kerdock rotations change nothing;
- the axis group has the same quadratic second moment and does not create an exception.

This provides an exact explanation for why a useful M194 primitive cannot consist solely of global normal energies and downstream adjoint magnitudes. It must retain a nonlinear gate-side quantity—such as signs, threshold crossings, or interactions between inherited boundaries—that is not determined by the basis second moment.

That agrees with the boundary-Stein diagnosis: for a deep preactivation, the required expectation contains inherited upstream boundary mass, not just a final or marginal boundary-energy term.

---

## 5. Oracle covariance gate

No empirical corpus evaluation is necessary because the observable is deterministically zero.

| Required quantity | Result |
|---|---:|
| Pooled \(\operatorname{Cov}(B,e)\) | \(0\), exactly |
| Within-network covariance | \(0\), exactly |
| \(\operatorname{Corr}(B,e)\) | Undefined because \(\operatorname{Var}(B)=0\) |
| Sign accuracy | Undefined; \(0.5\) under random/tie convention |
| Median within-network relationship | Zero |
| LONO coefficient stability | Degenerate |
| Tail magnitude | Zero |
| Gauge test | Exact pass |
| Permutation test | Exact pass |
| Shuffled-character null | Identical zero |
| Constant/skew control | Candidate collapses below this control |

The promotion gate requires stable positive grouped covariance before fitting. It fails exactly.

Therefore no \(\lambda\) is fitted. In any regularized least-squares implementation, the unique conservative result is

\[
\lambda=0.
\]

---

## 6. Cost

### Naive construction cost

Computing

\[
A=D_1^\top V_r
\]

by propagating an \(n\times r\) source backward through \(L-1\) dense \(n\times n\) matrices costs, under the convention that a multiply-add is two FLOPs,

\[
2(L-1)n^2r.
\]

The remaining costs are:

\[
n(2r-1)
\]

for the row norms of \(A\),

\[
n(2n-1)
\]

for the first-layer row norms, and

\[
2n-1
\]

for the final weighted contraction.

Thus

\[
F_{\rm naive}
=
2(L-1)n^2r+n(2r-1)+n(2n-1)+(2n-1).
\]

For

\[
n=256,\qquad L=32,\qquad r=36,
\]

this is

\[
\boxed{146{,}425{,}855\text{ FLOPs}}
\]

before any Walsh arithmetic.

Memory beyond existing weights is an \(n\times r\) adjoint array:

\[
256\times36=9{,}216
\]

scalars, approximately \(72\) KiB in float64.

### Simplified production cost

Once the identity is recognized, the correct implementation is

\[
B\leftarrow0.
\]

Therefore:

\[
\boxed{\text{incremental production FLOPs}=0,\qquad
\text{incremental memory}=0.}
\]

Computing the nominal 146 million-FLOP primitive would knowingly evaluate an algebraic zero.

---

## 7. Newness test

The candidate initially appears to be outside the static class because its kernel depends on:

- the realized first-layer boundary normals;
- downstream network path contractions;
- the network-specific output source.

But exact simplification removes all this dependence from the nontrivial Walsh modes. It therefore fails the required proof of newness:

\[
\boxed{
\text{weight-coupled syntax}
\not\Rightarrow
\text{weight-coupled phase information}.
}
\]

The previous experiments already found that fixed Walsh characters, output skew, and related signed block observables did not transfer. This construction does not contradict those results; it reduces to an even weaker statistic.

---

## Final M194 record

**Kernel tested:** downstream-adjoint-weighted boundary-normal-energy cubic Walsh bispectrum.

**Result:** every basis-resolved normal energy is identical by complete-basis Parseval.

**Identity:**

\[
q_a=
\frac{\rho^2}{n}
\sum_j
\|V_r^\top D_1e_j\|^2
\|w_j\|^2.
\]

**Observable:**

\[
B(W,Q)\equiv0.
\]

**Grouped covariance:**

\[
\operatorname{Cov}(B,e)=0.
\]

**Fitted estimator:** not run; covariance gate fails.

**Verdict:**

\[
\boxed{\textbf{FAIL — exact algebraic kill.}}
\]

This closes the entire subclass in which basis phase is generated only from **quadratic boundary-normal energies with node-independent downstream coefficients**. It does not close a genuinely gate-conditioned all-ancestor kernel whose coefficients depend on sign patterns or interacting boundary histories.

The v31 ledger correctly treats M194 as requiring one exact nonlinear primitive and kills it if grouped covariance is absent; this particular primitive fails even before data evaluation.
