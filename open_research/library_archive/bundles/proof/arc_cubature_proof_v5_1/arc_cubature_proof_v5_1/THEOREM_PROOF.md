# Computer-assisted near-optimality of Kerdock/MUB cubature

## 1. Exact statement

Define

\[
\kappa(t)=\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi},\qquad
K_0(t)=t,\qquad K_{r+1}=\kappa\circ K_r.
\]

Let \(\sigma\) be normalized surface measure on \(S^{255}\), and set

\[
A_0=\int_{S^{255}}K_{32}(\langle x,y\rangle)\,d\sigma(y),
\]

which is independent of \(x\). For a cubature rule

\[
Q=(x_1,\ldots,x_m;w_1,\ldots,w_m),\qquad
m\le 66{,}048,
\]

with \(x_i\in S^{255}\), \(w_i\ge0\), and \(\sum_iw_i=1\), define its
kernel discrepancy

\[
D_{K_{32}}^2(Q)=
\sum_{i,j}w_iw_jK_{32}(\langle x_i,x_j\rangle)-A_0.
\]

Let \(Q_{\mathrm K}\) be the uniform rule on the antipodal union of 129
pairwise mutually unbiased orthonormal bases in \(\mathbb R^{256}\). Such a
family is supplied by the classical real Kerdock construction.

### Theorem

\[
\frac{D_{K_{32}}^2(Q_{\mathrm K})}
{\inf_Q D_{K_{32}}^2(Q)}
\le 1.000233655010295.
\]

Equivalently, the Kerdock/MUB rule is at most

\[
\boxed{0.02336550102949\%}
\]

above the infimum over all rules in the class above.

This is a **one-sided** theorem. It does not prove that Kerdock is genuinely
suboptimal; the true excess could be zero.

## 2. Random-network interpretation

Suppose \(f\) is a random field on \(S^{255}\) satisfying

\[
\mathbb E[f(x)f(y)]=K_{32}(\langle x,y\rangle).
\]

For a deterministic rule, or a randomized rule chosen independently of \(f\),
rotational invariance gives

\[
\begin{aligned}
\mathbb E[(Qf-If)^2]
&=\sum_{i,j}w_iw_jK_{32}(\langle x_i,x_j\rangle)
 -2A_0+A_0\\
&=D_{K_{32}}^2(Q).
\end{aligned}
\]

Thus the deterministic kernel theorem becomes an ensemble-MSE theorem for
network-independent cubature. It does **not** cover nodes or weights selected
from a realized network, pilot adaptation, nonlinear estimators, signed
weights, or finite-width-specific estimators.

The identification of \(K_{32}\) with the normalized infinite-width ReLU
second-moment kernel is the standard NNGP covariance recursion. The core
near-optimality theorem itself only uses the explicit function \(K_{32}\)
defined above.

## 3. Positive-definite Gegenbauer decomposition

Let \(G_\ell^{(256)}\) denote the normalized Gegenbauer polynomial with
\(G_\ell^{(256)}(1)=1\). The spherical-harmonic addition theorem gives

\[
G_\ell^{(256)}(\langle x,y\rangle)
=\frac1{d_\ell}\sum_{r=1}^{d_\ell}Y_{\ell r}(x)Y_{\ell r}(y)
\]

for an orthonormal basis of degree-\(\ell\) spherical harmonics. Therefore,
for arbitrary real weights,

\[
\sum_{i,j}w_iw_jG_\ell^{(256)}(\langle x_i,x_j\rangle)
=\frac1{d_\ell}\sum_r\left(\sum_iw_iY_{\ell r}(x_i)\right)^2\ge0.
\]

This is the only positive-definiteness fact needed in the Delsarte argument.

## 4. Exact auxiliary polynomial witness

The proof uses

\[
h(t)=\sum_{\ell=0}^5c_\ell G_\ell^{(256)}(t),
\]

where the six \(c_\ell\) are exact rationals stored in
`auxiliary_coefficients_d256_L32_deg5.json`. Every \(c_\ell\) for
\(\ell\ge1\) is nonnegative.

A directed-rounding interval certificate proves

\[
h(t)<K_{32}(t)\qquad\text{for every }t\in[-1,1].
\]

The proof consists of:

1. exact rational conversion from Gegenbauer to monomial form;
2. exact Bernstein certification that \(h'(t)>0\) on \([-1,1]\);
3. directed-rounding enclosures for \(K_{32}\), \(K_{32}'\), and
   \(K_{32}''\);
4. 1,421 certified curvature subintervals with exact rational coverage;
5. a machine-checked sign diagram showing that the only possible interior
   maxima of \(g=h-K_{32}\) lie in three strict-concavity boxes;
6. strict negative upper bounds in those three boxes and at both endpoints.

The final global enclosure is

\[
\max_{[-1,1]}(h-K_{32})
\le -1.0045862406584556\times10^{-13}.
\]

The root locations used to choose boxes are merely untrusted hints. Every sign,
coverage, and inequality required by the proof is re-established with directed
interval arithmetic.

## 5. Weighted Delsarte bound

Write \(h=\sum_{\ell=0}^5c_\ell G_\ell^{(256)}\). By the addition theorem and
\(c_\ell\ge0\) for \(\ell\ge1\),

\[
\sum_{i,j}w_iw_jh(\langle x_i,x_j\rangle)\ge c_0.
\]

Since \(K_{32}-h\ge0\) pointwise and \(w_iw_j\ge0\), discarding every
off-diagonal residual term gives

\[
\begin{aligned}
E_K(Q)
&:=\sum_{i,j}w_iw_jK_{32}(\langle x_i,x_j\rangle)\\
&\ge c_0+\bigl(K_{32}(1)-h(1)\bigr)\sum_iw_i^2.
\end{aligned}
\]

Here \(K_{32}(1)=1\). Cauchy--Schwarz and \(m\le N=66{,}048\) imply

\[
\sum_iw_i^2\ge\frac1m\ge\frac1N,
\]

so every admissible rule satisfies

\[
E_K(Q)\ge B:=c_0+\frac{1-h(1)}{66{,}048}.
\]

The exact rational \(B\) is enclosed as

\[
B\in[
0.9747302328509000564066126514960711073288701902848240637848543566326762,
\]
\[
0.9747302328509000564066126514960711073288701902848240637848543566326763].
\]

## 6. Rigorous spherical mean

Differentiating \(\kappa\) gives

\[
\kappa'(t)=\frac12+\frac{\arcsin t}{\pi}.
\]

The standard positive-coefficient series for \(\arcsin\), followed by one
integration, shows that every Maclaurin coefficient of \(\kappa\) is
nonnegative. Since \(\kappa(1)=1\), the coefficient sum is one. Composition
preserves both properties, so

\[
K_{32}(t)=\sum_{k\ge0}b_kt^k,\qquad b_k\ge0,\qquad\sum_kb_k=1.
\]

For \(T=\langle x,U\rangle\), with \(U\sim\sigma\), odd moments vanish and

\[
\mathbb E[T^{2k}]=\prod_{j=0}^{k-1}\frac{2j+1}{256+2j}.
\]

The verifier computes directed intervals for the first 30 coefficients of
\(K_{32}\) by interval Taylor-jet composition. Every omitted even moment is at
most the first omitted one, and the omitted coefficient mass is at most one.
This yields

\[
A_0\in[
0.9747299895417147123122580852641911964220890140486520806041407254,
\]
\[
0.9747299895417147123124869552974612893380014764279519548528878898].
\]

The interval width is \(2.29\times10^{-22}\). A separate 90-digit quadrature
implementation agrees, but that non-rigorous audit is not used in the proof.

## 7. Kerdock/MUB energy

For a fixed signed MUB node, the inner-product multiplicities are

- one at \(1\);
- one at \(-1\);
- 510 at \(0\);
- 32,768 at \(+1/16\);
- 32,768 at \(-1/16\).

These counts follow directly from the antipodal union of 129 mutually unbiased
bases and sum to 66,048. Hence the uniform Kerdock energy is a five-value kernel
sum, rigorously enclosed by

\[
E_K(Q_{\mathrm K})\in[
0.9747302329077504666127808462108414633985481621607157811601376765058907,
\]
\[
0.9747302329077504666127808462108414633985481621607157811601376765059437].
\]

The proof is conditional only on the stated real-MUB incidence property.
Existence of such a 129-basis family in dimension 256 is the classical Kerdock
construction; it is a standard external construction rather than part of the
interval certificate.

## 8. Final one-sided comparison

The Kerdock discrepancy satisfies

\[
D_{K_{32}}^2(Q_{\mathrm K})
\le 2.43366035754300523\times10^{-7}.
\]

Every admissible rule satisfies

\[
D_{K_{32}}^2(Q)
\ge 2.43309185344094126\times10^{-7}.
\]

Therefore

\[
1\le
\frac{D_{K_{32}}^2(Q_{\mathrm K})}
{\inf_QD_{K_{32}}^2(Q)}
\le1.0002336550102949,
\]

and the true relative excess lies in

\[
[0,\,0.02336550102949\%].
\]

Similarly, the true additive suboptimality lies in

\[
[0,\,5.68504102061682\times10^{-11}].
\]

## 9. Scope and trust base

### Covered

- deterministic probability-weighted rules with at most 66,048 support points;
- randomized rules independent of the random field, after conditioning and
  averaging;
- the explicit infinite-width kernel \(K_{32}\) in dimension 256.

### Not covered

- signed weights;
- network-adaptive points or weights;
- pilot-sample adaptation;
- nonlinear or analytic-plus-residual estimators;
- finite-width networks.

### Trust base

The computer-assisted component trusts:

- CPython's exact integer and `Fraction` arithmetic;
- CPython `decimal`/libmpdec directed-rounding semantics for addition,
  multiplication, division, and neighboring representable values;
- the explicitly verified bracketing implementation of square root;
- the source files fixed by `PROOF_MANIFEST.sha256`.

It also uses the standard spherical-harmonic addition theorem and the classical
existence of the real Kerdock MUB family. This is a rigorous computer-assisted
proof within that explicit trust base, not a Lean/Coq formalization.

## 10. Reproduction

Fast theorem verification:

```bash
python verify_theorem_package.py
python verify_manifest.py
```

Full clean-room regeneration of every formal interval chunk:

```bash
./FULL_PROOF_REPRODUCE.sh
```

## References for external standard inputs

- A. R. Calderbank, P. J. Cameron, W. M. Kantor, and J. J. Seidel,
  “Z4-Kerdock Codes, Orthogonal Spreads, and Extremal Euclidean Line-Sets,”
  *Proceedings of the London Mathematical Society* 75(2), 436–480 (1997).
  This supplies the classical real Kerdock line/MUB construction used here.
- The spherical-harmonic addition theorem and normalized Gegenbauer
  positive-definiteness are standard harmonic-analysis results; the exact
  sum-of-squares implication needed by this proof is written out in Section 3.
- The interpretation of the explicit recursion `K_{r+1}=kappa∘K_r` as the
  infinite-width ReLU covariance kernel is standard neural-network Gaussian
  process theory; the core deterministic kernel theorem does not depend on
  that interpretation.
