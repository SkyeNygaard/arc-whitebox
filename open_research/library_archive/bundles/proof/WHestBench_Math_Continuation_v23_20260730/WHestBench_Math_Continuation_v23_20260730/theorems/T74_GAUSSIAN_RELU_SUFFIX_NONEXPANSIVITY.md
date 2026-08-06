# T74 — Gaussian–ReLU suffix nonexpansivity and replay-state transfer

**Date:** 2026-07-30  
**Status:** Exact ensemble theorem. Application requires the compared checkpoint states to be measurable before the independent Gaussian suffix is drawn. It does not identify the true integration target with a replayed checkpoint state.

## 1. One-layer identity

Let \(W\in\mathbb R^{m\times m}\) have independent entries

\[
W_{ij}\sim N(0,2/m),
\]

and let \(\sigma(t)=\max(t,0)\) act coordinatewise. For deterministic \(u,v\in\mathbb R^m\), put

\[
\rho={\langle u,v\rangle\over \|u\|\|v\|}
\]

when both vectors are nonzero, with the zero cases interpreted continuously. Define the normalized ReLU dual activation

\[
\kappa(\rho)=
{\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho\over\pi}.
\]

Then

\[
\boxed{
\mathbb E_W\|\sigma(Wu)-\sigma(Wv)\|^2
=
\|u\|^2+\|v\|^2-2\|u\|\|v\|\kappa(\rho).
}
\]

Since

\[
\kappa(\rho)-\rho
={\sqrt{1-\rho^2}-\rho\arccos\rho\over\pi}\ge0,
\qquad -1\le\rho\le1,
\]

we obtain

\[
\boxed{
\mathbb E_W\|\sigma(Wu)-\sigma(Wv)\|^2
\le \|u-v\|^2.
}
\]

The inequality is strict except on the positive collinear ray and the zero-vector boundary.

### Proof of \(\kappa(\rho)\ge\rho\)

Let \(h(\rho)=\kappa(\rho)-\rho\). Direct differentiation gives

\[
h'(\rho)=-{\arccos\rho\over\pi}\le0,
\qquad h(1)=0.
\]

Hence \(h(\rho)\ge0\) on \([-1,1]\).

## 2. Deep suffix theorem

Let

\[
F=\sigma(W_L\,\cdot)\circ\cdots\circ\sigma(W_1\,\cdot)
\]

where the \(W_j\) are independent width-\(m\) Gaussian He matrices as above. If random checkpoint states \(U,V\) are measurable independently of the suffix weights, then repeated conditional expectation gives

\[
\boxed{
\mathbb E\|F(U)-F(V)\|^2
\le
\mathbb E\|U-V\|^2.
}
\]

Thus an independent Gaussian–ReLU suffix cannot amplify checkpoint-state mean-square approximation error in ensemble expectation.

## 3. Source-coefficient corollary

Let a target-free checkpoint source basis be \(B=[b_1,\ldots,b_r]\), measurable before the suffix, and compare coefficients \(a,\widehat a\). Then

\[
U=u_0+Ba,
\qquad
V=u_0+B\widehat a
\]

satisfy

\[
\boxed{
\mathbb E\|F(U)-F(V)\|^2
\le
(a-\widehat a)^T(B^TB)(a-\widehat a).
}
\]

The physically correct coefficient metric at the checkpoint therefore upper-bounds the exact nonlinear final-output replay error under an independent random suffix. No linearization or gate-stability event is needed.

## 4. Replayable-target theorem

Suppose a desired final target has the form \(F(U_*)\) for a checkpoint state \(U_*\), and an estimator supplies \(\widehat U\), both independent of the future suffix. Then

\[
\boxed{
\mathbb E\|F(\widehat U)-F(U_*)\|^2
\le
\mathbb E\|\widehat U-U_*\|^2.
}
\]

This is a complete nonlinear transfer theorem **only when the target is actually replayable from a checkpoint state**.

## 5. A rigorous one-layer tail bound

For one row, ReLU is 1-Lipschitz, so

\[
|\sigma(w^Tu)-\sigma(w^Tv)|\le |w^T(u-v)|.
\]

Consequently, if \(D=\|\sigma(Wu)-\sigma(Wv)\|^2\), then

\[
\operatorname{Var}(D)
\le \sum_{i=1}^m\mathbb E(w_i^T(u-v))^4
={12\over m}\|u-v\|^4.
\]

Chebyshev therefore gives, for every \(t>0\),

\[
\Pr\left(D-\mathbb ED\ge t\|u-v\|^2\right)
\le {12\over mt^2}.
\]

This is conservative but completely distribution-free beyond the Gaussian row law.

## 6. Hostile scope audit

The theorem does **not** apply without modification when:

- the source basis or coefficients inspect the same suffix weights being averaged over;
- the physical benchmark target is an integration mean that is not equal to replay of a single checkpoint state;
- widths or variance scalings differ without inserting the corresponding norm factor;
- biases or normalization layers change the kernel identity;
- a fixed realized-network guarantee is claimed from an ensemble expectation.

In WHestBench, downstream-weighted source directions may depend on the suffix. Those constructions need leave-one-row, sample splitting, a conditional variant, or a deterministic operator argument. The theorem most directly favors prefix-measurable physical checkpoint channels.

## 7. Research implication

The nonlinear replay problem separates into two questions:

1. **State identifiability/replayability:** is there a checkpoint state or low-rank source state whose exact replay represents the desired correction?
2. **State approximation:** how accurately can that state be recovered legally?

For independent Gaussian suffixes, question 2 cannot worsen under exact nonlinear replay. The hard obstacle is therefore the target representation and the legality of the source, not generic gate amplification.
