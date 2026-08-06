# T71 — Arbitrary-total-mass signed floor and updated negative-mass frontier

**Date:** 2026-07-30  
**Status:** Computer-assisted specialization built from the independently rerun v21 order-320 certificate; abstract RKHS statements exact.

## 1. Why mass one should not be left implicit

A general linear estimator based on sampled function values need not have weights summing exactly to one. The v21 theorem was stated for mass-one rules. Because the depth-32 kernel has a dominant constant harmonic coefficient, dropping the mass constraint barely changes the answer—but that must be proved rather than assumed.

Let

\[
s=\sum_iw_i
\]

be the total mass. For one comparison profile

\[
L_a(t)=\sum_\ell a_\ell d_\ell G_\ell(t),
\]

put

\[
T=\sum_\ell a_\ell d_\ell,
\qquad S_2=\sum_\ell a_\ell^2d_\ell.
\]

The block traces become \(s a_\ell d_\ell\). Hence the rank bound gives

\[
R_{L_a^2}(Q)
\ge
S_2(1-2s)+{s^2T^2\over N}.
\]

For \(s=1\) this reduces to \(T^2/N-S_2\).

## 2. Constant-harmonic residual

Write the v21 comparison as

\[
H=\sum_jc_jL_j^2,
\qquad c_j={y_j\over B_j},
\]

where \(B_j=T_j^2/N-S_{2,j}\). Define

\[
A=\sum_j c_j{T_j^2\over N},
\qquad
B=\sum_jc_jS_{2,j},
\qquad
F_{21}=A-B.
\]

The exact certificate has

\[
B=3.84034975102058\times10^{-10}.
\]

The order-320 Maclaurin lower endpoints imply the certified constant Gegenbauer coefficient

\[
k_0\ge
0.97472998954171471231225812084613\ldots
\]

The comparison uses only \(B\) of this degree-zero capacity. Therefore the residual positive-definite kernel contributes

\[
(k_0-B)(1-s)^2.
\]

Combining all comparison profiles and the constant residual yields

\[
R_K(Q)
\ge
B(1-2s)+As^2+(k_0-B)(1-s)^2.
\]

This quadratic is minimized at

\[
s_*={k_0\over A+k_0-B}
=0.9999997660427924\ldots
\]

and gives

\[
\boxed{
R_K(Q)\ge
2.2804510650033141149442884649527472\times10^{-7}
}
\]

for **every static network-independent real-weight rule with at most 66,048 arbitrary nodes**, without any total-mass assumption.

Relative to the certified Kerdock upper endpoint,

\[
\boxed{
R_K(Q)\ge0.9370457376828170\,R_K(Q_{\rm Kerdock}),
}
\]

so

\[
\boxed{
\text{same-cost improvement}\le1.0671837667954823\times.
}
\]

Dropping mass one changes the floor by only

\[
5.33528\times10^{-14}
\]

in absolute MSE.

## 3. Scope consequence

The static signed theorem now covers:

- arbitrary real weights;
- arbitrary total weight;
- arbitrary spherical nodes;
- at most 66,048 nodes;
- rules fixed independently of the realized network;
- arbitrary linear reconstruction from the sampled values.

It still excludes:

- value-dependent or network-dependent weights;
- adaptive node selection from observed activations;
- nonlinear estimators;
- finite-width-specific objectives.

## 4. Updated bounded-negative-mass bridge

Let a mass-one signed rule have negative mass

\[
\nu=Q^-(S^{255}),
\qquad \|w\|_1=1+2\nu.
\]

The exact RKHS bridge gives

\[
R_K(Q)
\ge
\left(\sqrt{L_+}-\nu D_K\right)_+^2,
\]

where the latest certified positive-rule lower endpoint is

\[
L_+=2.4330928587565937917467205177\times10^{-7},
\]

and

\[
D_K\le0.2305727105711425311.
\]

The bridge and the v21 signed floor cross at

\[
\boxed{
\nu_*=6.81918054300676\times10^{-5},
\qquad
\|w\|_1=1.0001363836108601.
}
\]

Thus any rule with less negative mass than this has a stronger lower bound from proximity to the positive class.

Necessary negative mass to permit selected raw improvement factors over Kerdock:

| raw improvement | necessary \(\nu\) | necessary \(\|w\|_1\) |
|---:|---:|---:|
| 1.01× | \(1.03687\times10^{-5}\) | 1.00002074 |
| 1.05× | \(5.13135\times10^{-5}\) | 1.00010263 |
| 1.06718× | \(6.81918\times10^{-5}\) | 1.00013638 |
| 1.10× | \(9.93196\times10^{-5}\) | 1.00019864 |
| 1.20× | \(1.86167\times10^{-4}\) | 1.00037233 |
| 1.50× | \(3.92365\times10^{-4}\) | 1.00078473 |
| 2.00× | \(6.26410\times10^{-4}\) | 1.00125282 |
| 4.34× | \(1.11228\times10^{-3}\) | 1.00222457 |

These are necessary conditions only.

## 5. Interaction with T61

T61 proves something qualitatively stronger about the abstract floor:

> for every fixed finite \(V\), actual atomic rules with \(\|w\|_1\le V\) stay a positive distance above the v21 rank floor.

Therefore, if the v21 number is an infimum of actual signed cubature at all, approaching it requires

\[
\|w\|_1\to\infty,
\]

not merely crossing the small negative-mass threshold above.

The bridge controls modest signed perturbations quantitatively. The shared-profile compactness theorem shows that the rank boundary itself can only be approached by unboundedly ill-conditioned cancellations.

## 6. Competition interpretation

At equal evaluation cost, the static arbitrary-linear route is closed to at most about 6.72% raw improvement. To bridge a 4.34× adjusted gap while attaining the theorem floor, evaluation cost would need to fall to about

\[
{1.06718\over4.34}=0.2459
\]

of baseline, before accounting for signed-rule overhead or numerical instability.

The remaining static question is therefore not “can arbitrary signed weights find a large raw gain?” It is whether a radically cheaper rule can nearly saturate a boundary that T61 says is inaccessible to every bounded-total-variation atomic family.
