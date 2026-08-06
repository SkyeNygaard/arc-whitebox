# Prompt 1 continuation — canonical infinite Hermite factor

**Status:** Exact structural reduction and extensively verified conditional theorem candidate. One all-degree sign lemma remains open. Do not cite the 1.05× conclusion as unconditional until that lemma is interval-certified.

## 1. Result in one line

The stabilized degree-63 through degree-123 comparison kernels are truncations of a single canonical factor

\[
L(t)^2=K_{32}(t)-q(t),
\]

where `q` is a degree-five Hermite interpolant at three interior double-contact points. If the Taylor coefficients of `L` are nonnegative from degree two onward, then `L` is positive definite on every sphere and the weighted-rank proof gives

\[
R(Q)\ge 2.3232332157460956007\times10^{-7}
      \ge 0.9546250809178666303\,R_{\rm Kerdock}.
\]

Thus the same-cost raw improvement is at most

\[
1.047531664513261731\times,
\]

which would rule out every `1.05x` static signed gain.

The numerical margin over the required floor `20/21` is

\[
0.9546250809178666303-\frac{20}{21}
=0.00224412853691424942.
\]

## 2. The three-node Hermite construction

Let `K=K_32` be the normalized depth-32 ReLU kernel. The unique nearby solution of the equal-eigenvalue equations has contact points

\[
\begin{aligned}
r_1&=-0.10990557917082860659389183515567186756\ldots,\\
r_2&=-0.00221644643608450926048040158134283590\ldots,\\
r_3&= \phantom{-}0.10550430212733650322428064649498622266\ldots.
\end{aligned}
\]

Let `q` be the unique polynomial of degree at most five satisfying

\[
q(r_i)=K(r_i),\qquad q'(r_i)=K'(r_i),\qquad i=1,2,3.
\]

Its monomial coefficients, low to high degree, are

\[
\begin{aligned}
q_0&=0.97472047512360818874594392332449915056\ldots,\\
q_1&=0.00277530920126995461395118350384323324\ldots,\\
q_2&=0.00241781048536968422159911422584710486\ldots,\\
q_3&=0.00180972208243284365845725694176111449\ldots,\\
q_4&=0.00152931532095495331904719869596681996\ldots,\\
q_5&=0.00123830589643062783981919946959869259\ldots.
\end{aligned}
\]

Writing

\[
P(t)=\prod_{i=1}^3(t-r_i),
\]

the double-contact conditions imply

\[
K(t)-q(t)=P(t)^2S(t).
\]

The stable branch is

\[
L(t)=P(t)\sqrt{S(t)}.
\]

The roots and polynomial coefficients are frozen in `INFINITE_HERMITE_FACTOR_CANDIDATE.json`.

## 3. Why this is the limit of the finite certificates

The first four normalized harmonic eigenvalues of `L` satisfy

\[
c_0=c_1=c_2=c_3
=s=1.1537978723095753995324442523498259163\times10^{-8}
\]

to more than 100 decimal digits. The subsequent coefficients reproduce the stabilized low-degree coefficients of every optimized cutoff from 63 through 123. In particular,

\[
\frac{c_4}{s}=0.0066485616863371357\ldots,
\]

matching the finite sequence.

This explains the long equioscillation observed in the finite programs: for every harmonic degree at least six,

\[
[L^2]_r=[K-q]_r=k_r,
\]

because `q` has degree five. The apparent hundreds of active coefficient constraints are the shadow of the exact identity `L^2=K-q`.

## 4. Exact rank-defect formula

At the target budget

\[
N=66{,}048=1+256+32{,}895+32{,}896,
\]

the rank optimizer retains all degree-zero, degree-one and degree-two eigenvalues and 32,896 of the degree-three eigenvalues. Since these eigenvalues are all exactly `s`, the exact trace-preserving obstruction simplifies.

Let

\[
T=L(1),\qquad b_0=[G_0](L^2).
\]

Then the omitted trace is `T-Ns` and the omitted squared spectral mass is `b_0-Ns^2`, hence

\[
F_N=b_0-Ns^2+\frac{(T-Ns)^2}{N}
    =b_0-2sT+\frac{T^2}{N}.
\]

No infinite harmonic summation is required.

Because `L^2=K-q`,

\[
T=\sqrt{K(1)-q(1)}=\sqrt{1-q(1)}
=0.12453538408795207791484609638713092911\ldots,
\]

and

\[
b_0=k_0-q_0^{(G)}
=3.8208314118336746722082777492191527579\times10^{-10}.
\]

Substitution gives

\[
F_N=2.3232332157460956006982892551301585900\times10^{-7}.
\]

If `L` is positive definite, the comparison domination factor is exactly one: degrees at least six agree with `K`, and the Gegenbauer coefficients of `q` in degrees zero through five are positive.

## 5. The sole remaining theorem lemma

A sufficient condition for positive definiteness is

\[
[t^n]L(t)\ge0\qquad(n\ge2),
\]

together with the already positive degree-zero and degree-one Gegenbauer coefficients. Every monomial `t^n` has nonnegative normalized Gegenbauer projections, so this condition implies all harmonic coefficients of degree at least two are nonnegative.

What has been verified:

- 505-term high-precision MPFR-derived factor series;
- every coefficient of `S` and `sqrt(S)` through degree 505 is positive;
- every Taylor coefficient of `L` from degree 2 through 505 is positive;
- every Gegenbauer coefficient of `L` through degree 300 is positive;
- an independent long-double recurrence has no negative `S`, `sqrt(S)`, `L_{n>=2}`, or `log S` coefficient through orders 8,191 and 16,383;
- the two independent truncations agree exactly at all recorded interior indices through degree 8,000;
- the long-double first 511 coefficients agree with the MPFR midpoint calculation to maximum relative error about `2.2e-16`.

These are strong evidence, not an all-degree proof.

## 6. Eventual positivity mechanism

The only dominant unit-circle singularities are the ReLU branch points at `t=1` and `t=-1`. The leading `L` singular amplitudes are

\[
B_+=38.55680901191435065\ldots,
\qquad
|B_-|=0.01876454147244204914\ldots.
\]

Thus

\[
\frac{|B_-|}{B_+}=0.00048667257362100846\ldots,
\]

so the nonalternating positive-endpoint term dominates the alternating negative-endpoint term by a factor above 2,000. This proves the correct asymptotic sign once explicit remainder bounds are supplied.

The positive-endpoint Puiseux expansion was generated through `(1-t)^{19/2}`. Its odd half-power coefficients through that order are all positive. At degree 8,000 the finite expansion error is already on the scale predicted by the negative-endpoint contribution.

## 7. Exact remaining certification task

The best next task is narrow:

1. Enclose the three roots with interval Newton or Krawczyk arithmetic.
2. Reconstruct `q`, `P`, and the first `M` coefficients of `L` with directed Arb/MPFR intervals.
3. Verify `[t^n]L>0` for `2<=n<M`.
4. Construct directed Puiseux expansions near both endpoints, with explicit coefficient remainder bounds.
5. Bound the analytic middle-contour contribution using a dented/keyhole contour.
6. Choose an explicit `M` for which the positive `+1` contribution exceeds the absolute `-1` contribution plus both remainders for every `n>=M`.
7. Recompute `s`, `b_0`, `L(1)`, `F_N`, and the Kerdock ratio with directed endpoints.

The likely bridge point is far below 8,000, but 8,000 is already computationally inexpensive and provides a generous target.

## 8. Hostile interpretation

This continuation does **not yet** prove the 1.05× theorem. It proves:

- an exact algebraic identity conditional on the numerically isolated roots;
- a closed-form rank obstruction;
- a candidate floor with comfortable threshold margin;
- positivity through a very long finite prefix;
- a quantitatively dominant eventual-positive singularity;
- a precise, one-lemma route to completion.

A counterexample would now need either a negative Taylor coefficient beyond the verified range or failure of the claimed analytic continuation/remainder control. The first possibility is numerically implausible; the second is the real proof-engineering risk.
