# Infinite Hermite factor — remaining certification plan

## Target lemma

For the frozen interval solution `(r1,r2,r3)`, let `q` be the degree-five Hermite interpolant to `K32`, and choose the branch

\[
L(t)=\sqrt{K_{32}(t)-q(t)}
\]

whose zeros at the three contact points have the signs encoded by `P(t)`. Prove

\[
[t^n]L(t)>0\quad\text{for every }n\ge2.
\]

## Recommended proof split

### Finite prefix

Use Arb ball arithmetic. Avoid ascending division by `P^2`; it is ill-conditioned because `P(0)` is small. Divide the Taylor polynomial from high degree downward by the monic polynomial `P^2`, then take the square root recursively. Verify a prefix at least through 8,000 in chunks.

### Eventual tail

Use a slit-domain Darboux argument.

1. Expand `K`, `q`, and `L` in `u=sqrt(1-t)` at `+1`.
2. Expand in `v=sqrt(1+t)` at `-1`.
3. Keep enough odd half-powers that their exact coefficient contribution can be bounded directly.
4. Bound the remaining local terms on small endpoint circles.
5. Bound the middle contour on `|t|=1+eta` with endpoint dents.
6. Use the measured amplitude ratio `|B-|/B+ < 4.867e-4` to establish a uniform positive lower bound.

## Alternative sufficient lemma

The calculations also show `[t^n] log S(t)>0` through the checked range. If this can be proved for every `n>=1`, then

\[
\sqrt{S(t)}=\sqrt{S(0)}\exp\left(\frac12\sum_{n\ge1}[t^n]\log S(t)\,t^n\right)
\]

has nonnegative coefficients automatically. The final multiplication by the cubic `P` would still require a short coefficient inequality, but this may be easier than direct positivity of `L`.

## Failure conditions

Quarantine the candidate if any of the following occurs:

- the interval root boxes are not unique;
- `K-q` develops an additional zero on `[-1,1]`;
- a directed Taylor coefficient of `L` is nonpositive;
- the dented-contour analytic region contains an unhandled preimage branch point;
- the rigorous floor falls below `20/21` after interval widening.
