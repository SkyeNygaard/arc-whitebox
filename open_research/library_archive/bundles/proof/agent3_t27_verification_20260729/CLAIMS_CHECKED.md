# Agent 3 — Claims Checked

## Verdict

**VERIFIED AFTER SPECIFIED CORRECTIONS.**

The central T27 theorem is correct for symmetrized antipodal line rules supported on the fixed 33,024-line Kerdock universe, with arbitrary real line weights summing to one. The optimization genuinely includes negative line weights, negative basis totals, unequal weights, arbitrary line deletion, empty bases, partial bases, and zero-total signed cancellations.

The paper-facing statement should be corrected to say **line budget** rather than unrestricted point budget, and it should explicitly define the empty-basis convention.

## Independently derived theorem

Let

\[
\bar K(t)=\frac{K(t)+K(-t)}2,
\]

and let the line rule be

\[
Q_w f=\sum_{b=1}^{129}\sum_{i=1}^{256}w_{bi}\frac{f(u_{bi})+f(-u_{bi})}{2},
\qquad \sum_{b,i}w_{bi}=1.
\]

Write

\[
A=\bar K(1),\qquad O=\bar K(0),\qquad C=\bar K(1/16),
\qquad S_b=\sum_iw_{bi}.
\]

Because the Kerdock projective lines have only the three pair classes “same line,” “distinct lines in one basis,” and “different mutually unbiased bases,”

\[
\begin{aligned}
R(w)
&=\sum_{b,i,c,j}w_{bi}w_{cj}\bar K(\langle u_{bi},u_{cj}\rangle)-A_0\\
&=(C-A_0)+(O-C)\sum_bS_b^2+(A-O)\sum_{b,i}w_{bi}^2.
\end{aligned}
\]

This derivation is exact and does not use positivity of the weights.

For a support with `r_b` available lines in basis `b`, set `S_b=0` when `r_b=0`. For `r_b>0`,

\[
\sum_iw_{bi}^2\ge \frac{S_b^2}{r_b},
\]

with equality exactly when all retained line weights in that basis equal `S_b/r_b`. Hence

\[
R(w)\ge (C-A_0)+\sum_{b:r_b>0}c(r_b)S_b^2,
\qquad
c(r)=(O-C)+\frac{A-O}{r}.
\]

At dimension 256 and depth 32,

\[
A-O=0.011988581160655598>0,
\]

\[
O-C=-9.468153657654632\times10^{-6}<0,
\]

and

\[
c(256)=3.73622415011563\times10^{-5}>0.
\]

Since `c(r)` decreases on `1,…,256`, this proves `c(r)>0` throughout the allowed range.

Weighted Cauchy–Schwarz, valid for arbitrary real basis totals, gives

\[
1=\left(\sum_b S_b\right)^2
\le
\left(\sum_b c(r_b)S_b^2\right)
\left(\sum_b\frac1{c(r_b)}\right).
\]

Therefore the unique active-basis optimum is

\[
S_b=\frac{c(r_b)^{-1}}{\sum_jc(r_j)^{-1}},
\qquad
w_{bi}=\frac{S_b}{r_b},
\]

and

\[
R_{\min}(r_1,\ldots,r_{129})
=(C-A_0)+\frac1{\sum_b c(r_b)^{-1}}.
\]

All optimal active basis totals and line weights are positive. Signed weights are allowed by the theorem but are never optimal.

Define

\[
h(0)=0,\qquad
h(r)=\frac1{c(r)}=\frac{r}{(A-O)+(O-C)r}\quad (r>0).
\]

Writing `a=A-O>0` and `b=O-C<0`, the denominator remains positive through `r=256`, and

\[
h'(r)=\frac{a}{(a+br)^2}>0,
\qquad
h''(r)=\frac{-2ab}{(a+br)^3}>0.
\]

Thus `h` is increasing and strictly convex. Its discrete increments are strictly increasing. If two bases satisfy `1≤r≤s≤255`, transferring one line from the smaller to the larger strictly increases `h(r)+h(s)`. Repeated exchanges leave `q=floor(P/256)` complete bases and, when nonzero, one partial basis of size `s=P-256q`.

Hence, for every **line budget** `1≤P≤33,024`, the unique optimal count multiset is

\[
(\underbrace{256,\ldots,256}_{q\text{ times}},s,0,\ldots,0),
\]

with the `s` entry omitted when `s=0`. Basis labels and retained line identities inside a basis are symmetric, so the support is nonunique only under those symmetries.

## Edge cases checked

- `P=0`: infeasible because the weights must sum to one.
- `P=1`: one line with weight one.
- `P=255`: one 255-line partial basis.
- `P=256`: one complete basis.
- `P=257`: one complete basis plus one one-line partial basis.
- `P=33,023`: 128 complete bases plus a 255-line partial basis.
- `P=33,024`: all 129 complete bases, with uniform basis and line weights.
- A nonempty basis with total mass zero but nonzero signed line weights is strictly dominated: its contribution is `(A-O) sum_i w_i^2>0`.
- A retained zero-weight line is mathematically irrelevant and should not count as support when support means nonzero weights.

## Full-design numerical reproduction

The independent kernel reconstruction gives

\[
A_0=0.9747299895417145
\]

and full-design risk

\[
R=2.43366035798\times10^{-7},
\]

matching the archived value to displayed precision.
