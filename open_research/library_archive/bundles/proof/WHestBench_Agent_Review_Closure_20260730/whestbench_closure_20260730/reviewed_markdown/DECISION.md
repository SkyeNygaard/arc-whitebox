# Decision — Agent 4 / T16

## Verdict

**VERIFIED: the all-degree T16 reduced-cost claim is proved for `d=256`, `N=66,048`.**

Every unused normalized Gegenbauer degree has strictly negative reduced cost:

\[
r_\ell<0\quad\text{for every integer }\ell\ge6.
\]

The proof is stronger than the old scan:

- degrees `6..14,658` are certified by exact integer arithmetic;
- degrees `14,659..infinity` are certified by an analytic `ell^{-127}` bound.

## Paper-ready theorem

**Theorem (all-degree dual reduced-cost negativity).** Let `G_ell` be the normalized Gegenbauer polynomial on `S^255`, and let `N=66,048`. Define `q_0=1-1/N` and `q_ell=-1/N` for `ell>=1`. Let `t_1,t_2,t_3` be the three roots of

\[
22102t^3+21930t^2-87t-85=0,
\]

and let the positive weights `lambda_j` be the unique weights matching the moments `q_0,...,q_5`. Then

\[
q_\ell-\sum_{j=1}^3\lambda_jG_\ell(t_j)<0
\]

for every integer `ell>=6`.

## Recommended claim status

- `T16 reduced-cost tail`: **PROVED**.
- `Degree-5 auxiliary is exactly all-degree LP-optimal`: **PROVED conditional on an exact primal-attainment/complementarity certificate**; otherwise retain that final equality step as open.

## Stop condition

The assigned tail branch has reached a theorem. Further numerical scanning is unnecessary. The natural remaining work belongs to the primal-certificate agent: formalize exact contact/attainment if the paper wants the strongest all-degree LP-optimality wording.
