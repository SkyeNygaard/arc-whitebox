# Agent 5 — Signed-weight extension audit

## Decision

**VERIFIED AFTER STRENGTHENING; NUMERICAL CURVE COMPLETE; PRACTICALLY VACUOUS OUTSIDE THE KERDOCK-LINE UNIVERSE.**

Proposition 5 is correct under its stated assumptions. The proof uses only positive definiteness of the normalized Gegenbauer kernels, the certified minorant `h <= K`, and a bound on the negative mass of the signed measure. I found no counterexample.

Two additions materially improve the result:

1. The residual supremum is not merely bounded numerically:

   \[
   M=\sup_{-1\le t\le1}(K_{32}(t)-h(t))
    =K_{32}(1)-h(1)
    =\frac{156999263604490023}{9223372036854775808}.
   \]

   Hence

   \[
   M=0.017021894267861247050481476872363373331609182059764862060546875
   \]

   exactly.

2. The memo's diagonal step `sum w_i^2 >= 1/N` can be strengthened by separating positive and negative supports. This gives an exact integer support-count envelope and slightly larger necessary negative masses.

The strengthened curve still does not close unrestricted signed cubature. Even a certified **10%** Kerdock-relative improvement requires only

\[
\beta\ge 7.1332\times10^{-7},
\]

so the total variation need only exceed one by about `1.43e-6`. That is too small to be a meaningful competition-level exclusion.

## 1. Independent proof of Proposition 5

Let

\[
h(t)=\sum_{\ell=0}^L c_\ell G_\ell(t),\qquad
c_\ell\ge0\ (\ell\ge1),\qquad h(t)\le K(t),
\]

and define `q=K-h`, `q_1=q(1)`, and `M=sup q`. Let the real weights sum to one and let

\[
\beta=\sum_{w_i<0}|w_i|.
\]

The positive weights therefore have total mass `1+beta`.

The addition theorem makes every matrix

\[
G_\ell(\langle x_i,x_j\rangle)
\]

positive semidefinite, even for arbitrary real weights. Therefore

\[
\sum_{i,j}w_iw_jh(\langle x_i,x_j\rangle)\ge c_0.
\]

For the residual:

* diagonal terms equal `q_1 sum_i w_i^2`;
* same-sign off-diagonal terms are nonnegative;
* the ordered absolute mass of opposite-sign products is exactly
  `2 beta (1+beta)`;
* each opposite-sign residual value is at most `M`.

Thus

\[
\sum_{i,j}w_iw_jq(\langle x_i,x_j\rangle)
\ge q_1\sum_iw_i^2-2M\beta(1+\beta).
\]

Using `sum_i w_i^2 >= 1/N` proves the memo's statement:

\[
\boxed{
E_K(w)\ge c_0+\frac{q_1}{N}-2M\beta(1+\beta).
}
\]

Every inequality direction is correct. The result applies pointwise to deterministic signed rules and conditionally to randomized rules independent of the random field.

## 2. Stronger integer-support version

Suppose `beta>0`, with `p` positive and `n` negative nonzero weights. Groupwise Cauchy gives

\[
\sum_iw_i^2
\ge \frac{(1+\beta)^2}{p}+\frac{\beta^2}{n}.
\]

Because `p+n <= N`, the sharp lower envelope obtainable from support counts alone is

\[
D_N(\beta)=
\min_{1\le n\le N-1}
\left[
\frac{(1+\beta)^2}{N-n}+\frac{\beta^2}{n}
\right].
\]

Therefore:

\[
\boxed{
E_K(w)\ge c_0+q_1D_N(\beta)-2M\beta(1+\beta),\qquad \beta>0.
}
\]

At `beta=0`, use `D_N(0)=1/N`. This dominates Proposition 5. A simpler but weaker strengthening is

\[
\sum_iw_i^2\ge\frac{(1+2\beta)^2}{N},
\]

from the total variation `sum |w_i|=1+2 beta`.

For every threshold in the requested curve, the minimizing support split is one negative node and `N-1` positive nodes.

## 3. Directed-interval certificate for M

The exact rational witness gives

\[
q_1=1-h(1)=\frac{156999263604490023}{9223372036854775808}.
\]

The global maximum proof uses the rational cut `a=37/50`.

### Left interval `[-1,a]`

Exact Bernstein coefficients prove `h'(t)>0` throughout `[-1,1]`; the smallest coefficient is

\[
\frac{227100923375046163}{2351959869397967831040}>0.
\]

The deep ReLU kernel is increasing. Directed interval evaluation gives

\[
q(t)\le K(a)-h(-1)
\le 0.006637042585524519412595400468<M.
\]

### Right interval `[a,1]`

The normalized ReLU map is increasing and convex, so every iterate `K_r` is increasing and convex. Hence `K_32'` is increasing. Directed interval and exact Bernstein bounds give

\[
K_{32}'(a)-\sup_{[a,1]}h'
\ge 0.0007484168720425654682986292680>0.
\]

Thus `q` is strictly increasing on `[a,1]`, and its maximum is attained at `1`. Together with the left-region bound, this proves `M=q(1)` exactly.

The machine-readable certificate is `M_CERTIFICATE.json`; the reproducer is `signed_weight_certificate.py`.

## 4. Negative-mass exclusion curve

Two interpretations are reported.

* **Positive-certificate curve:** target is the stated percentage below the certified positive-class lower bound.
* **Kerdock-relative curve:** target is the stated percentage below the Kerdock MSE. This uses the conservative directed bounds
  `L = positive MSE lower bound` and `U = Kerdock MSE upper bound`, with
  `Delta=max(0,L-(1-p)U)`.

The table uses the stronger integer support-count envelope.

| Target improvement | beta required below positive certificate | beta required for Kerdock-relative target |
|---:|---:|---:|
| 0.01% | `8.2933e-10` | **not excluded** (`0`) |
| 0.1% | `7.2617e-9` | `5.5934e-9` |
| 1% | `7.1585e-8` | `6.9932e-8` |
| 5% | `3.5747e-7` | `3.5588e-7` |
| 10% | `7.1482e-7` | `7.1332e-7` |

The 0.01% Kerdock-relative target is not excluded because it is smaller than the existing certified positive-class gap of approximately `0.0233655%`.

The original Proposition 5 curve and full directed decimals are in `NEGATIVE_MASS_EXCLUSION_CURVE.csv` and `.json`.

## 5. Sharpness tests

### Low-dimensional adversarial optimization

I embedded four-node rules on `S^1` and `S^2` into `S^255`, preserving applicability of the dimension-256 Gegenbauer witness. Differential evolution jointly optimized node positions and positive weight allocation for sign split `3 positive + 1 negative` at `beta=0.1` and `0.5`.

The smallest slack above the strengthened bound was approximately:

* `0.0044980` at `beta=0.1`;
* `0.0162999` at `beta=0.5`.

No near-saturating geometry was found. These are exploratory global-search results, not proofs, but they show the residual penalty and diagonal/harmonic lower bounds cannot be simultaneously approached in these small geometries.

### Complete Kerdock-line universe

The exact association-scheme objective is stronger than Proposition 5. Inside all 33,024 projective Kerdock lines, arbitrary real line weights optimize at equal positive weights within complete bases; negative line or basis weights cannot improve the objective.

As a stress test, I generated 200 random signed perturbations at each of
`beta = 1e-6, 1e-4, 0.01, 0.1, 0.5`. There were zero improving trials. The minimum risk excesses ranged from `1.44e-8` to `1.88e-6`.

The theorem, not the random search, is authoritative here.

## 6. Why the bound is loose

Saturating the negative penalty requires nearly every positive-negative pair to occur where `q(t)` is close to its maximum. The maximum occurs at `t=1`, so this asks opposite-sign clouds to be almost coincident.

At the same time, saturating the harmonic and diagonal lower bounds asks for highly diffuse, design-like weights and geometry. Those requirements conflict. Proposition 5 discards this geometry and therefore pays the worst residual value on the entire opposite-sign product mass.

The scale mismatch is severe: `M` is about `1.7e-2`, while the Kerdock MSE is about `2.43e-7`. Consequently a sub-part-per-million negative mass is already enough to make the certificate permit a large relative MSE gain.

## 7. Scope

The result covers signed linear cubature measures with total mass one, at most 66,048 consolidated support points, and the explicit dimension-256 depth-32 infinite-width ReLU kernel.

It does not prove:

* optimality over arbitrary signed nodes outside the Kerdock line universe;
* finite-width width-256 optimality;
* network-adaptive signed-weight optimality;
* nonlinear estimator optimality;
* that satisfying the beta threshold is sufficient for an improvement.

`beta` should be interpreted as the Jordan negative mass of the consolidated signed measure. If canceling duplicate weights at the same node are retained as separate entries, beta becomes representation-dependent and can be inflated without changing the estimator.

## 8. Reproducibility status

The immutable v5 proof archive's fast theorem verifier passed, and all 32 manifest files verified. I also independently generated the new directed-interval `M` certificate and all curves.

I launched the full clean-room regeneration, but the environment's single-command runtime cap interrupted the serial regeneration before all curvature chunks completed. I therefore do **not** claim an independent complete regeneration of all 1,421 original subintervals in this run. The supplied proof audit reports that clean-room regeneration separately; this audit independently checked the shipped certificate and manifest.

## Final verdict

* **Proposition 5:** verified.
* **Integer-support strengthening:** proved.
* **Residual supremum M:** computer-assisted certified, exact rational.
* **Requested numerical curve:** complete.
* **Competition relevance:** weak/vacuous for unrestricted signed rules.
* **Kerdock-line signed rules:** closed by the stronger exact line-universe theorem.
* **Recommended paper use:** include as a scoped stability lemma, not as a closure of signed cubature.
