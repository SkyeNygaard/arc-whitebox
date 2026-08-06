# WHestBench Prompt 2 — finite-width reconciliation, strongest certificate, and closure decision

**Date:** 2026-07-30  
**Architecture:** input dimension 256, width 256, depth 32, bias-free He-scaled Gaussian matrices, ReLU after every matrix multiplication including the output layer  
**Rule class:** static, realized-network-independent, mass-one linear cubature with at most 66,048 arbitrary spherical nodes and arbitrary real signed weights  
**Risk:** exact finite-width ensemble MSE for one output coordinate; output-coordinate averaging is equivalent by exchangeability  
**Status:** strongest result below is an exact-rational, directed lower certificate under the declared transition-row proof components; clean-room independent implementation and named human proof review remain publication gates

---

## 1. Executive conclusion

### What had already been obtained elsewhere in the Library

Several important ingredients were already complete:

1. **General finite-width Hermite/noise-stability representation.** Other work had proved that a finite-width Gaussian-first-layer network kernel has a nonnegative power-series expansion
   \[
   K_m(t)=\sum_{n\ge 0}a_n t^n,\qquad a_n\ge 0.
   \]

2. **Exact finite-width fixed-MUB/Kerdock-line theorem.** The optimal allocation of arbitrary real mass-one weights inside a fixed union of mutually unbiased bases had already been derived. This is not an arbitrary-node theorem.

3. **Two-input empirical-Gram Markov representation and weak arbitrary-node floor.** Agent 2 had represented the two-input process as a Markov recursion on the empirical \(2\times 2\) Gram state and certified an architecture-specific floor of
   \[
   1.1641685686823468622\times 10^{-16}.
   \]

4. **Radial-chain continuation.** Agent 2 later improved the exact arbitrary-node finite-width floor to
   \[
   5.9401184424618019005\times10^{-10}.
   \]

5. **Degree-cutoff transfer frontier and moment-dual formulation.** The Library already identified degrees 22, 28, 62, 84 and 128 as retaining roughly 50%, 60%, 80%, 85% and 90% of the limiting signed-floor structure, and formulated the finite-width transfer problem as a nonnegative moment dual.

### What was not already complete elsewhere

The Library search found no independent copy of the following results:

- the exact Markov chain on **Hermite tensor degree**, rather than only on the random empirical Gram state;
- the formula
  \[
  K_{256}(t)=\sum_{k\ge0}(C^{32})_{1k}t^k
  \]
  for a row-stochastic tensor-degree transition operator \(C\);
- the integer-partition and tensor-output equality-class decomposition of transition energies;
- a collision-class excursion-and-return computation through 29 degree states;
- an arbitrary-node finite-width signed floor at the \(10^{-8}\) scale.

The strongest frozen result from this continuation is

\[
\boxed{
R_{\widetilde K_{256}}(Q)
\ge
1.92502416170783524100205877712113374289784088032
\times 10^{-8}.
}
\]

The selected polynomial subkernel retains at least

\[
0.943267864486136545527246554492391612431414354
\]

of total normalized kernel mass. Its exact complete-Kerdock risk is at most

\[
2.60304331364814790777553237099600786687417816399
\times10^{-8},
\]

so the lower certificate is within a factor

\[
1.3522133204493341765433996441179423998151
\]

of exhausting the declared component.

This is the strongest actual-width arbitrary-node signed/static floor found in the Library audit. It is approximately:

- **32.407×** Agent 2's radial-chain floor;
- **4.553×** the prior exact eight-state tensor-degree certificate;
- **14,048×** the completed first-layer full-chaos certificate;
- **165 million×** the first Agent 2 architecture-specific floor.

### Bottom-line decision

The immediate Prompt 2 goal is solved: there is a nontrivial, architecture-specific, arbitrary-node finite-width signed/static certificate, and the exact finite-width kernel has a tractable positive state-space representation.

The competition-scale goal is not solved. Using the limiting Kerdock MSE only as a common reference scale, the new floor is approximately **7.91%** of that reference. The floor needed to exclude a same-cost \(4.34\times\) gain is approximately

\[
5.6075123444700461\times10^{-8},
\]

which is **2.913×** above the new certificate. More decisively, the entire selected degree-28 component has Kerdock risk only \(2.6030\times10^{-8}\), still **2.154×** below that threshold. Therefore no comparison-weight optimization using only this extracted component can close the competitive gap.

The unresolved mathematical object is the high-degree excursion, persistence and return spectrum, especially the part needed by a degree-62 moment dual.

---

## 2. Reconciliation with prior Library work

| Prior result | Status before this continuation | Relationship to the new work |
|---|---|---|
| Nonnegative finite-width Hermite expansion | Proved | Necessary representation-level precursor, but does not compute the deep coefficient spectrum. |
| Exact finite-width fixed-MUB allocation theorem | Proved | Solves arbitrary weighting/deletion only within the fixed Kerdock/MUB line universe; does not compare arbitrary nodes. |
| Empirical \(2\times2\) Gram-state Markov recursion | Proved | Describes the exact random two-input process. The new tensor-degree chain is a deterministic spectral transition operator derived from orthogonal Hermite projections. |
| Agent 2 degree-six floor \(1.1642\times10^{-16}\) | Proved | Strictly superseded numerically for the same broad rule class. |
| Agent 2 radial-chain floor \(5.9401\times10^{-10}\) | Proved | Strongest pre-existing actual-width arbitrary-node floor; new result is 32.407× larger. |
| Finite-width transfer frontier | Proved as a conditional program | Correctly identified degree 62 as the first target retaining 80% of limiting signed-floor structure. Still requires actual-width coefficient/moment input. |
| Coupled 6,000-simulation fixed-MUB sanity check | Complete empirical diagnostic | Checks association signs only. It neither proves nor numerically completes arbitrary-node finite-width near-optimality. |

No prior Library artifact located by exact-number, theorem-name, semantic and filename searches contained the 29-state tensor-degree certificate or a stronger actual-width arbitrary-node floor.

---

## 3. Exact tensor-degree Markov theorem

For \(n\ge1\), define the normalized tensor feature

\[
F_n(x)=\frac{x^{\otimes n}}{\|x\|^{n-1}},
\qquad
F_0(x)=\|x\|.
\]

Let \(G\sim N(0,I_{256})\), and let \(H_k\) be total multivariate Gaussian Hermite degree \(k\). Define

\[
C_{nk}
=
\left\|
\operatorname{Proj}_{H_k}
\frac{F_n(\sqrt2\,G_+)}{\sqrt{256}}
\right\|_{L^2}^2.
\]

Then

\[
C_{nk}\ge0,
\qquad
\sum_{k\ge0}C_{nk}=1.
\]

Thus \(C\) is a row-stochastic Markov operator on nonnegative integer tensor degree. If \(U_{r,n}\) is the covariance kernel obtained by applying \(r\) finite-width ReLU layers to tensor feature \(n\), then

\[
U_{r+1,n}=\sum_{k\ge0}C_{nk}U_{r,k},
\qquad
U_{0,k}(t)=t^k.
\]

Because a scalar output coordinate is degree one,

\[
\boxed{
K_{256}(t)=U_{32,1}(t)=\sum_{k\ge0}(C^{32})_{1k}t^k.
}
\]

This identity includes finite-width norm fluctuations, empirical correlations, all 32 matrices and the post-ReLU output exactly. It is not a Gaussian-process approximation or a qualitative convergence statement.

### Exact supplementary identities

The continuation also derived two useful structural checks:

1. **Parity balance.** For every state \(n\ge1\), the transition row has equal even and odd mass.
2. **Critical degree drift.** The expected degree after one transition obeys an exact affine identity, and starting from state one the expected degree remains one through depth.

These identities constrain and validate the generated transition rows, but they do not by themselves lower-bound the harmonics relevant to cubature.

---

## 4. How the 29-state lower kernel is constructed

Let \(\underline C\) be a finite nonnegative matrix with

\[
0\le \underline C_{nk}\le C_{nk}
\]

entrywise. Since all entries are nonnegative,

\[
(\underline C^{32})_{1k}\le(C^{32})_{1k}
\]

coefficientwise. Therefore

\[
\underline K(t)
=
\sum_{k=0}^{28}(\underline C^{32})_{1k}t^k
\preceq K_{256}(t)
\]

is a valid positive-semidefinite subkernel.

The current \(29\times29\) matrix combines:

- complete low-state transition rows where available;
- exact Hermite energies for state zero;
- exact scalar-ReLU coefficients for state one;
- exact unrestricted tensor-row records for selected state/degree pairs;
- disjoint tensor-output equality classes:
  - all singleton output indices;
  - one pair;
  - one triple;
  - two pairs;
  - one block of four;
  - triple plus pair;
  - three pairs;
- support-restricted collision-free lower components when they exceed the other certified lower entries.

Alternative lower bounds for the same matrix entry are combined by taking their maximum. Orthogonal equality classes are combined by summation. Every row sum is checked to be at most one.

### Integer partitions replace Bell enumeration

For output tensor rank \(r\), equality patterns depend only on an integer partition

\[
\pi=(p_1,\dots,p_b)\vdash r.
\]

Its multiplicity is

\[
N(\pi)
=
\frac{r!}
{\prod_j p_j!\prod_s m_s(\pi)!}.
\]

This reduces the outer combinatorics from set-partition Bell growth to integer-partition growth. The partition implementation reproduces the earlier rank-three through rank-six endpoint calculations exactly.

### Directed propagation

The verifier starts at state one and performs 32 exact rational matrix multiplications. After every layer, each nonnegative state mass is rounded downward at 75 decimal places. Consequently all propagated masses remain rigorous lower bounds despite controlled denominator truncation.

The final selected mass interval is

\[
[0.943267864486136545527246554492391612431414354,
 0.943267864486136545527246554492391612431414355].
\]

---

## 5. Weighted-rank certificate

The degree-28 monomial subkernel is converted exactly to normalized Gegenbauer coefficients. Positivity of the triangular monomial-to-Gegenbauer conversion is checked coefficientwise.

A rationalized degree-14 harmonic weight vector \(w\) defines

\[
L_w(t)=\sum_{\ell=0}^{14}w_\ell h_\ell G_\ell(t),
\]

where \(h_\ell\) is harmonic multiplicity. The verifier expands \(L_w^2\) exactly and finds a positive rational \(\gamma\) such that

\[
\underline K_\ell\ge\gamma[L_w^2]_\ell
\]

at every nonconstant degree through 28.

The active degrees are

\[
6,7,8,17,10,13,9,12,23,14,15,11,16,25,27.
\]

The rank/trace theorem then gives

\[
R_{\widetilde K_{256}}(Q)
\ge
\gamma F_{66,048}(w)
\]

for every declared rule, including arbitrary signed weights and rank-deficient moment matrices.

The verifier reports a strictly positive minimum domination margin, so no active degree is relying on equality at floating-point precision.

---

## 6. Hostile interpretation

### What the result proves

For the exact stated ensemble architecture, every static rule

\[
Q=\sum_{i=1}^m q_i\delta_{x_i},
\qquad
m\le66,048,
\qquad
\sum_iq_i=1,
\]

with arbitrary nodes \(x_i\in S^{255}\) and arbitrary real weights satisfies the certified normalized MSE floor.

Randomized rules independent of the realized network are included by conditioning on their external randomness.

### What it does not prove

The theorem does not cover:

- nodes or weights adapted to the realized network;
- nonlinear processing of evaluated network outputs;
- free-total-mass rules;
- candidate-dependent residual transformations without a new kernel certificate;
- per-realized-network lower bounds;
- unrestricted white-box algorithms;
- an exact ratio to the actual finite-width Kerdock MSE;
- a finite-width arbitrary-node Delsarte near-optimality theorem.

### Remaining trust boundary

The exact final verifier uses rational arithmetic and directed downward rounding, but its transition entries come from generated row records. The proof trust base therefore still includes:

- the multivariate Hermite tensor projection formula;
- the integer-partition multiplicity formula;
- the radial Gamma-ratio reductions;
- the claim that the enumerated tensor-output equality classes are disjoint orthogonal components;
- correct generation and serialization of every row record used by the assembler.

A clean-room Arb/MPFR or exact-C implementation and named human review remain necessary before publication-level wording.

---

## 7. Are the experiments complete?

### Complete enough to freeze now

The following are complete and reproducible in the present package:

- the exact 29-state, degree-28 lower-matrix assembly from available row records;
- 32-step downward-rational propagation;
- exact monomial-to-Gegenbauer conversion;
- exact rational weighted-rank verification;
- exact selected-component Kerdock-risk calculation;
- degree-28 comparison optimization followed by rational freezing;
- a small completely-positive-mixture comparison search, which did not beat the frozen 29-state rank-one certificate and confirms that comparison optimization is secondary.

### Not complete elsewhere or here

The following remain open:

- complete unrestricted transition rows for all states and degrees through 62;
- rigorous aggregation of all high-degree excursions and returns;
- a degree-62 finite-width moment dual with an all-degree tail proof;
- a sharp finite-width Kerdock-risk denominator;
- a finite-width arbitrary-node Delsarte certificate;
- independent implementation review.

At the report freeze, additional unrestricted row computations for states 2–5 were still running. States 2 and 3, degree 14 completed and were incorporated before the final verifier run. The unfinished rows are optional monotone strengthenings and are not required for the theorem reported here. They must not be treated as completed evidence unless subsequently frozen with new hashes and a rerun certificate.

---

## 8. Competition relevance and stop/go decision

Let the limiting Kerdock MSE \(2.4336603575\times10^{-7}\) serve only as a common scale. The current floor is

\[
0.0790999514692072
\]

of that reference.

A \(4.34\times\) same-cost barrier requires a floor near

\[
5.60751234447\times10^{-8}.
\]

The present floor is smaller by a factor \(2.913\). The complete selected component is smaller by a factor \(2.154\). Therefore:

> More comparison-weight tuning, additional squared-kernel mixtures, or minor completion of the existing degree-28 component cannot close the competition gap.

Using Agent 2's rigorous all-width Kerdock upper bound gives only

\[
\frac{R(Q)}{R(Q_K)}\ge0.0015117478,
\]

which is far too weak for competitive closure. A sharp finite-width denominator is still needed for a meaningful actual-width ratio.

### Continue only on the following path

The next mathematically justified branch is:

1. target the degree-62 dual directly;
2. compute or bound only transition combinations active in that dual;
3. use collision-deficit classes plus aggregate tail constraints rather than completing arbitrary rows uniformly;
4. derive a rigorous excursion-return bound for paths leaving the finite cutoff and returning to active degrees;
5. separately upper-bound actual finite-width Kerdock risk.

### Stop conditions

Stop this finite-width proof branch if either:

- the degree-62 moment dual remains below the competition threshold even under an optimistic rigorous tail ceiling; or
- the selected component's exact Kerdock ceiling stays below the threshold after the dominant dual-active excursion classes are included.

Do not spend further material effort on:

- first-layer-only projections;
- radial-chain profile optimization;
- isolated low-state additions with sub-percent effect;
- comparison-weight polishing for the current degree-28 component.

---

## 9. Canonical theorem record proposed for the ledger

**Provisional title:** Exact 29-state finite-width tensor-degree signed/static floor.

**Evidence level:** exact analytic reduction plus exact-rational computer-assisted certificate; generated-row proof trust base; independent implementation pending.

**Statement:** For the width-256, depth-32, bias-free He-ReLU ensemble with a post-ReLU output, every static, realized-network-independent, mass-one linear cubature rule using at most 66,048 arbitrary spherical nodes and arbitrary real signed weights satisfies

\[
R_{\widetilde K_{256}}(Q)
\ge
1.92502416170783524100205877712113374289784088032\times10^{-8}.
\]

**Certificate:** exact Hermite tensor-degree Markov theorem; 29-state entrywise lower transition matrix; collision/equality-class subcomponents; 32-step exact rational propagation with 75-digit downward rounding; exact degree-28 Gegenbauer conversion; rational degree-14 weighted-rank comparison.

**Component ceiling:** selected-component complete-Kerdock risk at most

\[
2.60304331364814790777553237099600786687417816399\times10^{-8}.
\]

**Verdict:** strongest recorded actual-width arbitrary-node signed/static floor; major strengthening, but still competition-insufficient. The degree-28 component is locally saturated. Prioritize a degree-62 excursion-return/moment-dual theorem.

**Scope warning:** ensemble only; static and network-independent; mass one; no adaptive support, nonlinear estimator, realized-network theorem or exact finite-width Kerdock ratio.

---

## 10. Reproduction

Extract the reproducibility bundle into `/mnt/data`, then run:

```bash
cd /mnt/data
python verify_prompt2_markov_29state_degree28.py
```

Expected headline output:

```text
status: PASS
normalized floor lower endpoint:
1.92502416170783524100205877712113374289784088032e-8
selected component Kerdock upper endpoint:
2.60304331364814790777553237099600786687417816400e-8
selected kernel mass lower endpoint:
0.943267864486136545527246554492391612431414354
```

Core artifacts:

- `WHestBench_Prompt2_FiniteWidth_Reconciliation_and_Closure_20260730.md`
- `verify_prompt2_markov_29state_degree28.py`
- `prompt2_markov_29state_degree28_certificate.json`
- `explore_markov_29state.py`
- `explore_markov_23state.py`
- `prompt2_tensor_partition_core.py`
- `prompt2_full_hermite_core.py`
- generated tensor/equality-class row records included in the reproducibility bundle

The certificate JSON records the exact rational floor, coefficient lower bounds, comparison weights, active degrees, transition source counts, selected mass and limitations.

---

## 11. Final assessment

The strongest results were **not already obtained elsewhere** in the project. Prior agents had established the broad representation, the fixed-MUB theorem, a weak exact floor, a much better radial floor and the correct moment-dual program. This continuation supplies the missing deep finite-width tensor-degree state equation and raises the arbitrary-node signed/static floor to the \(10^{-8}\) scale.

The experiments needed to support this frozen intermediate theorem are complete. The experiments needed for a competition-scale or full finite-width near-optimality conclusion are not complete.

The honest final state is:

\[
\boxed{
\text{Prompt 2 immediate goal: solved.}
}
\]

\[
\boxed{
\text{Arbitrary-node finite-width competition closure: still open.}
}
\]

\[
\boxed{
\text{Best next path: degree-62 excursion-return moment dual, not local tuning.}
}
\]
