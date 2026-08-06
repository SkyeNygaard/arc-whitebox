# M153 exact observation-map and gauge audit

## Frozen runtime feature map

The reproduced T4 feature vector is

\[
\Phi(c_{17},p_2,p_4,p_{32},p_{128},\ldots)
\]

with nine components:

1. `cos(c17,p2)`;
2. `cos(c17,p4)`;
3. `cos(p2,p4)`;
4. `||p2||/||c17||`;
5. `||p4||/||c17||`;
6. minimum successive nested cosine;
7. maximum leave-one-basis angle sine;
8. `cos(p32,p128)`;
9. `||p32||/||p128||`.

The signed prediction targets are correction cosines or signed scalar coefficients relative to the scored error.

## Exact representation symmetry

Under simultaneous reversal of every represented candidate trajectory,

\[
(c_{17},p_2,p_4,p_{32},p_{128},\ldots)
\mapsto
-(c_{17},p_2,p_4,p_{32},p_{128},\ldots),
\]

all nine features are unchanged. A signed target such as

\[
y=\frac{\langle c_{17},e\rangle}{\|c_{17}\|\|e\|}
\]

changes sign when the candidate orientation reverses while the scored error is fixed.

Therefore the feature map factors through a quotient that removes the global candidate orientation. On an orientation-closed class, no deterministic function of these features can uniformly recover an orientation-dependent target; the two-point squared-error risk is at least the squared target magnitude.

## Important qualification

This is a theorem about the **representation**, not yet the WHestBench instance distribution. A physically generated candidate may carry a canonical orientation through its construction, and the actual network ensemble need not contain the reversed instance with equal law.

Consequently:

- M153 empirically closes the frozen feature/model dictionary;
- T46 explains a structural weakness of that feature map;
- neither result proves that all legal trajectory information lacks phase.

## Feature classification

| Feature type | Examples | Global sign behavior | Phase role |
|---|---|---|---|
| Norm | `||p4||/||c17||` | even | capacity only |
| Pairwise Gram/cosine | `cos(c17,p2)` | even under joint reversal | relative geometry only |
| Angle/disagreement magnitude | nested cosine, LOO angle | even | stability only |
| Canonically signed contraction | absent | odd | potential phase anchor |

## Required constructive change

A future feature map must include at least one legally available orientation-odd quantity with a deterministic sign convention. Candidate examples:

- contraction with a canonically signed final-layer adjoint vector;
- coefficients in a downstream singular basis whose signs are fixed by a deterministic rule;
- signed preactivation-margin or crossing contractions;
- a network-derived reference direction oriented by a fixed lexicographic or positive-pivot convention.

The odd feature must be tested for sign stability under floating-point ties and basis degeneracy. An arbitrary eigensolver sign is not a legal physical anchor.
