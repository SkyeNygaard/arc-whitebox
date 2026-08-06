# Agent 3 — Decision

## Decision

**VERIFIED AFTER SPECIFIED CORRECTIONS.**

T27 is a valid exact restricted theorem. It substantially strengthens the older rectangular-subset statement: within the fixed 33,024 antipodal Kerdock lines, no arbitrary deletion pattern, unequal weighting, signed weighting, negative basis total, or mixture of complete and partial bases can beat concentration into complete bases plus at most one partial basis.

## Claim classification

**PROVED UNDER AN EXPLICIT MODEL**

- infinite-width depth-32 normalized ReLU kernel;
- dimension 256;
- static/network-independent linear rules;
- fixed antipodal Kerdock-line universe;
- line weights summing to one.

The numerical constants were independently reproduced. Their displayed floating-point evaluation is not needed for the algebra beyond the signs `A-O>0`, `O-C<0`, and positivity at `r=256`.

## Paper-ready one-sentence claim

> For the dimension-256, depth-32 infinite-width ReLU kernel, among all static linear cubature rules supported on at most `P` symmetrized antipodal lines from the fixed 33,024-line Kerdock universe, with arbitrary real weights summing to one, the minimum ensemble MSE is attained by `floor(P/256)` complete orthonormal bases and at most one additional partial basis, with equal positive weights within each active basis and analytically determined positive basis masses.

## Prohibited extrapolation

Do not use T27 to claim unrestricted signed-weight optimality, arbitrary-node optimality, finite-width optimality, nonlinear-estimator optimality, or network-adaptive impossibility.
