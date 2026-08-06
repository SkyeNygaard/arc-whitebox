# Agent 2 dual-engine strengthening report

## Objective

Remove the final implementation-monoculture qualification from T22/T23 without broadening theorem scope.

## Completed work

### Full second-engine curvature proof

A direct-C GMP/MPFR implementation independently verifies all 1,421 certified subintervals and regenerates the full adaptive tree from 1,079 source intervals. It independently makes 342 splits, accepts 1,421 leaves, and reaches maximum depth 4. The minimum curvature-sign margin is `3.7863105650675452583865e-9`.

### Independent global minorant

The MPFR verifier independently certifies five critical boxes, four inflection boxes, two endpoint boxes, and both tails. It obtains

`max(h-K32) <= -1.00458624065845560440312456502465923923942077e-13`.

### Independent spherical mean

A separate interval Taylor-jet implementation produces an enclosure nested inside the original Decimal certificate, with width `2.288700332700929159124623793e-22`.

### Independent energy and theorem assembly

GMP exact rational arithmetic and MPFR recompute the five-value Kerdock energy, exact Delsarte bound, MSE lower bound, additive suboptimality, and final ratio. The independently obtained ratio upper bound is

`1.00023365501029481377066020018598171905...`,

which is below the published conservative `1.0002336550102949`.

### Software-verification evidence

- GCC 14.2 and Clang 17 produce byte-identical primary JSON outputs.
- AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer complete without findings.
- An exact rational checker verifies all dyadic ancestry, interval coverage, and complete coverage of `[-1,1]`.
- A clean one-command rerun script compiles and executes both compiler paths.

## Strongest defensible conclusion

T22 is a scoped, reproducible, dual-engine computer-assisted theorem. The full theorem-critical numerical certificate is independently reproduced by two arithmetic stacks. T23 is correspondingly strengthened from clean-room reproducibility to complete implementation-diverse numerical reproduction.

## Remaining limitations

- shared mathematical derivation and exact auxiliary witness;
- no proof-assistant formalization;
- no cross-operating-system matrix yet;
- no extension to signed, adaptive, network-dependent, nonlinear, or finite-width estimators.
