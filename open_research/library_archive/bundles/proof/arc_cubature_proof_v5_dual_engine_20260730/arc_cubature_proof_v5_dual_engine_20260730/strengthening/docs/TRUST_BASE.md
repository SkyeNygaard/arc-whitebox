# Minimal trust base after dual-engine strengthening

## Mathematical inputs

- exact integer and rational arithmetic;
- the standard spherical-harmonic/Gegenbauer positive-definiteness argument;
- the written Delsarte deduction from the certified pointwise minorant;
- the explicit definition of the normalized depth-32 ReLU kernel.

## Independently certified finite inputs

- exact rational auxiliary polynomial coefficients;
- explicit 129-basis point-set incidence and production-asset match;
- exact coverage of the interval decomposition over `[-1,1]`.

## Numerical software diversity

The complete theorem-critical numerical chain is certified twice:

1. CPython `Decimal`/libmpdec with exact `Fraction` arithmetic;
2. independent direct-C GMP rationals plus MPFR directed rounding.

The MPFR implementation independently reconstructs the entire curvature subdivision, global sign chain, kernel mean, Delsarte energy calculation, and final one-sided ratio. GCC and Clang outputs are byte-identical, and sanitizer builds pass.

## Remaining trust

- correctness of CPython, GMP, MPFR, the C compilers, operating system, and hardware;
- correctness of the shared mathematical reduction and theorem prose;
- correctness of the standard harmonic-analysis facts used by the Delsarte argument;
- no Lean, Coq, Isabelle, or HOL formalization;
- no completed cross-operating-system reproduction matrix;
- no claim outside the explicit nonnegative, network-independent, infinite-width estimator class.
