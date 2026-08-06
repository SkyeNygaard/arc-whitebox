# Minimal trust base after strengthening

## Proof-critical mathematical inputs

- exact integer and rational arithmetic;
- directed decimal arithmetic for the full curvature mesh;
- the standard spherical-harmonic/Gegenbauer positive-definiteness argument;
- the written Delsarte deduction from the certified pointwise minorant.

## Construction input removed from the trust base

The existence and incidence of the 129 real MUBs is now established by an explicit finite certificate shipped with this release. The certificate does not import the production construction code.

## Software evidence

- the original full mesh and theorem regenerate byte-for-byte under CPython 3.13.5 C `decimal`;
- pure `_pydecimal` independently regenerates the kernel mean, selected curvature chunks, and downstream theorem assembly;
- exact Kerdock incidence uses a Python-integer implementation independent of the production code;
- a complete full-release manifest detects modification of every included chunk and source file.

## Remaining uneliminated trust

- correctness of both CPython decimal implementations and the interpreter/runtime;
- no full second-engine regeneration of all 1,079 curvature intervals;
- no proof-assistant formalization of the analytic deductions;
- no cross-OS or cross-Python CI evidence in this release.
