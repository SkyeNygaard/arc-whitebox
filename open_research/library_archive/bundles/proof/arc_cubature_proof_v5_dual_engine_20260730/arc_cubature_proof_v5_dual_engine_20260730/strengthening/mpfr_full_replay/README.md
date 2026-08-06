# Independent GMP/MPFR theorem replay

This directory supplies a second theorem-critical arithmetic implementation for T22/T23. It does not import the original Python interval modules.

It verifies and independently regenerates:

1. all 1,421 certified curvature subintervals;
2. the 1,079-to-1,421 adaptive subdivision tree, including all 342 splits;
3. the five critical boxes, four inflection boxes, endpoint boxes, tails, and global strict minorant;
4. the depth-32 spherical kernel mean using interval Taylor jets;
5. the five-value Kerdock energy, exact Delsarte bound, MSE bound, and final one-sided ratio.

The implementation uses GMP exact rationals and MPFR directed rounding. The source includes the minimal public MPFR ABI declarations needed because this execution environment supplied the runtime library but not the development header.

Run:

```bash
./run_mpfr_replay.sh
```

The script builds and runs with GCC and, when available, Clang, then requires byte-identical machine-readable outputs.

This is an independently implemented computer-assisted replay, not proof-assistant formalization. It shares the exact rational auxiliary witness and candidate interval endpoints with the original package; those endpoints are treated as untrusted hints and all required signs and inequalities are re-certified.
