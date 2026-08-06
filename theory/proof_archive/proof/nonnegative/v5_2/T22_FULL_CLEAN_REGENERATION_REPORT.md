# T22 v5.1 full clean-regeneration report

**Status:** Passed locally.

The canonical v5.1 proof archive was extracted into a clean working tree. Generated theorem outputs and deterministic curvature outputs were removed. The base proof artifacts and all 1,079 source curvature intervals were regenerated in bounded deterministic chunks, producing the complete 1,421-leaf certified partition. All downstream theorem artifacts were then reassembled.

Final checks:

- `verify_theorem_package.py`: passed;
- certified subintervals: `1,421`;
- global pointwise upper margin: `-1.0045862406584556044e-13`;
- kernel-mean interval width: `2.288700332700929e-22`;
- one-sided theorem logic: verified;
- `verify_manifest.py`: `manifest verified: 59 files`.

This closes the previously incomplete local clean-regeneration gate. A separate direct-C GMP/MPFR audit supplies implementation diversity for the theorem-critical numerical path. Public release still requires an externally published digest and named human sign-off.
