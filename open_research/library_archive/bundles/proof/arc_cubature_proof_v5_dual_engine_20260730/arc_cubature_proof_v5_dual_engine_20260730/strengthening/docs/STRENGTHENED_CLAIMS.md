# Strongest defensible claims after the dual-engine strengthening round

## Claim A — scoped numerical theorem

**Status: COMPUTER-ASSISTED CERTIFIED, DUAL-ENGINE REPRODUCED.**

For the explicit dimension-256, depth-32 infinite-width normalized ReLU kernel, the 66,048-point antipodal real-MUB rule has ensemble kernel discrepancy at most `1.0002336550102949` times the infimum over network-independent nonnegative-weight probability cubature rules with at most 66,048 support points. Equivalently, its true relative excess is in `[0, 0.02336550102949%]`.

The statement remains one-sided and does not prove strict suboptimality.

## Claim B — evaluated point-set incidence

**Status: COMPUTER-ASSISTED CERTIFIED by explicit finite construction.**

An independent integer implementation constructs 128 signed-Walsh bases plus the coordinate basis, checks all 8,128 nontrivial basis pairs, certifies exact normalized cross-inner-product magnitude `1/16`, derives multiplicities `1,1,510,32768,32768`, and exactly matches the archived production chirp asset.

## Claim C — complete arithmetic implementation diversity

**Status: COMPLETE SECOND-ENGINE REPRODUCTION OF THE NUMERICAL CERTIFICATE.**

A separate direct-C implementation using GMP exact rationals and MPFR directed rounding independently:

- verifies all 1,421 shipped curvature subintervals;
- regenerates the 1,079-to-1,421 subdivision tree with the same 342 splits and maximum depth 4;
- verifies the full critical/inflection/endpoint/tail sign chain and global minorant;
- recomputes the spherical kernel mean by interval Taylor jets;
- recomputes the Kerdock energy, exact Delsarte bound, MSE lower bound, additive bound, and final ratio.

It obtains ratio upper bound `1.00023365501029481377066020018598...`, nested inside the published conservative enclosure.

GCC 14.2 and Clang 17 produce byte-identical machine-readable outputs. AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer report no findings.

## Claim D — release reproducibility

**Status: RELEASE-HARDENED.**

The all-file release manifest covers the original proof package, all 23 curvature chunks, construction evidence, both arithmetic implementations, verification scripts, and dual-engine outputs. The release archive has a separately distributed SHA-256 digest.

## Necessary qualification

This is not proof-assistant formalization. The two numerical implementations share the mathematical derivation, exact rational auxiliary witness, and candidate interval endpoints. The second implementation treats those endpoints as untrusted hints and re-certifies every required inequality and sign.
