# Full independent GMP/MPFR replay report

## Result

**PASS.** Every theorem-critical numerical component of the scoped T22 theorem now has a second directed-rounding implementation.

## Curvature and pointwise minorant

- Existing certified subintervals checked: **1,421 / 1,421**.
- Initial mesh independently processed: **1,079 intervals**.
- Independent splits: **342**.
- Independently accepted subintervals: **1,421**.
- Maximum split depth: **4**.
- Minimum curvature-sign margin: `3.7863105650675452583865e-9`.
- Independent global bound:
  `max(h-K32) <= -1.00458624065845560440312456502465923923942077e-13`.

The independent engine also verifies all five critical boxes, four inflection boxes, two endpoint boxes, and two tails.

## Spherical mean

The independent MPFR Taylor-jet implementation obtains an interval nested inside the original Decimal interval:

- lower: `0.9747299895417147123122580852641911964220890140486520806041407254213009084865296608546785269021549126165336351382698212387`
- upper: `0.9747299895417147123124869552974612893380014764279519548528878897899822492889587914393646111541331672659727213065310068183`
- width: `2.288700332700929159124623792998742487...e-22`.

## Final ratio

GMP exact rational arithmetic plus MPFR produces:

- Kerdock MSE upper: `2.43366035754300522760946650266976459148e-7`;
- certified optimum MSE lower: `2.43309185344094125696198609817990868714e-7`;
- ratio upper: `1.00023365501029481377066020018598171905...`;
- relative-excess upper: `0.023365501029481377066020018598171905...%`;
- additive-suboptimality upper: `5.68504102061681947147703560696779719e-11`.

These are nested inside the published conservative enclosures.

## Software diversity

- GCC 14.2 and Clang 17 produced byte-identical JSON results for all five primary outputs.
- AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer completed without findings and produced byte-identical outputs.
- Exact-rational coverage verification establishes no gaps or overlaps over `[-1,1]` and verifies every certified interval is a legal dyadic descendant of its source interval.

## Claim consequence

The prior qualification “partial implementation-diverse interval replication” can be removed. The correct claim is now **complete independent second-engine reproduction of every theorem-critical numerical component**, within the shared mathematical derivation and exact witness.
