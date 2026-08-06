# Reproduction matrix

| Check | CPython Decimal/libmpdec | Pure `_pydecimal` | Direct C GMP/MPFR | Independent integers |
|---|---:|---:|---:|---:|
| Full 1,079-interval source mesh | clean regeneration | selected chunks | **full regeneration** | exact coverage |
| All 1,421 curvature subintervals | certified | selected chunks | **1,421/1,421 verified** | exact dyadic ancestry |
| Adaptive split tree | 342 splits, depth 4 | not run fully | **342 splits, depth 4** | exact coverage |
| Global sign diagram | certified | byte-identical component replay | **fully independently verified** | ordering/coverage |
| Pointwise minorant | certified | downstream replay | **independent strict bound** | n/a |
| Spherical kernel mean | certified | byte-identical | **independent interval Taylor jets** | exact moments |
| Kerdock incidence | multiplicity output | byte-identical | five-value energy replay | **explicit full construction** |
| Delsarte bound and final ratio | certified | scalar replay | **complete independent replay** | exact rational coefficients |
| Compiler diversity | n/a | n/a | **GCC/Clang byte-identical** | n/a |
| Sanitizers | n/a | n/a | **ASan/UBSan/LSan pass** | n/a |
| Full release integrity | original manifest | n/a | all-file manifest | separate archive digest |
