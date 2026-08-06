# Reproduction matrix

| Check | Original C decimal | Pure `_pydecimal` | Independent integers | Independent high precision |
|---|---:|---:|---:|---:|
| Full 1,079-interval mesh | byte-identical clean regeneration | not completed | n/a | numerical spot audit |
| Representative chunks 550, 605, 606 | yes | byte-identical | n/a | n/a |
| Assembled coverage certificate | yes | byte-identical from chunks | exact coverage checks | yes |
| Sign diagram | yes | byte-identical original audit | independently derived audit | yes |
| Spherical kernel mean | yes | byte-identical | exact moment fractions | independent quadrature audit |
| Kerdock incidence/multiplicities | yes | byte-identical multiplicity output | full explicit construction | n/a |
| Final one-sided ratio | yes | independent outward scalar replay | exact fractions | numerical agreement |
| Full release file integrity | original 32-entry manifest | n/a | strengthened all-file manifest | separate archive digest |
