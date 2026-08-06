# Discrepancies

1. **“Whole Stein family is zero.”** Overstates V67. Exact only for bounded-degree polynomial fields and the stated bias-free one-hidden-layer homogeneous ReLU class.
2. **“Anything analytically integrable is low degree.”** Mathematically false; explicit Poisson-kernel counterexample supplied.
3. **“Degree-6/8/10 controls failed frozen.”** Only the selected degree-6+8 four-direction configuration received an independent frozen run. Other variants are exploratory.
4. **“Degree 6 measured at 13.93%.”** The figure belongs to an exact infinite-width kernel decomposition, not a finite-width measurement. The finite-width probe was variance-intractable.
5. **“Only live error is degree 6.”** The first live degree is 6; higher even degrees carry most of the remaining limiting-kernel error.
6. **V67 mixes proof and experiment.** Create a theorem row for exact blockwise cancellation and retain the `3.12e-17` value as a numerical reproduction.
7. **The shrinkage result is not frozen.** It reused the frozen-validation cohort as development and should remain labeled exploratory/development.
