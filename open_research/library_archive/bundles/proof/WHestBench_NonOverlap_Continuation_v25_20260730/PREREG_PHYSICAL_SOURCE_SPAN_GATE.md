# Preregistration — physical source-span gate for the frozen CLAF design

## Frozen primary design

- exposed networks only;
- existing 64/65 complete-basis split;
- start depth `s=9`;
- `M=64` rays per fold;
- existing even-deviation plane selector;
- unchanged source recurrence `c_s(x)`;
- final-output metric `G=W31^T W31`;
- no coefficient learning and no protected cohort.

## Primary metric

For the union of the two fold source matrices `C_AB`, compute

\[
r_{\mathrm{span}}
=\frac{\min_\alpha\|W31(d_s-C_{AB}\alpha)\|^2}
{\|W31d_s\|^2}.
\]

Use a rank-revealing eigendecomposition after whitening by `G`. Report numerical tolerances and the complete singular spectrum.

## Primary pass gate

\[
r_{\mathrm{span}}\le0.00502
\]

with no required group above the gate. The threshold comes from the frozen CLAF complete-score arithmetic.

## Secondary diagnostics

- fold A and fold B span residuals separately;
- same-basis, cross-basis, random-plane, and rank-one controls;
- 32 and 128 rays only after the 64-ray primary is evaluated;
- physical effective rank needed for 90%, 99%, and 99.5% capture;
- conditioning and coefficient norm;
- source defect norm and full exact-correction headroom.

## Decision rule

- **Fail:** stop CLAF entirely if the union-span oracle misses the gate.
- **Pass with poor conditioning:** continue only after a coefficient-stability theorem or bounded rule.
- **Pass with margin:** implement coefficient-one fan and exact replay; do not fit coefficients before that test.
