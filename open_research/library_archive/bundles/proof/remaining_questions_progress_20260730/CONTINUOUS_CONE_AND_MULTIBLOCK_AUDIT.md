# Comparison-cone audit after overflow repair

## Discovered numerical bug

The first continuous-radius column generator formed derivative coefficients such as `c*e-b*f` directly. At high harmonic degree these products overflowed double precision, so interior stationary points were silently omitted. The earlier numerical claim that the continuous adjacent cone was saturated was not valid.

The repair independently rescales the numerator and denominator quadratics before forming their derivative. Positive independent rescaling does not change stationary points of their ratio.

## Corrected results

- released exact adjacent-grid certificate: `0.9370459569114724` of Kerdock upper;
- corrected continuous-adjacent numerical LP: `0.9370496015333069`;
- correct general contiguous multiblock numerical LP: `0.9370553397756372`.

The improvements over the released exact witness are respectively:

- continuous radii: `0.00036446` percentage point;
- richer multidegree profiles: `0.00093829` percentage point.

These two improvements are numerical discoveries, not certified theorem constants. The exact 93.7046% certificate remains the paper-level result.

## Consequence

Even after fixing the overflow and using the stronger general block-trace lemma, neither continuous radius optimization nor richer contiguous feature profiles explain a material part of the remaining 6.295% theorem gap. The next mathematical target is point-evaluation realizability, not further radius-grid engineering.
