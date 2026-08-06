# Oracle source coefficients: why constant-first is mandatory

The five-source global coefficient is

\[
(-0.5591,\ 0.3145,\ 0.1725,\ 0.2405,\ 0.2257).
\]

The feature policy does not behave like a small residual refinement. Across development, validation, and confirmation, its mean first coefficient is positive (`0.133`, `0.048`, and `0.057`), and it matches the global first-source sign in only 25–33% of cases. The other four source signs match in every case.

On confirmation, the feature rule is worse than the global rule in 8 of 12 cases and increases the unweighted mean case ratio by about `0.143`. Its pooled ratio is `0.872`, versus `0.778` for the global rule, with a much worse tail.

This identifies a concrete confound:

> the flexible model is relearning—and frequently reversing—the average source action rather than estimating a small instance-specific residual around it.

The next protocol must freeze the global coefficient `a0` and train only

\[
\delta a(X),\qquad a(X)=a_0+\lambda\delta a(X),\quad 0\le\lambda\le1,
\]

with `lambda` frozen by grouped development. Feature value must be scored against the matched global baseline. This does not guarantee success; it removes a major avoidable failure mode and makes abstention (`lambda=0`) explicit.
