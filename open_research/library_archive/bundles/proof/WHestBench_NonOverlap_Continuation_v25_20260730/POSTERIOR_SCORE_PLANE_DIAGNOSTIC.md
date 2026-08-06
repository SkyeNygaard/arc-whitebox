# Posterior-score plane diagnostic

**Status:** deterministic numerical diagnostic in the limiting final-preactivation Gaussian model. It is not a closure theorem for the richer finite-width CLAF late-source field.

## Calculation

Using the exact 129 real-MUB incidence, the depth-31 covariance kernel, and the complete fixed-support posterior:

- construct the one-dimensional global association score direction `r0`;
- grant exact preactivation observations at uniformly spaced rays on a two-dimensional plane;
- compute the exact Gaussian posterior covariance of those new observations after conditioning on all 66,048 Kerdock values;
- report the Schur-complement fraction of `||r0||^2` captured.

## Results

| Plane | Rays | Captured global-score energy |
|---|---:|---:|
| same Kerdock basis | 16 | 5.974e-5 |
| same Kerdock basis | 32 | 6.355e-5 |
| same Kerdock basis | 64 | 6.396e-5 |
| same Kerdock basis | 128 | 6.399e-5 |
| cross Kerdock bases | 16 | 5.989e-5 |
| cross Kerdock bases | 32 | 6.355e-5 |
| cross Kerdock bases | 64 | 6.395e-5 |
| cross Kerdock bases | 128 | 6.398e-5 |

The result is stable to posterior-covariance eigenvalue clipping over `1e-12` through `1e-7`; the 64-ray same-basis capture remains approximately `6.40e-5`.

## Meaning

The score direction is extraordinarily diffuse relative to a two-plane point-evaluation transcript. Increasing rays from 32 to 128 is essentially saturated. This strongly disfavors the current plane choice in the fixed-preactivation posterior channel.

## Scope guard

CLAF evaluates a transported late-source potential, not merely the final-preactivation Gaussian row. Therefore this diagnostic does not prove the physical source-span residual for CLAF. T80 supplies the correct decisive test for that richer field.
