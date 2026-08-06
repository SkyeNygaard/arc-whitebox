# Verified Open-Questions Addendum

## Finite-width fixed-support result

The new finite-width argument is accepted with a tightened statement.

Let `Y(x)=F_Z(Wx)`, where the rows of `W` are independent standard Gaussian vectors, `Z` is independent of `W`, and `F_Z` is square integrable and input-independent. Conditioning on `Z` and expanding `F_Z` in multivariate Hermite polynomials gives

\[
K_m(t)=\mathbb E\langle Y(x),Y(y)\rangle=\sum_{n\ge0}a_nt^n,\qquad a_n\ge0.
\]

After antipodal line symmetrization, only even coefficients remain. On a real-MUB universe the association values satisfy

\[
A-O=\sum_{r\ge1}a_{2r},\quad
O-C=-\sum_{r\ge1}a_{2r}d^{-r},
\]

and

\[
(A-O)+d(O-C)=\sum_{r\ge2}a_{2r}(1-d^{1-r}).
\]

Thus the exact T27 allocation theorem applies whenever the even component is nonconstant and has positive Hermite mass at some even degree at least four. For a standard nondegenerate finite ReLU network this is the intended condition; the paper should state the condition explicitly instead of hiding it behind informal “mild nondegeneracy.”

This proves finite-width fixed-support optimality. It does not certify arbitrary-node finite-width near-optimality or compute the absolute finite-width Kerdock MSE.

## Group-invariant information result

The invariant-information projection theorem is valid under a measure-preserving group action, equivariant error and invariant observation map. Its orientation-blind corollary requires conditional sign symmetry; the existing empirical failures are consistent with, but do not prove, that symmetry for WHestBench.

## Residual spectral recertification

For a deterministic bounded rotation-equivariant linear surrogate, each harmonic degree is multiplied by a scalar, and residual variance is multiplied by `|1-tau_l|^2`. This gives a correct recertification recipe. General network-dependent nonlinear or candidate-dependent surrogates remain outside the result.

## Validation performed in this pass

- The supplied verifier passed its exact algebra and orientation-blind symmetry checks.
- Its finite-width Markov simulation was treated only as a noisy sanity check.
- The Hermite/noise-stability argument was reviewed independently at the level of theorem assumptions and coefficient signs.

## Required wording

Use: “finite-width extension of the fixed-MUB-support theorem under explicit Gaussian-first-layer and nondegeneracy assumptions.”

Do not use: “finite-width Kerdock optimality,” without the fixed-support qualification.
