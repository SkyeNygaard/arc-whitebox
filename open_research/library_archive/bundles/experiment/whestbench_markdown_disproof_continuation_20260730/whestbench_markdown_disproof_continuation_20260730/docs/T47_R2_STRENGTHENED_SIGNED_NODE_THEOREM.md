# T47-R2 — Strengthened Weighted Harmonic Rank Floor

**Status:** dual-stack computer-assisted certified.  
**Scope:** dimension 256, depth-32 limiting normalized ReLU kernel; static, network-independent, mass-one linear cubature; at most 66,048 arbitrary spherical nodes; arbitrary real weights.

## Statement

Every admissible rule satisfies

\[
R_{K_{32}}(Q) \ge 1.486477823691692199205348920799113356042073114405760804883283330382893220172192897042154673402266531E-7.
\]

Using the certified complete-Kerdock MSE upper endpoint,

\[
R_{K_{32}}(Q) \ge 0.6107992099573098167548734683367801325016165687480884574205433422658960056609730815417199665904831677\,R_{\mathrm{Kerdock}}.
\]

Therefore the maximum same-cost raw improvement is at most

\[
1.637199236177617740208281690867658785095971701858447488463638213863606096302494557408655906728062700.
\]

## Certificate

The exact decimal strings in `INDEPENDENT_WEIGHTED_RANK_L23.json` are interpreted as rational numbers. They define a degree-23 weighted harmonic feature kernel. Its squared Gegenbauer coefficients are positive. The directed lower kernel coefficients are obtained from a 384-bit MPFR interval Taylor jet of order 47. The minimum coefficient ratio binds at degree 8. The exact rank/trace defect is then multiplied by this ratio.

## Trust base

- direct-C MPFR directed arithmetic;
- handwritten exact rational Gegenbauer recurrence;
- independent SymPy/exact-rational reproduction;
- exact integer harmonic dimensions and rank selection;
- fixed rational weights, with no dependency on numerical optimizer optimality.

The numerical optimizer was used only to discover weights. Feasibility and theorem value are rechecked exactly.

## Limits

This theorem does not cover realized finite-width kernels, adaptive or network-dependent rules, nonlinear estimators, or free-total-mass rules. It is a global signed floor, not signed near-optimality.
