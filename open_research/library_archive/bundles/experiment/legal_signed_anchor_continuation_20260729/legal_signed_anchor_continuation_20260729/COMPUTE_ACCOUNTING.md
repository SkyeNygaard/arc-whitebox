# Compute accounting

All counts are projections against the 175.62B full-129-basis baseline and require official subprocess FlopScope verification before any shipping claim.

| Component | Projected added compute | Notes |
|---|---:|---|
| Centered full-covariance transport | ~1.95B | 29 layers, two dense covariance products per layer |
| Affine surrogate composition | ~1.07B | 32 dense 256x256 compositions |
| 1,024 + 2,048 full-depth pilot rows | ~8.17B | baseline-proportional row estimate |
| q128 adjoint/source contractions | <1–2B | depends on reuse of gates and reductions |
| Cross-scale pilot package total | ~10.2–12.2B | plausibly below 14B; statistically rejected |
| Transport-only analytic candidate | ~2–4B | comfortably below cap; no FlopScope certificate |
