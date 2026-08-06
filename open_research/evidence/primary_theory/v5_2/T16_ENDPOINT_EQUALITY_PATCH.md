# T16 endpoint-equality patch

**Status:** directed-decimal certified.

The interior Hermite remainder proves

\[
K_{32}(t)-h_*(t)>0
\]

for every noncontact point in `(-1,1)`, because `K32^(6)>0` and the squared contact polynomial is positive away from the three roots. Continuity alone gives only endpoint nonnegativity, so the old equality-only-at-contacts wording needed explicit endpoint separation.

Using the certified Gegenbauer coefficient intervals from `T16_PRIMAL_DUAL_CERTIFICATE.json` and the independent directed evaluation of `K32(-1)`, the bundled endpoint script proves

\[
K_{32}(1)-h_*(1)
\in
[0.0170218942683709807001391155978126223540679072640,
 0.0170218942683709807001391155978126223540679072641],
\]

and

\[
K_{32}(-1)-h_*(-1)
\in
[2.2051871290807434455869043041150906917944744889\times10^{-7},
 2.2051871290807434455869043041150906917944744890\times10^{-7}].
\]

Both are strictly positive. Therefore equality in the global minorant occurs exactly at the three interior Hermite contact nodes.

Regenerate with:

```bash
python certify_k32_mub_line_spectrum.py
python certify_t16_endpoints.py
```

The resulting machine-readable certificate is `T16_ENDPOINT_CERTIFICATE.json`.
