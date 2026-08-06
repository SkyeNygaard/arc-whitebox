# Observability metric salvage — total capacity, transferred value, and fraction

**Status:** analytically proved bookkeeping convention.

Let `H_runtime` be a closed correction-information subspace and let `H_oracle` be a larger oracle subspace. For baseline error `e`, define

\[
V_{\rm runtime}=\|P_{H_{\rm runtime}}e\|_{L^2}^2,
\qquad
V_{\rm oracle}=\|P_{H_{\rm oracle}}e\|_{L^2}^2.
\]

Nestedness gives

\[
0\le V_{\rm runtime}\le V_{\rm oracle}.
\]

The always-defined quantities are:

- **oracle capacity:** `V_oracle`;
- **transferred value:** `V_runtime`;
- **unobserved value:**
  \[
  V_{\rm oracle}-V_{\rm runtime}\ge0.
  \]

Only when `V_oracle>0` define the transferred fraction

\[
F_{\rm transfer}
=\frac{V_{\rm runtime}}{V_{\rm oracle}}
\in[0,1].
\]

If `V_oracle=0`, nestedness forces `V_runtime=0`; report **zero oracle capacity** rather than assigning an arbitrary ratio. This separates two scientifically different findings:

1. `V_oracle=0`: the proposed correction class has no useful capacity;
2. `V_oracle>0` but a small transfer fraction: useful capacity exists but legal information does not recover it.

This three-number reporting convention avoids the undefined `0/0` edge case and preserves the capacity-versus-observability distinction.
