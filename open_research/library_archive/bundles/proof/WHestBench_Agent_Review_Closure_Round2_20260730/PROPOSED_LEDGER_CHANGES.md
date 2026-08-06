# Proposed ledger changes — round 2

## Add theorem rows

### T38 — finite-width Kerdock-line theorem

- **Status:** PROVED UNDER EXPLICIT FINITE-WIDTH MODEL
- **Claim:** For a nondegenerate finite-width, finite-depth ReLU network with independent standard-Gaussian first-layer rows and later randomness independent of that layer, arbitrary real mass-one weighting and deletion in the fixed symmetrized real-MUB/Kerdock line universe are optimized by complete bases plus at most one partial basis.
- **Exclusions:** arbitrary nodes, network-dependent support/weights, nonlinear aggregation, arbitrary signed-node cubature.
- **Source:** `OPEN_QUESTIONS_PROGRESS_20260730.md`; `T38_FINITE_WIDTH_KERDOCK_LINE_THEOREM.md`.

### T39 — invariant-information projection

- **Status:** PROVED; APPLICATION CONDITIONAL
- **Claim:** Group-invariant runtime information recovers only the invariant component of an equivariant error.
- **Required qualification:** any WHestBench zero-value corollary must separately prove that the actual observation map and conditional law have the proposed symmetry.

### T40 — equivariant residual spectral multiplier

- **Status:** PROVED UNDER EXPLICIT OPERATOR MODEL
- **Claim:** For a deterministic bounded rotation-equivariant linear surrogate, `q_l^res=|1-tau_l|^2 q_l`.
- **Use:** residual-kernel recertification recipe, not automatic persistence of Kerdock optimality.

## Amend existing claim rows

1. Expand T27's scope from infinite-width only to the exact finite-width model supplied by T38.
2. Keep arbitrary-node T22 explicitly infinite-width.
3. Mark the Gaussian Bayes/no-adaptation cascade route INVALIDATED for the actual post-ReLU output.
4. Replace universal observability-gap language by the T33/T39 conditional-information framework.
5. Keep T29 as fixed-support linear optimality, not all-algorithm Bayes optimality.

## Correct stale statuses

- C58 → `Superseded / closed by C67`.
- C66 → `Superseded / closed by C67`.
- A06 → `Superseded by A42/A43`.
- A19 and A30 → `Closed/superseded by A39`.
- V40 → `Superseded by V59`.
- M86 → `Tested continuation closed; interface retained`.
- M95 → `Implementation retained; deployable source path closed`.
- M73, V73, V74, M87, M94, ARCI-4 → oracle/scientific mechanism or infrastructure, not live branch.
- A34/A36/A37 → adopted infrastructure/governance.
- C55 → operationally closed/inconclusive under the current hard research stop.

## Genuine live/open items

Retain active status only for:

- A42/A43/A47/A48 official external measurement;
- A46 local raw-data reproduction block;
- T34 arbitrary-node finite-width theorem;
- release/human-review actions recorded in `Genuine Open Questions`.
