# Proposed canonical-ledger changes

## New row T29

| Field | Value |
|---|---|
| ID | T29 |
| Evidence level | Exact analytic identity with independent numerical check |
| Family | Blockwise ReLU-Stein annihilation |
| Experiment | Prove the average of `div phi-x^T phi` for `phi(x)=sum_j a_j ReLU(v_j^T x)` on every antipodal orthonormal-basis block after exact radialization |
| Result | Exact zero for every block; independent random 256D check had absolute block mean `1.15e-14` |
| Verdict | PROVED UNDER EXPLICIT MODEL; excludes biased, depth-2+, nonhomogeneous, and node-dependent fields |
| Primary source | Agent8 harmonic audit, Lemma 3; `verify_harmonic_claims.py` |
| Status | Closed |
| Evidence tier | Proof / analytic |
| Overlap cluster | Theory & proof / control variates |

## Revised V67 result

> Polynomial Stein fields are exactly annihilated when each component has degree at most 4. T29 proves exact blockwise annihilation for fixed bias-free one-hidden-layer ReLU fields. The archived numerical check had maximum block residual `3.12e-17`. This does not cover biased or depth-2+ neural Stein fields.

## Revised Paper Claims Matrix rows

### Named low-degree controls

- Claim: Named low-degree and homogeneous one-layer controls are annihilated.
- Status: Proved, class-specific.
- Scope: radialized angular degree at most 5; polynomial Stein component degree at most 4; T29 one-hidden-layer bias-free ReLU class.
- Wording: “Several named low-degree and homogeneous one-layer control classes vanish exactly under complete Kerdock.”

### High-degree zonal controls

- Claim: Small tested degree-6+ zonal dictionaries did not validate.
- Status: frozen empirical for selected degree-6+8 rule; exploratory for remaining sweep.
- Wording: “A frozen four-direction degree-6+8 correction failed; no tested small degree-6/8/10 dictionary produced a validated gain.”
