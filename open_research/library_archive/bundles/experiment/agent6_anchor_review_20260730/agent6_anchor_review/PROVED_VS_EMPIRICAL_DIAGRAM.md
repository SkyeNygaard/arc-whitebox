# Layer-31 claim map

```mermaid
flowchart TD
    A[Protected estimator error e] --> B[Proposed correction u]
    B --> C[Exact quadratic risk identity]
    C --> D[Selector value depends on signed inner product]
    C --> E[Correction subspace S]
    E --> F[Linearized layer-31 map J]
    F --> G[Invariant threshold: ||J xi||² < ||J d||²]
    G --> H[True final ReLU replay]
    H --> I[Gate-crossing remainder bound]

    J[Same-design folds Zi = mu + b + eps_i] --> K[Common-bias non-identifiability]
    K --> L[Fold disagreement cannot supply absolute phase]

    M[M146 headline curve] --> N[Provisional empirical claim]
    N --> O[Raw rows / seeds / perturbation manifest missing]
    N --> P[Internally consistent with quadratic noise law]
    N --> Q[Structured-direction robustness untested]

    R[T4 legal anchor experiments] --> S[Near-zero or negative signed transfer]
    R --> T[Rotation-specific phase does not group]

    C:::proved
    D:::proved
    G:::model
    I:::proved
    K:::proved
    N:::empirical
    O:::missing
    P:::check
    Q:::open
    S:::empirical
    T:::empirical

    classDef proved fill:#d8f3dc,stroke:#2d6a4f;
    classDef model fill:#fff3bf,stroke:#a66f00;
    classDef empirical fill:#dbeafe,stroke:#1d4ed8;
    classDef missing fill:#fee2e2,stroke:#b91c1c;
    classDef check fill:#ede9fe,stroke:#6d28d9;
    classDef open fill:#f3f4f6,stroke:#4b5563;
```

## Claim labels

- **PROVED:** correction-risk identity; constrained selector formula; common-bias non-identifiability; scalar/vector ReLU crossing bounds.
- **PROVED UNDER EXPLICIT MODEL:** layer-31 replacement threshold in the frozen linearized correction subspace.
- **EXPLORATORY EMPIRICAL:** M146's 60-network values until original rows and manifests are restored.
- **FROZEN EMPIRICAL:** T4 legal-anchor closure and activation-region validation, within their stated cohorts and protocols.
- **OPEN:** robustness of the M146 scalar threshold to actual residual, analytic-residual, companion-residual, downstream-singular, and kink-concentrated directions.
