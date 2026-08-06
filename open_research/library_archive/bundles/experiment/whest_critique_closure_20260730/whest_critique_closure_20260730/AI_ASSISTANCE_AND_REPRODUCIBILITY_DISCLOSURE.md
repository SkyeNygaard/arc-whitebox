# AI Assistance, Reproducibility, and Human Responsibility Disclosure

## Draft disclosure for the manuscript

Generative AI systems assisted with research organization, code drafting, symbolic manipulation, proof exploration, adversarial review, experiment scripting, report generation and manuscript drafting. Model-generated agreement is not treated as independent expert verification. Every result is classified by its underlying evidence—exact proof, computer-assisted certificate, frozen empirical output, oracle diagnostic or exploratory analysis—rather than by the number of agents that endorsed it.

The T22 and T16 results rely on released source code, exact rational/integer arithmetic, directed interval arithmetic, deterministic artifacts and hash manifests. They are computer-assisted proofs, not proof-assistant formalizations. The finite-width support theorem and information-theoretic results are conventional mathematical arguments whose final statements and proofs require named human review.

Named human authors retain responsibility for:

1. checking every theorem statement and its assumptions;
2. reading the Hermite, Faà di Bruno, duality, association-scheme and conditional-expectation arguments line by line;
3. running proof-critical code from a clean environment;
4. verifying that every empirical number links to the stated rows, cohort, selection chronology and metric;
5. confirming related-work and novelty claims;
6. approving the final scope, limitations and conclusions.

## Required artifact-level disclosure

For each released file, the final repository should record:

- whether it was authored, modified or reviewed with model assistance;
- the human reviewer responsible for sign-off;
- the command and environment used to regenerate it;
- its SHA-256 digest;
- whether it is proof-critical, evidence-bearing, explanatory or archival only.

## Prohibited wording

The manuscript must not describe the proofs as “formally verified,” the agents as independent human referees, or an internal hash file as external authentication.
