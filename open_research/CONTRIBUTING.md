# Contributing

Contributions that correct or falsify existing work are as welcome as improvements.

## Contribution types

### Reproduction

Use the reproduction-gap issue template. Include environment, exact commands, hashes, and generated artifacts.

### Correction

State the original claim, the smallest counterexample or inconsistency, and proposed replacement wording. Preserve the old record through a superseding ledger row.

### New estimator

State:

- estimator class;
- what new runtime information or structure it uses;
- which prior closed class it leaves;
- oracle gate;
- legal rollout;
- correct structured-error metric;
- full cost and tail behavior;
- frozen promotion threshold.

### Proof improvement

Include a theorem statement, assumptions, proof, equality cases, hostile examples, and exact/computer-assisted verification boundary.

## Data and protected evaluation

Do not upload restricted benchmark data or protected results without permission. Disclose every cohort used for tuning or evaluation.

## Style

- Avoid universal language for scoped results.
- Use `reported` when raw artifacts are unavailable.
- Separate raw MSE from adjusted score.
- Separate oracle capacity from deployability.
- Charge the union of queries across outputs.
