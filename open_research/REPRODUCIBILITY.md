# Reproducibility contract

## Theory claims

A complete independent verification should:

1. regenerate the depth-32 ReLU kernel coefficient intervals with a separate directed-arithmetic implementation;
2. verify the kernel mean and Kerdock energy intervals;
3. rerun the T16 degree-five primal/dual certificate;
4. replay the original signed witness and the conservative frozen-witness inertia/sign-count verifier against independently generated coefficient endpoints;
5. verify every quoted ratio using consistent upper/lower endpoints;
6. record operating system, language/runtime, arbitrary-precision libraries, and exact commands;
7. publish hashes for every proof-critical file;
8. distinguish the archived reoptimized T70 report from the fully replayable frozen-witness headline.

Reusing inherited interval endpoints is a replay, not an independent reconstruction.

## Empirical claims

A complete empirical reproduction should provide:

- exact source commit;
- environment lockfile;
- data/split identifiers;
- random seeds and frozen configurations;
- exact package archive and SHA-256;
- commands and stdout/stderr;
- per-network results, not only aggregate summaries;
- tracked FLOPs and residual wall time;
- failure and fallback behavior;
- protected-data disclosure.

## Grouping and leakage

Derived rotations or repeated networks must be grouped by base network. Hyperparameters and thresholds must be frozen before evaluating a new group. Oracle fitting and legal rollout must be reported separately.

## Missing empirical bundle

The independent audit identified ten required items for the final v31 empirical claims, including the 44 regeneration scripts, saved arrays, exact metric definitions, package archive, environment lockfile, official JSON, and root-failure output. They remain open issues.

## Clean-checkout commands

```bash
python scripts/check_release_strict.py
python scripts/run_core_verification.py
```

The first command checks manifest coverage, hashes, internal links, placeholders, and required evidence. The second replays the all-degree reduced-cost proof, validates the recovered v5.2 bundle, independently recomputes the Kerdock risk as a non-directed numerical sanity check, and replays both signed-witness verifiers.
