# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 5 — Set-level coreset and subspace compression

**Priority:** low; broad offline exploration  
**Workspace:** `paths/05_set_level_coreset/`

## Goal

Find a qualitatively different low-compute estimator that propagates far fewer Kerdock rows while preserving the signed whole-set cancellation needed for the final mean.

This is not a row-importance problem. K2, nonlinear per-row phase scores, and independent pairwise complementarity all found local signal while missing the same-support oracle-error gate by factors of 197–729.

## Broad representation families

Agents may explore:

- weight-product and ReLU-adjusted product subspaces;
- layerwise sensitivity subspaces;
- downstream Gramians and adjoint subspaces;
- subspace-exact positive cubature;
- conditioning-aware set objectives;
- direct prediction of same-support oracle-weight error;
- set-level neural scorers;
- differentiable offline support selection;
- positive bounded-weight constructions;
- precomputed coreset libraries;
- Kerdock/Clifford symmetry-orbit tables;
- Grassmannian quantization;
- support families indexed by weight-derived invariants;
- spectral sparsification and volume/conditioning objectives;
- batch/set encoders that score an entire support jointly;
- offline optimal-transport or determinantal constructions, provided runtime selection is closed-form or table-based.

## First gate — before runtime optimization

For fresh exact-geometry width-256 networks, an affordable representation must predict a support with:

- same-support oracle-weight added MSE approximately at most 1.1e-8;
- safe worst network;
- positive, bounded weights and acceptable conditioning;
- complete runtime selection cost below saved propagation.

Row-importance correlation, pairwise correlation, top-support recall, or support overlap are diagnostics only.

## Runtime constraints

No runtime NNLS, greedy herding, exchange rounds, general LP, or iterative solver. Effective compute includes residual wall at 1e11 FLOPs per second, making even modest Python/solver latency fatal near the compute floor.

A runtime method must be structured, precomputed, closed-form, or a very small compiled inference pass.

## Suggested first falsification

Measure whether oracle support quality is predictable from affordable weight-product, ReLU-adjusted, and downstream-sensitivity subspaces. Then construct the best frozen subspace-moment support and evaluate same-support oracle weights. Stop immediately when the gate is missed.

Only after this gate passes may an agent implement support weights or the final estimator.
