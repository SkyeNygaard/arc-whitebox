# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 6 — Compute liberation and implementation

**Priority:** continuous support path  
**Workspace:** `paths/06_compute_liberation/`

## Goal

Free measured compute and residual wall so Paths 1–4 can fit, without introducing another approximate suffix compiler.

All integrated suffix compiler variants are closed in production:

- two-layer candidate/base approximately 1.02796;
- fixed-three approximately 1.04162;
- adaptive 2–6 approximately 1.11956;
- all confidence intervals above 1.

The unmodified partial-tree/Winograd package is the baseline.

## Broad implementation families

Agents may explore:

- fused direct-control accumulation;
- computing K32 features during existing propagation;
- reusing FWHT, Kerdock, or Winograd intermediates;
- fused mean and pair-moment reductions;
- eliminating all layer-31 replay;
- direct low-rank final-output contractions;
- matrix layout, packing, and batch-size changes;
- persistent buffers and allocation removal;
- compiled C/C++/SIMD kernels;
- lower-dispatch basis loops;
- memory-mapped/prepacked asset layouts;
- shared K32/K128 teacher computation during research;
- incremental basis packages for Path 4;
- mixed precision with frozen prediction-equivalence tests;
- compensated scalar accumulation where precision is binding;
- eliminating Python passes over 66,048 rows;
- exact FlopScope instrumentation and residual-wall attribution;
- caching network-only weight products or adjoints when reused by several anchor components.

## Exclusions

Do not revive symbolic suffix composition, stable-neuron classifiers, fixed-three shrinkage, adaptive-depth selection, or proxy arithmetic against the obsolete assembly-free baseline.

## Gates

Savings count only when measured in the complete subprocess package against the production baseline.

Each change must report:

- tracked FLOPs;
- residual wall time;
- peak memory;
- raw prediction equality or bounded error;
- complete adjusted score when numerical results change.

A compute change that reduces raw FLOPs but increases effective compute or changes predictions unsafely fails.
