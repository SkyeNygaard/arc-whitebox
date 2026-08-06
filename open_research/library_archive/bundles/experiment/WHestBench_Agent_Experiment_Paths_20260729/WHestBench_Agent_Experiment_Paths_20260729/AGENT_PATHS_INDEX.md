# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Start here

The canonical ledger contains an **Agent Paths** sheet and a **Document Index** sheet pointing to this folder. Read `00_SHARED_PROTOCOL.md` before selecting a path.

## Path index

1. **Legal signed-anchor estimation** — primary statistical lane. Estimate the frozen K32 lower-order anchor legally and cheaply.
2. **Learned sign, scale, and abstention** — predict only the missing network-specific phase, scale, residual, and harmful-tail risk.
3. **Internally centered defect transport** — propagate small Kerdock-versus-Gaussian defects rather than large absolute moments.
4. **Kerdock allocation and adaptive basis count** — use 112 or fewer bases to pay for an already-working legal anchor.
5. **Set-level coreset and subspace compression** — low-probability offline route; continue only after the same-support oracle-error gate.
6. **Compute liberation and implementation** — fuse and reuse arithmetic without inventing another suffix compiler.

## Current canonical facts

- Production baseline adjusted score on Mini-100: approximately 1.47368e-7 at 175.62B effective compute.
- Integrated compiler results: two-layer 1.02796, fixed-three 1.04162, adaptive 1.11956 candidate/base; all closed.
- Uniform independent oracle anchors: K32 approximately 0.403 candidate/base and 80.56% full-oracle-gap capture; K128 approximately 0.322 and 91.53% capture.
- Direct final-output control and layer-31 replay are tied; direct control avoids about 8.657B FLOPs.
- Lower-order means and pair moments are the radial-Hermite target. Connected-only is approximately neutral.
- The remaining blocker is legal signed-anchor estimation with rotation/no-headroom safety.

## Dependency graph

- Paths 1–3 and 6 may run immediately.
- Path 2 should use the best frozen analytic estimator from Path 1 when available, but may begin with oracle labels and exposed training data.
- Path 4 may perform offline oracle allocation work, but may not open a fresh production validation until a legal anchor passes at 129 bases.
- Path 5 is offline and low priority; no runtime implementation until the support gate passes.
