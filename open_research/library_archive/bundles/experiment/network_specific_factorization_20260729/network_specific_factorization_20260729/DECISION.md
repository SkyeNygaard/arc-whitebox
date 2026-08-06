# Decision — Retain One Module; Do Not Promote

## Decision

**Retain only the `128 x 256` output-contribution representation and the legal `beta_bar` SVD subspace as mechanism modules. Close the tested coefficient-learning family and do not promote a deployable estimator.**

A larger generic predictor is not justified by this experiment.

## Main findings

### 1. The optimistic oracle-state geometry is strongly low dimensional

On 24 frozen validation base networks with three rotations each, using the primary same-reference oracle-state construction:

| Arm | Aggregate ratio | Wins | Worst | Mean correction cosine |
|---|---:|---:|---:|---:|
| Full exact lower defect | 0.0847 | 72/72 | 0.2370 | 1.000 |
| Exact per-network rank 1 | 0.5811 | 69/72 | 1.0109 | 0.621 |
| Exact per-network rank 2 | 0.4301 | 72/72 | 0.8946 | 0.770 |
| Exact per-network rank 4 | 0.3000 | 72/72 | 0.6636 | 0.869 |
| Exact per-network rank 8 | 0.2056 | 72/72 | 0.5101 | 0.930 |
| Exact per-network rank 12 | 0.1561 | 72/72 | 0.4166 | 0.960 |

Thus rank 2 already meets the raw score gate in that oracle-state diagnostic. However, retaining 90% of the full correction benefit required median rank 10.5 and p90 rank 16. The literal “rank 2–8 captures nearly everything” premise is too strong, even though rank 2–8 has an excellent score ceiling.

### 2. Legal subspace selection is easier than signed coefficient prediction

The most useful legal deterministic subspace was the right singular space of the fold-averaged direct-control map `beta_bar`.

- `beta_bar` rank-2 subspace with oracle coefficients: ratio 0.6582, 70/72 wins, worst 1.0278.
- Same legal subspace with predicted signed coefficients: ratio 0.9993, 36/72 wins, mean cosine 0.027.
- Direct full-anchor-vector learner: ratio 1.0541, 4/72 wins, worst 1.2163.
- Frozen output template with learned scalar: ratio 1.0511.

The subspace miss is real—the exact rank-2 arm is 0.4301 versus 0.6582 with legal modes—but coefficient phase/scale is the dominant deployability failure.

### 3. Rotation stability does not solve identifiability

- Exact rank-2 mode subspaces: mean cross-rotation principal angle 23.9° in the primary run.
- Legal `beta_bar` subspaces: mean angle 2.0°.

The legal subspace is highly stable, but stable orientation did not make the signed coefficients predictable. Rotation-conditioned legal features therefore did not restore phase identifiability.

### 4. Independent reference streams invalidate promotion-level oracle claims

The independent audit used disjoint 32,768-node aggregate anchor and target pools per network. It found:

- median anchor-half correction relative disagreement: 1.779;
- median anchor-half correction cosine: 0.127;
- median target-reference noise equal to 0.664 of measured baseline pooled MSE.

With those finite-reference labels, the full estimated lower correction itself had independent pooled ratio 1.063, showing that the available independent teacher is not precise enough to certify the true oracle ceiling. This does not refute the existence of low-rank structure; it shows that the supplied/sandbox reference precision cannot support a promotion claim.

The deployable conclusions are still negative and stable under independent targets:

- legal rank-2 coefficient predictor: independent pooled ratio 1.011;
- direct 256-vector learner: independent pooled ratio 1.129;
- template-scalar learner: independent pooled ratio 1.125.

### 5. Codebooks and pooled modes do not rescue the branch

- Pooled rank-2 output basis: ratio 0.8987 in the primary diagnostic.
- Eight-template legal-selected codebook with oracle coefficients: ratio 0.9561.
- Raw suffix weight-product rank-2 modes with oracle coefficients: ratio 0.9836.

The network-specific modes cannot be replaced by a small universal output codebook at the tested rank.

## Identifiability answers

- **Is subspace selection easier than coefficient prediction?** Yes, substantially.
- **Is sign easier after projection onto a legal deterministic subspace?** No. Predicted correction cosine is approximately zero.
- **Are failures primarily subspace, sign, or scale?** Both subspace miss and coefficient error matter, but signed coefficient error dominates the deployable gap.
- **Does rotation conditioning restore sign?** No.
- **Does a finite codebook identify the modes?** No at the tested size/rank.
- **Can the failed broad anchor learner be reused?** Not as a direct vector predictor; it regresses.
- **Does K128 provide a better teacher?** Not answered: matched raw K32/K128 vectors were absent.
- **Should a larger model be trained?** No. Better independent lower-defect labels or a legal analytic coefficient/phase estimator must come first.

## Recommended handoff

Keep two pieces:

1. `C = diag(delta_anchor) beta_bar` as the authoritative factorization object for future exact-array audits.
2. `beta_bar` SVD as the strongest tested legal candidate subspace.

A future continuation is justified only after obtaining high-precision, independent lower-defect teachers. It should estimate 4–12 signed coefficients inside the frozen legal subspace, not predict the full 128-vector anchor or train a broader network model.
