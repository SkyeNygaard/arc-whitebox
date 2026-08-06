# Tracked Winograd propagation for the Kerdock design

## Outcome

The strongest audited estimator keeps the existing 66,048-row Kerdock/MUB
spherical 5-design and changes only the 31 later-layer matrix products. A
tracked depth-5 Strassen--Winograd implementation lowers the estimator from
about 272 billion billed FLOPs to exactly `175,822,834,176` billed FLOPs on
the official width-256, depth-32 task.

On official mini row 0 in the isolated subprocess runner:

- raw final-layer MSE: `1.729180638676553e-7`;
- effective compute: `181,288,408,133.19`;
- score multiplier: `0.6665015004896688`;
- adjusted score: `1.1525014902956064e-7`;
- predict wall time: `24.2783 s` under a `30 s` limit;
- residual wall time: `0.05466 s`;
- no compute, wall-time, residual-time, or combined-budget failure.

The selection-ID 0--9 audit found mean raw MSE
`1.7153620094080708e-7`, versus `1.7154893470257354e-7` for ordinary dense
propagation. The raw delta is `-1.2734e-11`, so numerical reassociation did
not damage the estimate. The maximum final-mean drift was `7.551e-6`.

## Why rectangular recursion matters

The design matrix has shape `(66048, 256)`, or exactly `(258 * 256, 256)`.
Padding it into square batches and billing reshapes/copies gives away much of
Strassen's advantage. The implemented recursion instead splits the last two
axes directly:

```text
(m, n) @ (n, n)
  -> seven (m/2, n/2) @ (n/2, n/2) products
```

All earlier product indices remain explicit tensor axes. They are never
flattened with a billed reshape. The first three recursion levels are packed;
the final two are evaluated depth-first. Quadrants are reconstructed with a
single `fnp.block` per node.

The rank-7 schedule is the 15-add Winograd variant. Per node, before recursive
products, it uses four tall-left additions and four small-right additions.
Reconstruction uses seven tall-output additions. `fnp.block` adds one
output-copy charge. For a parent `(m,n) @ (n,n)`, the exact direct recurrence
at depth `d` is:

```text
leaf
+ 3.75 * sum_l 7^l * m_l * n_l
+ 1.00 * sum_l 7^l * n_l^2
```

where the coefficient `3.75` includes the billed output assembly. This
accounting predicts the measured tracker totals exactly.

## Schedule audit

For 66,048 rows, depth 5 with three packed levels was the best measured
effective-compute point:

| schedule | one-layer billed FLOPs | key tradeoff |
|---|---:|---|
| depth 4, hybrid p3 | about `5.798 B` | faster, but more multiplication |
| depth 5, hybrid p2 | `5.549 B` | lower arithmetic, higher Python residual |
| depth 5, hybrid p3 | `5.640 B` | best full-network effective compute |
| depth 5, hybrid p4 | `5.799 B` | too much packed copying and wall time |

Pure depth-5/p3 beat depth-4/p3 and mixed-depth schedules after the residual
wall-time penalty was included.

## Larger-row no-go bound

Fast multiplication does not make the attractive 132,096- or 198,144-row
full ensembles fit. For 132,096 rows the exact direct Winograd arithmetic is
minimized at depth 5, about `10.933 B` per later layer. Thirty-one later
layers plus ReLU already cost about `341.0 B`, before the first layer and
residual-time charge. Even hypothetically deleting every block-copy charge
leaves about `317.3 B`.

Therefore any larger design must compress, couple, or correct rows before
propagation; another recursion depth alone cannot solve it.

## Tensor-leg permutation no-go

All six dihedral symmetries of the rank-7 tensor were also implemented and
checked as exact integer coefficient identities. A coefficient-factor
permutation does not permute Winograd's straight-line addition counts in the
naive way. When a 4-to-7 encoder factor is moved to the output leg, the
implementation needs its 7-to-4 adjoint circuit:

```text
U input: 4 additions       U^T output: 7 additions
V input: 4 additions       V^T output: 7 additions
W^T input: 4 additions     W output:   7 additions
```

After rank-sign normalization, every one of the six schemes is therefore
`4 / 4 / 7`, not `4 / 7 / 4`, and every scheme tracked exactly
`5,639,804,672` FLOPs for one depth-5/p3 later layer. The fastest micro-timed
permutation (`V,Z,U`) produced a worse row-0 adjusted result
(`1.15838e-7`) than the packaged standard orientation (`1.15250e-7`).
The full audit is `scripts/eval_winograd_leg_permutations.py`.

## Sparse-design frontier

Combining measured raw MSE with tracker-exact Winograd projections did not
produce a smaller-basis winner. The projected adjusted frontier decreased
through the full 129-basis construction. Representative adjusted projections
were about:

- 80 bases: `1.502e-7`;
- 96 bases: `1.510e-7`;
- 112 bases: `1.344e-7`;
- all 129 bases: `1.15e-7` to `1.17e-7`, depending on measured residual time.

The full design remains the robust choice.

## Artifacts and reproduction

The deployable package is:

```text
submissions/kerdock_mub5_winograd/submission.tar.gz
SHA-256 3d54e05c7615aa841f2a2480840ba01e9d87d2dd5fce0473c1f3504fff3253c8
```

It contains only `estimator.py`, `fast_matmul.py`,
`kerdock_mub5_seed3.npz`, and `manifest.json`. The original dense estimator
is untouched in `submissions/kerdock_mub5`.

Reproduce validation and row 0:

```bash
.venv/bin/whest validate-package \
  submissions/kerdock_mub5_winograd/submission.tar.gz

env HF_HOME=/tmp/arc-whest-hf \
  HF_DATASETS_CACHE=/tmp/arc-whest-hf/datasets \
  XDG_CACHE_HOME=/tmp/arc-whest-cache \
  .venv/bin/whest run \
  --estimator submissions/kerdock_mub5_winograd/estimator.py \
  --runner subprocess \
  --dataset data/official_phase1_mini \
  --split mini \
  --n-mlps 1 \
  --detail full \
  --profile \
  --wall-time-limit 30 \
  --max-threads 1
```

The standalone broader audit is `scripts/eval_strassen_audit.py`; the
90,624-row multifidelity audit is
`scripts/eval_strassen_multifidelity_audit.py`.
