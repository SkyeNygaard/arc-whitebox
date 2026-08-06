# Sparse MUB adjusted-score frontier

## Question

Could a union of fewer antipodal Kerdock bases accept worse raw cubature error
but win after the challenge's compute multiplier? Every equal-weight union of
orthonormal antipodal bases remains an exact spherical 3-design, and each
basis costs only 512 rows.

All subset work used official IDs 0--49. IDs 50--99 were not inspected.

## Honest subset protocol

We cached the final-layer mean of each of 129 bases at rotation seeds
0, 1, 3, and 5. Four candidate families were compared:

1. bases from seed 3 only;
2. unrestricted bases from all four rotations;
3. common basis-index groups across all four rotations;
4. independently selected bases with an equal quota per rotation.

Greedy equal-weight subset selection and one-for-one exchanges were fitted on
40 networks and evaluated on the held-out 10-network fold. A stricter nested
consensus used four 30-network inner fits per outer fold, ranking bases by
selection frequency and mean greedy rank.

Unrestricted rotated unions severely overfit: their fold-to-fold Jaccard
similarity was often below 0.2, and their nested test MSE was worse than
seed-3-only subsets. Seed-3-only selection became stable as the subset
approached the complete design.

## Frontier

The table uses the better honest estimate (direct nested selection or nested
consensus) at each size and the separately profiled safe Strassen/Winograd
compute multiplier.

| bases | rows | honest raw MSE | compute multiplier | projected adjusted |
|---:|---:|---:|---:|---:|
| 24 | 12,288 | 1.4742e-6 | 0.132463 | 1.9527e-7 |
| 32 | 16,384 | 1.0477e-6 | 0.173649 | 1.8195e-7 |
| 48 | 24,576 | 6.6452e-7 | 0.256020 | 1.7013e-7 |
| 64 | 32,768 | 4.8647e-7 | 0.338392 | 1.6462e-7 |
| 80 | 40,960 | 3.5890e-7 | 0.418623 | 1.5024e-7 |
| 96 | 49,152 | 3.0268e-7 | 0.498753 | 1.5096e-7 |
| 112 | 57,344 | 2.3223e-7 | 0.578883 | 1.3444e-7 |
| 129 | 66,048 | 1.7587e-7 | 0.664021 | **1.1678e-7** |

The compute-aware optimum is the complete 129-basis design. The degree-four
cancellation gained by the final bases is worth more than their marginal
compute cost, even after fast-matrix-multiplication discounts.

## Important overfitting diagnostic

Selecting 129 bases from the unrestricted four-rotation pool on all 50
networks produced an apparently excellent raw MSE near `8e-8`, but honest
five-fold test MSE was about `3e-7`. This 3.7x reversal is a useful warning:
the 50 networks provide far fewer effective independent observations than
their 12,800 output coordinates suggest. Rotation/basis subset optimization
must be nested at the network level.

## Artifacts

- `scripts/eval_sparse_kerdock_frontier.py`
- `results/sparse_kerdock_frontier_selection.json`
- `scripts/eval_sparse_kerdock_consensus.py`
- `results/sparse_kerdock_consensus_selection.json`
- `scripts/eval_sparse_mub_herding.py`
- `results/sparse_mub_herding_selection.json`
- `results/strassen_sparse_basis_frontier.json`
