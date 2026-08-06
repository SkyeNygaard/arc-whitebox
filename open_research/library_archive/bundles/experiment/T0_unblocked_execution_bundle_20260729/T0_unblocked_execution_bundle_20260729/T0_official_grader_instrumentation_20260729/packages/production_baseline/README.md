# Kerdock/MUB 5-design with partial-tree Winograd propagation

This preserves the validated 66,048-point Kerdock maximal-real-MUB spherical
5-design and replaces its 31 ordinary dense propagation products with an
honest tracked depth-5 Strassen--Winograd algorithm.

The recursion keeps the first three seven-product indices as tensor axes. The
final two levels retain their sixteen decoded quadrants as a small Python
tree, assemble them once, and then decode the outer levels conventionally.
This removes seven large intermediate `fnp.block` copies per hidden layer.
It does not use untracked NumPy arithmetic.

Selection IDs 0--9:

- dense Kerdock raw MSE: `1.715489347e-7`;
- Winograd raw MSE: `1.715362009e-7`;
- maximum final-mean drift: `7.551e-6`;
- tracked FLOPs per network: `170,906,815,488`;
- audited ID-0 effective compute: `175,871,462,860`;
- audited ID-0 adjusted score: `1.118079200e-7`.

The isolated official row-0 profile took 24.28 seconds, with 0.0547 seconds
of charged residual time, below the 30-second predict guard. No budget or
time gate fired. The original dense Kerdock submission remains in its own
directory.


## Path 6 candidate

Only the final layer is evaluated in 2,048-row chunks and reduced immediately.
The propagated 66,048-point design, weights, depth-5 partial-tree Winograd kernel,
and scoring identity are unchanged. The goal is lower wall time and peak memory,
not an approximate compiler.
