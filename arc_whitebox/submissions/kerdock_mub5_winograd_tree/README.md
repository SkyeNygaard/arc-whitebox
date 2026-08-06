# Kerdock/MUB 5-design with partial-tree Winograd propagation

**Status:** structurally valid research candidate. The archive SHA-256 is
`a7f5e1e58639192e33e0886e776b4c8392399a7879e372bed557811516ec93e7`.
Only the isolated ID-0 audit below is attached directly to this archive. A later
all-100 exposed-Mini result is reported elsewhere in the project, but the exact
matching shipping archive and hash were not recovered; do not present that later
number as independently reproduced or competition-certified.

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
of charged residual time, below the 30-second predict guard. No budget or time
gate fired. Before uploading, capture a fresh full exposed-Mini run and record
the resulting archive SHA-256 beside the result JSON.
