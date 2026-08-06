# Kerdock/MUB 5-design with Winograd propagation

This preserves the validated 66,048-point Kerdock maximal-real-MUB spherical
5-design and replaces its 31 ordinary dense propagation products with an
honest tracked depth-5 Strassen--Winograd algorithm.

The recursion keeps the first three seven-product indices as tensor axes,
recurses depth-first for the final two levels, and reconstructs quadrants with
one-charge `fnp.block`. It does not use untracked NumPy arithmetic.

Selection IDs 0--9:

- dense Kerdock raw MSE: `1.715489347e-7`;
- Winograd raw MSE: `1.715362009e-7`;
- maximum final-mean drift: `7.551e-6`;
- tracked FLOPs per network: `175,822,834,176`;
- official row-0 adjusted score: `1.152501490e-7`.

The isolated official row-0 profile took 24.28 seconds, with 0.0547 seconds
of charged residual time, below the 30-second predict guard. No budget or
time gate fired. The original dense Kerdock submission remains in its own
directory.
