# T16 interval-proof supersession note

The early `close_t16_primal_dual.py` artifact used the claims

- `F''/F' < 9/4`, and
- `B_{6,2} >= -kappa^(6)/4`

to prove sixth-derivative positivity. That route is **not canonical** and must not be cited. Later hostile review correctly required the published proof to use the directly certified inequalities

- `F''/F' < 2.398586389549085 < 3`, and
- `kappa^(6)+3B_{6,2}>0` on `[-1,0]`.

The new `t16_mpmath_iv_second_stack.py` independently implements this corrected route. It reproduces the Decimal/libmpdec bounds to more than 70 digits and uses a Krawczyk coefficient enclosure rather than naive interval linear solving.

The earlier artifact should be retained only as historical provenance and marked **SUPERSEDED — DO NOT CITE**.
