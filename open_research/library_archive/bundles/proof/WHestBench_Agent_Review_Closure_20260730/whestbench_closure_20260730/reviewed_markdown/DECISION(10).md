# DECISION

## Verdict

# VERIFIED AFTER SPECIFIED CORRECTIONS

The mathematical T22 near-optimality theorem is valid within its stated scope, conditional on the directed-interval minorant and spherical-mean certificates. I found no logical error in the ensemble-MSE identity, Delsarte argument, diagonal residual term, node-budget inequality, Kerdock multiplicities, spherical-mean tail logic, or final inequality directions.

The independent checker reconstructs the exact appendix witness and reproduces the theorem-critical non-proof numerics, including the global near-contact value to the displayed digits.

The qualification is required because the shared evidence set contains a stale machine-readable artifact that falsely gives a positive lower bound on Kerdock suboptimality. The theorem is one-sided. Once that artifact is removed or replaced and the randomized-rule wording is tightened, my verdict becomes simply **VERIFIED**.

## Independent theorem statement

Let `K32` be the explicit normalized depth-32 ReLU kernel on `S^255`, and let `C_N` be the class of linear rules

`Qf = sum_{i=1}^m w_i f(x_i)`

with `m<=66048`, `x_i in S^255`, `w_i>=0`, and `sum_i w_i=1`. Let `Q_K` be the uniform rule on the antipodal union of 129 real MUBs. Then the certified constants imply

`2.4330918534409412569e-7 <= inf_{Q in C_N} D_K32^2(Q)`

and

`D_K32^2(Q_K) <= 2.4336603575430052277e-7`.

Consequently,

`1 <= D_K32^2(Q_K) / inf_{Q in C_N} D_K32^2(Q) <= 1.000233655010295`.

For any scalar random field with second moment `K32(<x,y>)`, the same statement is an ensemble-MSE result for deterministic rules independent of the field and for randomized admissible rules independent of the field almost surely.
