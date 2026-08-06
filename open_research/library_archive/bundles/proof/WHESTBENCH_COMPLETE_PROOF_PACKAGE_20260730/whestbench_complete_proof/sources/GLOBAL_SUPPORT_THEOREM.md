# Global antipodal Kerdock-support theorem

Let the 33,024 projective Kerdock lines be indexed by basis `b=1,...,129` and line `i=1,...,256`. Give the antipodal line `(u,-u)` real total weight `w_{bi}`, and require `sum w_{bi}=1`. Define basis totals `S_b=sum_i w_{bi}`.

For the symmetrized depth-32 ReLU kernel, only three pair classes occur inside this universe:

- same line: `A=(K(1)+K(-1))/2`;
- distinct lines in one orthonormal basis: `O=K(0)`;
- lines in different mutually unbiased bases: `C=(K(1/16)+K(-1/16))/2`.

If `A0` is the exact spherical kernel mean, the ensemble integration MSE is exactly

```
R(w) = (C-A0) + (O-C) sum_b S_b^2 + (A-O) sum_{b,i} w_{bi}^2.
```

At depth 32 and dimension 256:

```
A-O =  0.011988581160655598 > 0
O-C = -0.000009468153657654632 < 0
C-A0 = -0.000000046263743724850315
```

For a fixed support with `r_b` retained lines in basis `b`, Cauchy–Schwarz gives

```
sum_i w_{bi}^2 >= S_b^2 / r_b,
```

with equality only for equal weights within that support, even when signed weights are allowed. Hence define

```
c(r) = (O-C) + (A-O)/r.
```

All `c(r)>0` for `1<=r<=256`, and the exact optimal basis masses are

```
S_b = [1/c(r_b)] / sum_j [1/c(r_j)],
R_min(r_1,...,r_129) = (C-A0) + 1 / sum_b [1/c(r_b)].
```

With a total line budget `P`, optimizing the support reduces to maximizing

```
H = sum_b h(r_b),   h(r)=r / [(A-O)+(O-C)r].
```

Because `O-C<0`, `h` is strictly convex. Subject to `0<=r_b<=256` and `sum r_b=P`, the global optimum therefore fills complete bases as aggressively as possible: `floor(P/256)` complete bases plus at most one partial basis. This is a global result over every antipodal support and every real weighting inside the Kerdock line universe—not merely rectangular subsets.

The theorem covers the infinite-width depth-32 kernel objective and network-independent supports/weights. It does not cover network-adaptive finite-width selection, non-antipodal rules, nonlinear estimators, or nodes outside the Kerdock line universe.

Numerical stress test: 2,000 randomized count/weight trials at each of 13 budgets, including signed within-basis perturbations, produced zero violations. The proof, not the randomized test, is authoritative.
