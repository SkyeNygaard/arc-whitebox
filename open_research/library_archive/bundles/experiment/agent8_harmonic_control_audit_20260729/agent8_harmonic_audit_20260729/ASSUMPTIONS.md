# Assumptions

## Radialized polynomial lemmas

- The complete Kerdock rule used by the project is a spherical 5-design.
- Radial expectation is evaluated exactly and does not introduce node-dependent fitting.
- Learned coefficients are fixed functions of network weights before evaluation and do not depend on the current quadrature node.
- Any downstream map claimed to preserve the no-op is linear; nonlinear suffixes may generate higher degrees.

## One-hidden-layer ReLU Stein lemma

- Vector field: `phi(x)=sum_j a_j ReLU(v_j^T x)`.
- No input biases.
- Parameters are fixed with respect to `x`.
- Each block is the antipodal union of an orthonormal basis.
- Gaussian radialization uses `E[R^2]=d` exactly.
- At quadrature nodes with `v_j^T x=0`, use the symmetric convention `ReLU'(0)=1/2`; alternatively assume generic nonzero projections. This convention is immaterial to the Gaussian expectation but material to exact blockwise quadrature.

## Harmonic oracle figures

- Dimension 256, depth-32 infinite-width normalized ReLU kernel.
- Harmonic components are interpreted in the exact kernel decomposition.
- “Ceiling” means exact removal of only the named orthogonal degree component, with no compute charge and no effect on other degrees.
- No finite-width transfer is assumed.

## Frozen experiment

- Only the selected degree-6+8 four-direction configuration has an independent frozen stage.
- The shrinkage stage is development on the former frozen cohort and cannot be promoted to another frozen result.
