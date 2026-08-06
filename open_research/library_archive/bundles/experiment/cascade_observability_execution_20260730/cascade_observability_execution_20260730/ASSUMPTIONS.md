# Assumptions and Scope

## Verified for retained architecture-matched code

- input dimension 256;
- width 256;
- depth 32;
- independent Gaussian matrices with scale `sqrt(2/256)`;
- no biases;
- ReLU after every matrix multiplication, including the final layer;
- complete Kerdock support represented by 128 chirp-Hadamard bases plus the coordinate basis and antipodes.

## Required by the all-width fixed-linear symmetry theorem

- square-integrable output field;
- rotationally invariant network prior and input integration measure;
- zonal second-moment kernel;
- complete Kerdock support with constant kernel Gram row sum;
- fixed linear weights and total mass one.

## Not assumed / not proved

- Gaussianity of the post-ReLU output process;
- optimality among nonlinear or data-dependent algorithms;
- no value from adaptive point evaluations;
- a universal finite-width `O(L/n)` exploitability sector;
- irreducible independent per-layer injection floors;
- a complete coherence matrix for the archived oracle ladder;
- safety or deployability from average correction cosine alone.
