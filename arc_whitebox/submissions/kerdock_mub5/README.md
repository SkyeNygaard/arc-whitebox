# Kerdock maximal-MUB spherical 5-design

**Status:** validated reference archive. The archive SHA-256 is
`e60c0a686188f9fe030c1a3769b29859d539902d9a43be40e4b6f9883dd663ae`.

This estimator replaces generic randomized QMC with the 66,048-point spherical
5-design obtained from the maximal 129 real mutually unbiased bases in
dimension 256.

The 128 non-coordinate bases come from the binary Kerdock code. Their
signed-Hadamard structure permits an exact fast Walsh-Hadamard first layer;
the coordinate basis is also exact without a dense first-layer multiply.
All later layers are ordinary tracked `flopscope.numpy` matrix products.

Directions have radius `E[chi_256]`, integrating the Gaussian radial variable
exactly. The design integrates every spherical polynomial of degree at most
five exactly, including all odd components, covariance, and fourth moments.
Rotation seed 3 was selected on official mini IDs 0-49 and then frozen.

Frozen research protocol and official validation:

- selection IDs 0-49: raw final MSE about `1.759e-7`;
- frozen holdout IDs 50-99: raw final MSE `2.80643696e-7`;
- official all-100 raw final MSE `2.28259133e-7`;
- official all-100 adjusted score **`2.25656459e-7`**;
- tracked FLOPs per network `268,835,176,704`;
- mean effective compute `268,898,960,582`;
- failures `0 / 100`.

The structured estimator reproduces the dense research implementation within
float-association noise. `submission.tar.gz` is the archive to validate before a
competition upload. Results apply only to the documented exposed Mini cohort;
they do not certify a protected-evaluation score.
