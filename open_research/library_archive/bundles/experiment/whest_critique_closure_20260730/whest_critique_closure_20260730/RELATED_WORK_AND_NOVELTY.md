# Related Work and Novelty Positioning

## Purpose

This note addresses the hostile-referee concern that novelty had not been demonstrated. It is a targeted positioning review, not a claim of exhaustive priority search.

## Mathematical foundations

Delsarte, Goethals and Seidel introduced spherical designs and the Gegenbauer/linear-programming framework for spherical codes. The present work uses that classical machinery in a kernel-energy lower-bound problem rather than claiming a new general LP method.

Cohn and Kumar developed universal energy optimality for sharp spherical configurations and completely monotone potentials. Their results explain why design strength and few-distance structure can imply broad energy optimality. The deep-ReLU kernel considered here does not permit an immediate universal-optimality conclusion; the project instead constructs and certifies a kernel-specific minorant and explicitly records low-dimensional counterexamples to naive design-optimality extrapolation.

Calderbank, Cameron, Kantor and Seidel connected Kerdock codes, orthogonal spreads and extremal Euclidean line sets. Later MUB/design/frame work clarifies the projective-design structure of complete mutually unbiased bases. These sources supply the construction and association geometry; they do not, in the material located in this search, supply the challenge-specific deep-ReLU auxiliary certificate or the exact arbitrary-real-weight optimization at every line budget used here.

Schoenberg's characterization of positive-definite zonal kernels by nonnegative Gegenbauer coefficients is the harmonic foundation for kernel-energy arguments on spheres.

## Neural kernels and probabilistic integration

Cho and Saul introduced arc-cosine kernels associated with infinite random neural features. Lee et al. established the deep infinite-width neural-network/Gaussian-process correspondence and recursive covariance computation. The present limiting kernel is an instance of that lineage; the novelty is not the ReLU kernel recursion itself.

Bayesian/probabilistic quadrature and kernel quadrature formulate integration error through a positive-definite kernel. Briol et al. survey probabilistic integration, and Kanagawa et al. review the equivalence between Gaussian-process and RKHS/kernel viewpoints. The ensemble-MSE identity in this paper is a specialization of that broader principle to a random neural integrand and a rotationally invariant spherical domain.

Oates, Girolami and Chopin develop control functionals and Stein-based variance reduction. The empirical companion's Stein/control experiments should be positioned as model-specific null-space and failure results, not as a general negative result for control functionals.

## Contributions that can plausibly be claimed

The safest novelty claims are:

1. a computer-assisted, kernel-specific arbitrary-node lower certificate at `d=256`, depth 32 and `N=66,048`, yielding a one-sided Kerdock gap below `0.023325%` in the static nonnegative class;
2. a computer-assisted proof that the associated degree-5 Hermite minorant is the unique optimizer of the all-degree auxiliary LP;
3. an exact arbitrary-real-weight support-allocation theorem within a fixed real-MUB/Kerdock line universe, plus a finite-width extension for Gaussian-first-layer ensembles under explicit nondegeneracy;
4. an exact all-width fixed-complete-support uniform-weight theorem by symmetry;
5. a unified, reproducibly labeled boundary map separating geometric capacity, observable signed phase, oracle value, deployability, tail risk and compute.

## Claims that should not be made

- that spherical-design LP bounds, kernel quadrature, MUB geometry, Gaussian noise stability, or conditional-expectation projection theory are themselves new;
- that Kerdock is universally energy optimal;
- that the finite-width T27 extension is an arbitrary-node finite-width T22 theorem;
- that the literature search proves no related certificate exists;
- that failure to locate a prior result establishes priority.

## Bibliography core

- Delsarte, P.; Goethals, J. M.; Seidel, J. J. “Spherical codes and designs.” *Geometriae Dedicata* 6 (1977), 363–388. DOI: 10.1007/BF03187604.
- Schoenberg, I. J. “Positive definite functions on spheres.” *Duke Mathematical Journal* 9 (1942).
- Calderbank, A. R.; Cameron, P. J.; Kantor, W. M.; Seidel, J. J. “Z4-Kerdock Codes, Orthogonal Spreads, and Extremal Euclidean Line-Sets.” *Proceedings of the London Mathematical Society* 75 (1997), 436–480. DOI: 10.1112/S0024611597000403.
- Cohn, H.; Kumar, A. “Universally optimal distribution of points on spheres.” *Journal of the American Mathematical Society* 20 (2007), 99–148. DOI: 10.1090/S0894-0347-06-00546-7.
- Klappenecker, A.; Rötteler, M. “Mutually Unbiased Bases, Spherical Designs, and Frames.” SPIE Wavelets XI (2005).
- Hughes, D.; Waldron, S. “Spherical (t,t)-designs with a small number of vectors.” *Linear Algebra and its Applications* 608 (2021), 84–106. DOI: 10.1016/j.laa.2020.08.010.
- Cho, Y.; Saul, L. K. “Kernel Methods for Deep Learning.” NeurIPS 2009.
- Lee, J.; Bahri, Y.; Novak, R.; Schoenholz, S.; Pennington, J.; Sohl-Dickstein, J. “Deep Neural Networks as Gaussian Processes.” ICLR 2018; arXiv:1711.00165.
- Briol, F.-X.; Oates, C. J.; Girolami, M.; Osborne, M. A.; Sejdinovic, D. “Probabilistic Integration: A Role in Statistical Computation?” *Statistical Science* 34 (2019), 1–22; arXiv:1512.00933.
- Kanagawa, M.; Hennig, P.; Sejdinovic, D.; Sriperumbudur, B. K. “Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences.” arXiv:1807.02582.
- Oates, C. J.; Girolami, M.; Chopin, N. “Control Functionals for Monte Carlo Integration.” *JRSS B* 79 (2017), 695–718. DOI: 10.1111/rssb.12185.
