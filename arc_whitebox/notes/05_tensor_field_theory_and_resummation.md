# Tensor/field-theory routes and a tested hierarchy resummation

This note investigates whether tensor trains, CP/Tucker compression,
Wiener--Hermite chaos, diagrammatic cumulants, free probability, or
resummation can make joint K3/K4 propagation accurate enough at width 256 and
depth 32. FLOPs count a multiply and add as two operations, matching the
challenge convention. The per-MLP budget is `B = 2.72e11`.

## Experimental result: shrink the perturbative hierarchy, do not extrapolate it

The official implementation provides four increasingly rich approximations:
K1-simple, K2-simple, factorized K3-simple, and factorized K3-augment. I treated
these as a finite hierarchy and tested three resummations:

1. scalar Richardson correction from K3-simple to K3-augment;
2. a nonnegative, sum-to-one mixture of all four levels;
3. componentwise stabilized Shanks/Padé extrapolation.

All parameters were fit on official Mini IDs 0--9. Because the nonnegative
mixture cleared a pre-registered 10% gain gate, its parameters were frozen and
IDs 70--79 were opened once.

| method | selection MSE | ratio | frozen holdout MSE | ratio |
|---|---:|---:|---:|---:|
| K3-augment | 7.3509583e-6 | 1.000 | 8.1983853e-6 | 1.000 |
| Richardson | 5.5534607e-6 | 0.755 | 6.4399397e-6 | 0.786 |
| nonnegative hierarchy | **4.4958805e-6** | **0.612** | **5.5690239e-6** | **0.679** |
| best stabilized Shanks | 1.3051178e-4 | 17.75 | not opened | -- |

The frozen mixture is

```text
0.007852 K1 + 0.097752 K2 + 0.186029 K3-simple + 0.708367 K3-augment.
```

The Richardson coefficient is `-0.191114`, so the useful operation is a
shrinkage of the augment correction back toward K3-simple, not an extrapolation
beyond augment. The failed Shanks experiment is also informative: successive
`k` algorithms are not coefficients of one scalar convergent series, and
per-neuron small denominators create poles even after flooring and clipping.

This is a statistically real correction but not yet a legal competitive
submission. The literal prototype runs four propagators, and K3-augment alone
is over budget. The engineering target is therefore to expose lower-order
truncation means during one shared pass, or to distill the correction into a
cheap state-dependent shrinkage.

Reproduction:

- `scripts/eval_hierarchy_resummation.py`
- `scripts/eval_hierarchy_resummation_holdout.py`
- `results/hierarchy_resummation_selection.json`
- `results/hierarchy_resummation_holdout.json`

## Width-256, depth-32 feasibility model

### Dense cumulants

Applying a dense `n x n` weight to all `k` legs of an order-`k` tensor costs
approximately `2 k n^(k+1)` FLOPs per layer.

| state | 32-layer linear-leg cost | budget multiple |
|---|---:|---:|
| dense K3 | 824,633,720,832 | 3.03x |
| dense K4 | 281,474,976,710,656 | 1,034.8x |

Storage is already 16,777,216 entries for K3 and 4,294,967,296 for K4.
Symmetry changes constants, not the conclusion.

### Existing CP carrier

A rank-`R` third-order CP tensor needs three `W @ factor` products, or
`6 n^2 R`. In K3-simple the rank grows by `3n` per layer. The 32 emitted states
sum to 405,504 columns, while the 31 states that are subsequently transformed
sum to 380,928; their linear legs cost 149.79B FLOPs. Adding the K2 carrier
gives the repository's 151.93B lower bound, still excluding nonlinear factor
assembly. K3-augment's 32 emitted states sum to 659,456 columns; transforming
all of them once would already cost 259.31B, leaving almost no budget for its
substantial nonlinear matrix contractions. Its isolated runtime is about ten
times K3-simple, confirming that it is not a practical budgeted path.

Thus CP is the right geometry for exact source-term insertion but unrestricted
rank growth is the bottleneck. Capping rank by stochastic column resampling has
already added about `5e-5` MSE at rank 1,536. Any viable CP method needs
deterministic, observable-aware recompression rather than a globally unbiased
tensor approximation.

### Tucker/shared-subspace carrier

For symmetric Tucker K3 with a shared basis `U in R^(n x r)` and core `G in
R^(r x r x r)`, storage is `nr + r^3` and the linear basis update costs
`2n^2r`:

| rank | state entries | 32-layer basis FLOPs |
|---:|---:|---:|
| 32 | 40,960 | 134,217,728 |
| 64 | 278,528 | 268,435,456 |

Those numbers look excellent, but they omit the hard operation. Elementwise
ReLU Wick multipliers and diagonal source terms do not preserve a shared
subspace. Projection/recompression, and especially the off-subspace residual
needed for later cross terms, dominate accuracy. The earlier full-core plus
unbiased residual prototype was unstable, while shared-subspace shrinkage
helped K3 only when it retained the full CP carrier. Tucker remains promising
only as an **observable-aware** state, not as a global tensor approximation.

### Tensor train

An order-`k` tensor with mode size `n` and uniform TT rank `r` stores roughly
`2nr + (k-2)nr^2` numbers. Applying dense weights to every physical leg costs
about `2kn^2r^2` per layer, with rounding on the order of `knr^3`. For K4:

| TT rank | storage | 32-layer leg transforms | 32-layer rounding |
|---:|---:|---:|---:|
| 16 | 139,264 | 4.295B | 0.134B |
| 32 | 540,672 | 17.180B | 1.074B |
| 64 | 2,129,920 | 68.719B | 8.590B |

The arithmetic can fit for small ranks, but K3/K4 have only three/four modes,
each layer densely rotates every mode, and the ReLU introduces diagonal
couplings across the physical indices. This is not the long, weakly entangled
mode chain where TT is strongest. TT is worth only a small rank-spectrum
diagnostic; it is not the leading implementation path. These storage and
rounding scalings follow the original tensor-train construction and TT-SVD
rounding algorithm: [Oseledets 2011](https://doi.org/10.1137/090752286).

### Wiener--Hermite/polynomial chaos

For 256 Gaussian input variables, a total-degree-`p` chaos basis contains
`binom(256+p,p)` terms:

| degree | coefficients per scalar output |
|---:|---:|
| 2 | 33,153 |
| 3 | 2,862,209 |
| 4 | 186,043,585 |
| 5 | 9,711,475,137 |

Even degree 3 is too large when attached to 256 neurons. ReLU also has an
infinite Hermite expansion and composition grows polynomial degree with depth.
Low-dimensional polynomial chaos can converge rapidly, but its classical
advantage is explicitly for low-dimensional stochastic inputs
([Xiu and Karniadakis 2002](https://doi.org/10.1137/S1064827501387826)).
Here it becomes plausible only after an adaptive active-subspace reduction;
that reduction is precisely the difficult joint-state approximation.

### Diagrams, Edgeworth, and field-theory resummation

Finite-width field theory gives the right *ordering principle*. Yaida derives a
perturbative, renormalization-group-like layer flow for non-Gaussian
finite-width networks
([Yaida 2020](https://proceedings.mlr.press/v107/yaida20a.html)), and
Feynman-diagram power counting can organize higher-order wide-network terms
([Dyer and Gur-Ari 2020](https://arxiv.org/abs/1909.11304)). Fourth-Hermite
finite-size corrections are also the leading Edgeworth term in a shallow
symmetric setting
([Antognini 2019](https://arxiv.org/abs/1908.10030)), while recent
non-asymptotic work proves arbitrary-order multivariate Edgeworth accuracy
under suitable finite-dimensional covariance assumptions
([Celli 2026](https://arxiv.org/abs/2605.24072)).

The limitation is that those theories average over random-network ensembles or
fixed finite sets of observables. This challenge conditions on one realized
weight tensor and asks for 256 neuron-specific means after rank collapse.
Diagram topology reduces the number of algebraic terms, but it does not remove
the neuron-index carrier. The useful translation is a self-consistent
(``dressed'') K2 propagator plus selected K3/K4 vertices, projected onto the
small set of final observables, rather than a global dense cumulant.

### Free probability and dynamical mean field

Free probability accurately describes limiting spectra such as deep-network
Jacobian singular values
([Pennington, Schoenholz, and Ganguli 2018](https://proceedings.mlr.press/v84/pennington18a.html)).
DMFT and its finite-width corrections similarly produce ensemble order
parameters and their fluctuations
([Bordelon and Pehlevan 2023](https://arxiv.org/abs/2304.03408)).
They can estimate response amplification, select ranks, and motivate
self-energy/Dyson resummation. They cannot by themselves recover the
orientation- and neuron-specific mean vector of a fixed realized MLP. Treat
them as diagnostics and priors, not as the estimator.

## Most promising new compressed state

The best tensor/field-theory next experiment is an
**observable-projected skeleton expansion**:

1. run K2 forward to obtain means, covariances, and ReLU Wick gates;
2. run the final-output response basis backward through the gated Jacobians;
3. SVD-compress that 256-output response family to rank `r = 8,16,32`;
4. evaluate only connected K3/K4 diagrams whose external legs lie in that
   response basis, using a self-consistent K2 propagator for internal lines;
5. apply the frozen hierarchy shrinkage to the resulting correction.

The rough carrier arithmetic is `O(L n^2 r)` for forward/backward basis
transport plus `O(L n r^3)` for K3 vertices and `O(L n r^4)` for K4 vertices.
At `r=16`, these are about 0.067B, 0.034B, and 0.537B FLOPs respectively,
before constant/topology factors, leaving substantial room below the 27.2B
10%-score threshold. Unlike global Tucker, this compression optimizes exactly
the contractions that can affect the scored means. Its failure mode is
measurable: if the held-out correction residual outside the response basis
does not fall rapidly with `r`, the tensor route is closed.

## Decisions

- **Promote:** hierarchy shrinkage as a calibration principle; implement
  lower-order taps in a shared pass or distill a local shrinkage.
- **Promote:** observable-projected K3/K4 skeleton expansion, starting with
  rank spectra and K3-only contraction accuracy.
- **Diagnostic only:** free probability/DMFT for response amplification and
  rank selection.
- **Deprioritize:** global tensor train, full polynomial chaos, dense K4.
- **Reject:** componentwise Padé/Shanks on the raw `k` hierarchy.
