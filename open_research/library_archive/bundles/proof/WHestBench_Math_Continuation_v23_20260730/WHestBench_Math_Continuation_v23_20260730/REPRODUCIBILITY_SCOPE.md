# Reproducibility scope

`python verify_all.py` performs a clean relative-path replay of:

1. the released v21 degree-280 rational certificate against its stored directed interval endpoints;
2. the simultaneous block-trace/atomic-strictness/Sturm checks;
3. exact-rational replay of the T70 headline witness and all T75 sign-count witnesses;
4. the T74 Gaussian-ReLU nonexpansivity formula audit;
5. the parallel exact Gaussian crossing formula audit;
6. the parallel finite-width coefficient-monotonicity counterexample.

The package **does not independently regenerate** the order-320 directed interval endpoints of the depth-32 kernel. That remains an external Arb/FLINT-quality reproduction and human-review gate. The exploratory floating LP is discovery code only; theorem-critical promotion uses rounded rational witnesses and exact replay.
