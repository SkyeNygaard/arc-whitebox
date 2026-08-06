# Theorem scope matrix

| Result | Width/model | Nodes/support | Weights | Network dependence | Estimator type | Conclusion |
|---|---|---|---|---|---|---|
| T22 | Infinite-width depth-32 ReLU kernel, `d=256` | At most 66,048 arbitrary spherical nodes | Nonnegative, sum one | Fixed or randomized independently of field/network | Linear cubature | Kerdock is within the certified one-sided relative gap of the class infimum. |
| T16 full auxiliary LP | Same limiting kernel | Auxiliary-function LP, not direct node optimization | Nonnegative Gegenbauer coefficients above degree zero | Not applicable | Delsarte certificate | Unique degree-five optimal auxiliary minorant; no higher harmonic improves the bound. |
| T27 | Same limiting kernel | At most `P` symmetrized lines from fixed 33,024-line Kerdock universe | Arbitrary real line weights, sum one | Static/network-independent | Linear cubature | Complete bases plus at most one partial basis attain the exact optimum. |
| Signed beta stability | Same limiting kernel | At most 66,048 consolidated arbitrary nodes | Arbitrary real, sum one, bounded Jordan negative mass | Static or independent randomized | Linear signed cubature | Necessary lower bound depending on negative mass; not sharp enough for closure. |
| Correction-risk | General Hilbert-space random variables | Not a cubature theorem | Not applicable | May be network-specific | Additive correction | Exact risk quadratic and optimal scalar. |
| Replacement | Explicit subspace/replay model | Intermediate state | Not applicable | May be network-specific | State replacement | Downstream-weighted necessary-and-sufficient gate under assumptions. |
| Common-bias | Explicit observation model `Z_i=mu+b+eps_i` | Same-design subestimates | Not applicable | Observables restricted to model | Any measurable estimator from those observations | `mu` and shared `b` are not separately identifiable. |
| Harmonic annihilation | Gaussian radialization + complete Kerdock 5-design | Full Kerdock blocks | Fixed control coefficients within stated class | Coefficients may depend on network but not nodes where stated | Control variates | Named low-degree/homogeneous classes have exactly zero correction. |
