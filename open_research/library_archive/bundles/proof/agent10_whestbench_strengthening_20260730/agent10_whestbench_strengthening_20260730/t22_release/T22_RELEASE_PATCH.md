# T22/T23 release patch

**Status:** The mathematical certificate is retained. This patch removes the stale two-sided interpretation and incorporates the hostile reproducibility audit’s release-hygiene corrections.

## Canonical semantic correction

The old machine-readable object reported a narrow positive interval for:

- additive suboptimality;
- Kerdock/optimum ratio;
- relative excess;
- relative excess percent.

Those intervals describe the difference between Kerdock’s certified upper bound and the auxiliary certificate expression. They are **not** lower bounds on Kerdock’s actual distance from the infimum. Kerdock itself is feasible, so the mathematically valid intervals are:

- actual additive suboptimality: `[0, 5.685041020616819...e-11]`;
- actual multiplicative ratio: `[1, 1.0002336550102948...]`;
- actual relative excess: `[0, 0.0002336550102948...]`;
- actual relative excess percent: `[0, 0.0233655010294814...]`.

The canonical file is:

`FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32_CANONICAL.json`

The validator `validate_t22_one_sided.py` rejects the stale artifact and accepts the canonical one.

## T23 release-hygiene corrections

1. Replace “31 deterministic files” with **32 manifest-tracked canonical files**.
2. State that **23 intermediate curvature chunks** are regenerated and reassembled but are not individually listed in the primary proof manifest.
3. Replace “immutable manifest” with **fixed during verification** unless the archive hash is externally signed or anchored.
4. Publish the exact source-archive SHA-256 alongside the release.
5. Pin the tested environment and add a second operating-system/Python-version CI run.
6. Do not call the theorem “formally verified”; call it **computer-assisted certified**.

## Paper-ready wording

> A clean-room hostile audit regenerated all 32 manifest-tracked canonical outputs and all 23 intermediate curvature chunks byte-for-byte. The exact-rational witness, directed-rounding interval coverage, curvature sign chain, spherical kernel mean, Kerdock multiplicities, and outward-rounded one-sided ratio were independently checked. The result is a rigorous computer-assisted proof within its explicit trust base, not a proof-assistant formalization. The theorem is one-sided: Kerdock’s true excess over the infimum may be zero.

## Scope guard

The theorem covers static or independently randomized nonnegative-weight linear rules with at most 66,048 nodes under the dimension-256, depth-32 infinite-width ReLU kernel. It does not cover signed weights, finite width, network-adaptive points or weights, pilot adaptation, or nonlinear estimators.
