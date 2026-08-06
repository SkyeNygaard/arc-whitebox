# WHestBench white-box competition candidates

This is the small, public-facing subset of a larger local research workspace for
the ARC White-Box Estimation Challenge. It estimates post-ReLU activation means
of a width-256, depth-32 random MLP under the challenge FLOP budget.

The repository tracks no benchmark weights, ground truth, caches, vendor
checkout, or exploratory result bundle. Those materials stay local so that a
fresh clone contains only reviewable code and submission assets.

## Candidates

| candidate | status | evidence boundary |
|---|---|---|
| [`production_baseline_320802`](submissions/production_baseline_320802/) | **graded Phase 1 submission** | The estimator behind AIcrowd submission `#320802`: adjusted `1.55e-7`, 50/50 public MLPs, zero failures. The archive validates with the public CLI. It is identified by its recorded operating point, not by a byte comparison with the tarball AIcrowd holds. |
| [`kerdock_mub5`](submissions/kerdock_mub5/) | validated reference archive | The archive is structurally valid and carries the frozen 66,048-node Kerdock/MUB 5-design baseline. Its documented all-100 exposed-Mini result is `2.25656459e-7` adjusted. |
| [`kerdock_mub5_winograd_tree`](submissions/kerdock_mub5_winograd_tree/) | research candidate | Its archive is structurally valid. The `1.4641716e-7` all-100 exposed-Mini figure is **reported**, not independently reproduced from a recovered matching archive and hash. |

The baseline is the safe competition reference. Do not describe the Winograd-tree
figure as a verified result until the exact shipping archive, SHA-256, and fresh
full exposed-Mini output have been captured together.

## Validate an archive

Install the official public challenge CLI, then run:

```bash
whest validate-package submissions/kerdock_mub5/submission.tar.gz
whest validate-package submissions/kerdock_mub5_winograd_tree/submission.tar.gz
```

The tracked release check also verifies archive contents and SHA-256 values:

```bash
python ../scripts/check_competition_release.py
```

The complete release policy, evidence labels, and competition handoff are in
[`../whestbench/`](../whestbench/). The ignored root-level `estimator.py` is
quarantined: it is not a candidate, is not part of the release, and must not be
selected as a fallback.
