# WHestBench: static cubature limits and competition candidates

This is the canonical public-release surface for the WHestBench research and
competition work in this repository. It separates runnable submission assets,
numerical diagnostics, external-review claims, and local-only research material.

## What this release says

| topic | public status | evidence boundary |
|---|---|---|
| Graded Phase 1 submission #320802 | official grader result | Adjusted `1.55e-7`, 50/50 public MLPs scored, zero failures. Public split only; the 50-MLP private split is sealed until Phase 2 close. See [`phase1_320802.json`](phase1_320802.json) and [`../arc_whitebox/submissions/production_baseline_320802/`](../arc_whitebox/submissions/production_baseline_320802/). |
| Kerdock/MUB reference submission | checked archive | The frozen `kerdock_mub5` archive validates with the public challenge CLI. Its documented `2.25656459e-7` result is limited to the exposed Mini-100 cohort. |
| Winograd-tree candidate | checked archive, reported score | The archive validates, but the later `1.4641716e-7` exposed-Mini score is not independently reproduced from a recovered matching archive and hash. |
| Certificate replay | replayable | The frozen arXiv ancillary proof archive is published in [`../theory/proof_archive/`](../theory/proof_archive/) (not yet submitted to arXiv). Five checks replay in ~40 s and pass in CI. This is exact replay downstream of stored interval inputs, not independent reconstruction. |
| Full experiment record | research record | Every experiment bundle, audit, and ledger row is published in [`../open_research/`](../open_research/) and [`ledger/`](ledger/), including retracted and superseded work. Corrections to that release are in [`../open_research/RECONCILIATION_20260804.md`](../open_research/RECONCILIATION_20260804.md). |
| Nonnegative static-rule theorem | external-review claim | It is scoped to the dimension-256, depth-32 limiting kernel, static network-independent linear cubature, at most 66,048 nodes, and nonnegative mass-one weights. Independent interval reconstruction and human proof review remain open. |
| Signed static-rule ceiling | external-review claim | The audited frozen-witness bound is a fixed-node-budget result, not an equal-FLOPs or equal-wall-time claim. |
| Oracle and negative-result ledger | research record | Entries retain their evidence labels; reported empirical rows are not presented as independently reproduced. |

No protected evaluation was opened during the final research phase covered by
this release. Multiple LLM agents assisted with ideation, proof attempts, code
review, synthesis, and drafting; that assistance is not independent evidence.

## Repository map

- [`../arc_whitebox/`](../arc_whitebox/) contains the small challenge
  submission archives and their source assets, including the graded
  `production_baseline_320802` package.
- [`phase1_320802.json`](phase1_320802.json) binds the graded submission id to
  its archive, hashes, method, cost, ablations, negative results, and limits.
- [`../arc_ceiling/`](../arc_ceiling/) contains standalone numerical diagnostics
  for static cubature geometry.
- [`claims.csv`](claims.csv) is a compact, machine-readable status map for the
  release headlines.
- [`COMPETITION.md`](COMPETITION.md) gives the upload and evidence-capture gate.
- [`RELEASE_STATUS.md`](RELEASE_STATUS.md) lists what is ready now and what must
  remain qualified.
- [`papers/`](papers/) and [`ledger/`](ledger/) explain the manuscript and ledger
  publication boundaries.

## Verify a clean checkout

From the repository root, with the official starter kit installed
(`pip install "git+https://github.com/AIcrowd/whest-starterkit.git"`):

```bash
python scripts/check_competition_release.py
whest validate-package arc_whitebox/submissions/production_baseline_320802/submission.tar.gz
whest validate-package arc_whitebox/submissions/kerdock_mub5/submission.tar.gz
whest validate-package arc_whitebox/submissions/kerdock_mub5_winograd_tree/submission.tar.gz
```

The first command checks the tracked public surface, source/archive hashes,
archive contents, Python syntax, and the absence of oversized or local-only
competition assets. It establishes release consistency, not independent theorem
verification or benchmark reproduction.
