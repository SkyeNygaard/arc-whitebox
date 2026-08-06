# Reconciliation, 2026-08-04

`open_research/` was assembled on 2026-08-02 from an accessible subset of the
project archive. When it was merged into this repository on 2026-08-04, several
of its "missing artifact" entries turned out to be recoverable from the working
tree, and one important thing was absent from it entirely: the release predates
the competition submission being identified.

This file records what changed and what did not. The original documents are
**not** edited — [`RELEASE_STATUS.md`](RELEASE_STATUS.md) and
[`BASELINE_PACKAGE_MISSING.md`](BASELINE_PACKAGE_MISSING.md) are preserved as
written, because a research release that quietly rewrites its own provenance is
worth less than one that shows its corrections. Read them together with this
file.

Every hash below was recomputed on 2026-08-04 from the files in this
repository.

---

## 1. The release predates the graded submission

`README.md` says the release is "not a claim of a new benchmark submission" and
`RELEASE_STATUS.md` concludes "No new deployable estimator is included."

That was true of the theory release. It is not true of the project. AIcrowd
Phase 1 submission **#320802** was submitted 2026-07-29 13:03, graded
successfully, and scored adjusted **1.55e-7** on all 50 public MLPs with zero
failures — 4.2x the Monte Carlo reference.

The estimator, its archive, its hashes, and its evidence boundary are in
[`../arc_whitebox/submissions/production_baseline_320802/`](../arc_whitebox/submissions/production_baseline_320802/)
and [`../whestbench/phase1_320802.json`](../whestbench/phase1_320802.json).

## 2. The "missing" 129-basis package is present

`BASELINE_PACKAGE_MISSING.md` reports that the exact final 129-basis package
could not be found. It is present, and its manifest authenticates it.

| file | sha256 |
|---|---|
| `whestbench_final_129basis_20260730.tar.gz` | `c01c7da4e8737e189ab4f9b1eccce2b90dbde17fb5d085319f07d30d57ac444c` |
| ↳ `estimator.py` | `7955fae1fc3bb956f9a3ca2754befbe1eef3b39b99ab6e703f43ab8aa3dfb139` |
| ↳ `fast_matmul.py` | `fb1b93cb625b66ce5f26220ea3b6b685dbb9887d50f8756cafa9426577d45085` |
| ↳ `kerdock_mub5_seed3.npz` | `58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad` |

The archive contains six files (`README.md`, `SHA256SUMS.txt`, `estimator.py`,
`fast_matmul.py`, `kerdock_mub5_seed3.npz`, `manifest.json`), and the three
hashes its `manifest.json` declares match the three files it ships. The Kerdock
asset matches the hash `BASELINE_PACKAGE_MISSING.md` itself lists as safe to
state.

It is not tracked in Git — it embeds a 250 KB binary design asset already
published inside the submission packages — but it is in the working tree and
its identity is now recorded here.

**The more useful correction:** this package was never submitted. It was built
2026-07-30, one day *after* the last Phase 1 submission. The authentication
effort in `BASELINE_PACKAGE_MISSING.md` was aimed at the wrong artifact.

## 3. The "missing" official Mini-100 JSON is present

`RELEASE_STATUS.md` lists "Official Mini-100 JSON | Missing | Aggregate values
reported only."

The full run record is at
[`../arc_whitebox/submissions/production_baseline_320802/official_129basis_mini100_20260731.json`](../arc_whitebox/submissions/production_baseline_320802/official_129basis_mini100_20260731.json)
— 645 KB, all 100 networks, per-layer MSE, per-network FLOP and wall-time
breakdowns down to individual operation counts, `whestbench` 0.13.0,
`flopscope` 0.9.1, dataset SHA-256
`5b00938b6bd809fe80acef08772c5654edf467863225ca9e304b76c779ecf433`, adjusted
`1.4641716e-7`, 0/100 failures.

Note this is the exposed **Mini development split**, a different set of
networks from the graded AIcrowd cohort. Compare the network names:
`daniel-harrison` here, `patricia-hawkins` there. The two must not be chained.

## 4. The "mismatched manifest" is real, and was fixed downstream

`BASELINE_PACKAGE_MISSING.md` warns that "the accessible
`production_partial_tree_source` package has a mismatched estimator manifest."
That is accurate about that archive, and the warning not to substitute it was
the right instinct. Two things refine it.

First, the mismatch is in the *manifest*, not the code. The package's
`estimator.py` (`f1e32ce4…`) and `fast_matmul.py` (`fb1b93cb…`) are the correct
production files; its `manifest.json` (`975409ee…`) declared hashes that did
not match them.

Second, that defect was already corrected. The T0 instrumentation bundle
re-froze the same three source files under a manifest whose declared hashes do
match the shipped members, and recorded the original mismatch rather than
hiding it in `manifest_actual.json`. The result is
`production_baseline.tar.gz`, SHA-256
`77be0e8865b2aeee6c6c16314cac4d38496efefed6b2b758f75bc3033bb6b7bc`. Its own
`PACKAGE_AUDIT.json` records `manifest_hashes_ok: true`, and `whest
validate-package` passes on it. That is the archive published in
[`../arc_whitebox/submissions/production_baseline_320802/`](../arc_whitebox/submissions/production_baseline_320802/).

The hash recorded as the mismatching estimator,
`7c3fcc2ac542bda41ab568e62428ec75b7edec6f146d61a036c0710d9ee49694`, belongs to
the earlier `kerdock_mub5_winograd_tree` estimator — the build *before* the
chunked final layer.

## 5. What is still genuinely missing

Searched across both the competition working tree and the imported archive on
2026-08-04, and not found:

- **Mixture K ladder** (M205 / T106) — scripts, arrays, and metric metadata.
- **Pooled-within Taylor recentering** — reported values only; no script.
- **Direct / Hermite rank sweep** (M206 / T107) — the cost arithmetic
  reproduces, the relative-error table does not have a runnable source.
- **Root package failure bundle** — the reported 2/2 smoke failure under the
  current FlopScope API is documented but the run artifacts are absent.
- **Independent reconstruction of the depth-32 coefficient and curvature
  interval stack** — an external gate, unchanged. See
  [`../theory/README.md`](../theory/README.md).

These stay marked missing. The negative results that depend on them are
labeled as archived or reported wherever they appear, including in the
competition write-up.

## 6. What was removed on import

Three files were excluded from `review/`: `Northeastern_Outreach_Revised.md`,
`.docx`, and `.pdf`. They are an outreach draft naming three individuals and
carrying their institutional email addresses. Those people did not agree to
appear in a public repository, and the project's own publication policy already
said outreach drafts and reviewer contact lists do not belong here. They remain
in the local copy.

The remaining review materials — the external review packet, reviewer
questions, and the two-page overview — name no third parties and are published.

Bulk numerical arrays (`.npz`, `.npy`, `.pt`, `.joblib`) and one 8.5 MB source
zip were also excluded on size grounds. Everything else came across: 3,916
files including all 35 experiment bundles and all proof bundles.
