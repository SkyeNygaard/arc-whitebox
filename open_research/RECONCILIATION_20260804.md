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
outside this repository, in `../whestbench-private/outreach/`.

The remaining review materials — the external review packet, reviewer
questions, and the two-page overview — name no third parties and are published.

Bulk numerical arrays (`.npz`, `.npy`, `.pt`, `.joblib`) were also excluded on
size grounds. They are not in Git, but they are in the working tree beside the
bundles they belong to. Everything else came across: all 35 experiment bundles
and all proof bundles.

## 7. Correction, 2026-08-11: the source zips were not size exclusions

The paragraph above originally read "and one 8.5 MB source zip". That was wrong
twice over, and the error is worth stating rather than editing away.

The 8.5 MB archive it meant, `arc_code.zip`, was never missing — it is at the
repository root, excluded from Git by name in [`../.gitignore`](../.gitignore)
and present in the working tree the whole time.

What *was* missing was four different archives, in
`library_archive/bundles/experiment/WHestBench_Experiment_Launch_Pack_20260729_v2/.../sources/`:

| archive | scripts | result records | notes |
|---|---|---|---|
| `FULL_MULTILEVEL_CONTINUATION_BUNDLE.zip` | 15 | 53 | 2 |
| `activation_region_continuation_bundle.zip` | 10 | 13 | 5 |
| `agent1_layer31_deployability_round.zip` | 4 | 8 | 2 |
| `equivariant_weight_model_repro_20260729.zip` | 7 | 12 | 2 |

Together they hold 36 Python scripts, 86 result records, and their notes — 133
files, about 1 MB. At least 114 of those 133 appear nowhere else in this
repository, even by filename alone. They are not
bulk arrays, and the size rule never applied to them. They were dropped by
accident. They are now restored, and `.gitignore` already says that an
experiment you cannot find is a bug in that file rather than a deliberate
omission — this was one.

Three further archives remain excluded, and that is deliberate:
`whestbench_t22_t23_dual_engine_release_v5_2_20260730.zip`,
`Agent5_Competition_Opportunity_Experiments_20260730.zip`, and
`arc_cubature_proof_v5_1.zip`. Their contents were checked file by file against
this repository on 2026-08-11: everything inside them is already published
unzipped in the same bundle, except compiled-Python caches and one classifier
(`anchor_rf_classifier.joblib`, kept in the working tree under the bulk-array
rule).

### Also recovered on 2026-08-11

From the pre-monorepo working folder, all verified byte-identical to their
originals:

- [`../theory/paper/arxiv_20260803/`](../theory/paper/arxiv_20260803/) — the
  2026-08-03 arXiv submission source and the compiled full paper. Note that
  `../theory/paper/main.tex` is a *later* revision; the two share a filename and
  differ in content.
- [`evidence/02_Evidence_Manifest.csv`](evidence/02_Evidence_Manifest.csv) — 30
  rows mapping each public claim to its evidence, approved wording, and the
  specific overclaim to avoid.
- [`ledger/history/`](ledger/history/) — the intermediate canonical ledgers
  (v5, v10, v14, v15, v26, v28), the two experiment-ledger CSVs, and the
  research catalogs. The reconciled v31 in [`ledger/`](ledger/) remains the
  authoritative one; these are the working snapshots behind it.
- [`library_archive/reports/WHestBench_Subagent_Handoffs_v21.md`](library_archive/reports/WHestBench_Subagent_Handoffs_v21.md)
- [`audit/SHA256SUMS_WHestBench_20260802_AUDITED_V2.txt`](audit/SHA256SUMS_WHestBench_20260802_AUDITED_V2.txt)
  — checksums for two archives that live outside this machine.
- `Makefile` — `make check` and `make manifest`, both of which call scripts that
  were already here.
