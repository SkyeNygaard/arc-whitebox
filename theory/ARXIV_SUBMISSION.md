# arXiv submission checklist

Status: **not yet submitted.** Everything below is prepared and verified; the
upload itself needs your arXiv account.

## What you upload

**One file: `dist/arxiv_submission_static_cubature.tar.gz`.** Build it from the
repository root, then upload it as the submission's source package. Nothing else
in this repository goes to arXiv.

```bash
python scripts/build_arxiv_bundle.py
```

It is built on demand and not stored in Git (`dist/` is ignored), so if it is
not there, run the command. About 105 KB, 32 files:

```
main.tex          paper source, self-contained (no \input, no graphics)
main.bbl          pre-built bibliography — arXiv does not run BibTeX
references.bib    included for completeness; not required for the build
anc/              29 files, the frozen proof archive, uploaded verbatim
```

`anc/` is arXiv's reserved name for ancillary files. They are published beside
the paper and are not compiled, so nothing in the hashed set is touched.

The build refuses to produce a bundle whose ancillary archive fails its own hash
check, so a successful run is itself the check that the certificates still
replay.

### What not to upload

Three things in this repository look like the submission and are not:

| looks like it | what it actually is |
|---|---|
| [`paper/arxiv_20260803/`](paper/arxiv_20260803/) | the **superseded** 2026-08-03 draft, kept for provenance. Its `main.tex` is a different, shorter paper than [`paper/main.tex`](paper/main.tex). |
| `paper/arxiv_20260803/WHestBench_Full_Paper_Clear_Submission_Ready.{pdf,docx,md}` | compiled output of that older draft. arXiv wants LaTeX source, not a PDF. |
| [`proof_archive/`](proof_archive/) on its own | the ancillary files *before* packaging. Upload them inside the bundle as `anc/`, never as a separate submission. |

The source of truth is [`paper/main.tex`](paper/main.tex). If you are ever
unsure which file is current, rebuild the bundle rather than assembling one by
hand — the script only ever reads `paper/` and `proof_archive/`.

## Verified before packaging

| check | result |
|---|---|
| `anc/scripts/check_package.py` after copying into the bundle | `Package hashes verified.` |
| same check again after extracting the built tarball | `Package hashes verified.` |
| all five certificate checks replayed from inside `anc/` | pass, 40.4 s wall |
| `main.tex` external dependencies | none — no `\input`, `\include`, or `\includegraphics` |
| `main.bbl` present | yes, 13 entries |

The paper source has **not** been compiled locally — there is no TeX
installation on this machine. arXiv's AutoTeX will compile it on upload; if it
reports an error, that is the first thing to look at.

## Suggested metadata

- **Title:** Limits of Static Cubature for Deep ReLU Gaussian Expectations
- **Author:** Skye Nygaard (Independent Researcher)
- **Primary category:** `math.NA` — the result is a lower bound on cubature
  error for a specific kernel and node budget.
- **Cross-list:** `cs.LG` (the application and the audience) and `math.CO`
  (Delsarte linear-programming bounds, Kerdock codes, spherical designs).
  `math.MG` is a defensible alternative to `math.CO` if you would rather sit
  next to the universal-optimality literature.
- **License:** CC BY 4.0, matching the stated intent in
  [`../open_research/LICENSE-DOCS.md`](../open_research/LICENSE-DOCS.md).
- **Comments field:** worth stating the trust boundary here, not only in the
  paper — something like "Computer-assisted; ancillary archive replays the
  rational and integer arithmetic of the certificates. Interval inputs are not
  independently reconstructed."

## After it is submitted

Once you have an arXiv ID, these still say the archive is unsubmitted and
should be updated:

- [`README.md`](../README.md) — "the frozen arXiv ancillary bundle (not yet submitted)"
- [`README.md`](README.md) — the `proof_archive/` bullet, and "the runner as frozen for arXiv"
- [`../whestbench/README.md`](../whestbench/README.md) — the Certificate replay row
- [`../CITATION.cff`](../CITATION.cff) — add the arXiv DOI as a `preferred-citation`
