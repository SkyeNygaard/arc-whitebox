# Competition handoff

## Upload gate

1. Run `python scripts/check_competition_release.py` from the repository root.
2. Run `whest validate-package` on the exact archive that will be uploaded.
3. Record the archive SHA-256, public CLI version, run command, dataset split,
   result JSON, and validation output together.
4. Upload only after the full exposed-Mini run finishes with no failures.
5. Keep protected evaluation sealed unless the competition rules explicitly
   authorize access.

## Graded Phase 1 submission

`arc_whitebox/submissions/production_baseline_320802/submission.tar.gz`
is the estimator behind AIcrowd submission **#320802** — adjusted `1.55e-7`,
50/50 public MLPs scored, zero failures, graded successfully. Its SHA-256 is
`77be0e8865b2aeee6c6c16314cac4d38496efefed6b2b758f75bc3033bb6b7bc` and it
passes `whest validate-package`. The full binding record, including the
identification basis and its limits, is in
[`phase1_320802.json`](phase1_320802.json).

This is the submission the Phase 1 algorithmic-contribution write-up is tied
to. ARC already holds the uploaded tarball for that id; no new code upload is
needed or possible.

## Candidate status

`arc_whitebox/submissions/kerdock_mub5/submission.tar.gz` is the checked
reference archive. Its SHA-256 is
`e60c0a686188f9fe030c1a3769b29859d539902d9a43be40e4b6f9883dd663ae`.

`arc_whitebox/submissions/kerdock_mub5_winograd_tree/submission.tar.gz` is a
checked archive with SHA-256
`a7f5e1e58639192e33e0886e776b4c8392399a7879e372bed557811516ec93e7`.
Its later all-100 score is reported only; recreate it from that exact archive
before using it in a leaderboard, paper, or public comparison.

The ignored root `estimator.py` is quarantined because it is reported to fail
under the current FlopScope API. It must not enter packaging, automated candidate
selection, or fallback logic.

## Results language

Use “validated archive” only after `whest validate-package` passes for the exact
archive. Use “reported exposed-Mini result” for the `1.4641716e-7` figure until
the archive, SHA-256, result JSON, and rerun are all tied together. Never call an
exposed-Mini result a protected, independently certified, or competition-final
score.
