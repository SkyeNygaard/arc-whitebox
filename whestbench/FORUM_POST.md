# Forum post draft — Challenge Discussion Forum

Post this yourself on the ARC White-Box Estimation Challenge 2026 Discussion
Forum (Discourse category 2991 — new topic:
<https://discourse.aicrowd.com/new-topic?category_id=2991>). Attach
`whestbench/papers/Phase1_Algorithmic_Contribution_320802.pdf`.

**Deadline: 17 August 2026, 23:59 UTC.**

Channel confirmed by `aicrowd_team` in forum topic 18130 (2026-08-10): the
algorithmic-contribution write-up may be emailed to `arc-whestbench@aicrowd.com`
*or* posted on Discourse, and **must include the submission number** it relates
to. Posting publicly rather than emailing only is also what puts the write-up in
front of the community, which is the basis on which discretionary Community
Contribution awards are made (Rules §6: no application process — the Sponsor
reaches out).

---

**Title:** `[Phase 1 write-up] Kerdock/MUB spherical cubature + tracked Winograd propagation — submission #320802, and five negative results`

---

Phase 1 algorithmic-contribution write-up for **submission #320802**
(`skye_nygaard`, graded successfully, 50/50 public MLPs, 0 failures, adjusted
**1.55e-7**, 4.2x the Monte Carlo reference). PDF attached; all code is MIT and
public.

This submission is also one of my two entries nominated for the Phase 1 private
re-evaluation (the other is #321366, adjusted 1.64e-7 — the same estimator at a
worse compute operating point).

I finished #47 of 200 ranked entries, so this is not a leaderboard post. It is a
mechanism post, plus a pile of things that did not work, which I think are the
more useful half.

<!-- CHECK BEFORE POSTING: #47 was the standing at 2026-08-10 14:34 UTC and the
     board was still moving on the final day. Refresh from the closed
     leaderboard before you post. -->


## The mechanism in three sentences

The input distribution is an isotropic Gaussian, so every post-ReLU mean is an
integral over the sphere in dimension 256. I evaluate that integral on a fixed
**66,048-point spherical 5-design** built from the Kerdock code and its
associated maximal real MUBs — 128 Kerdock bases plus the coordinate basis,
every direction carried with its antipode, frozen once as a shipped asset and
reused unchanged for every network. Antipodal closure kills every odd-degree
term for free, so a 5-design costs what a 4-design would.

The part I would actually point at: **the same algebraic object that gives the
accuracy also gives the speed.** The Kerdock bases are chirps acted on by a
Walsh–Hadamard transform, so applying the design to `W_0` is eight butterfly
stages, not an explicit 66,048 x 256 by 256 x 256 product.

## The result I would like people to check

Holding the design, the node set, the radius, and the returned rows exactly
constant on one fixed 100-network development cohort, and changing only the
arithmetic used to propagate:

| arm | raw final MSE | effective compute | budget | adjusted |
|---|---:|---:|---:|---:|
| dense propagation | 2.2826e-7 | 2.689e11 | 98.86% | 2.2565646e-7 |
| tracked depth-5 Strassen–Winograd | 2.2819e-7 | 1.748e11 | 64.27% | **1.4641716e-7** |

Raw MSE is unchanged to three significant figures; the adjusted score improves
**1.5412x**, entirely from the charged-cost term. Depth-5 Winograd charges
`7^5 = 16,807` products where conventional blocking charges `32^3 = 32,768`, a
1.9497x reduction in charged multiplies, and the product is exact rather than
approximate.

Under an effective-compute score, an exact restructuring of the arithmetic was
worth as much to me as a better statistical rule. If you are sitting near the
budget ceiling, that is probably the cheapest headroom available.

## The accounting mistake I made, in case it saves you a day

**Do not select candidates on tracked FLOPs.** Effective compute is
`tracked FLOPs + 1e11 * residual wall seconds`, and the two criteria disagree
by more than the margins people are fighting over.

A streaming variant of my kernel saved exactly the 524,123,904 tracked
operations it was projected to save and cut peak memory by 82.74%, with output
bit-identical to the shipped arm. I still had to reject it: it paid for those
operations with roughly 2.2 s of extra charged residual wall time per network —
about +222e9 effective compute against a 5.241 ms break-even margin — and it
pushed wall time past the 30 s local predict-time limit I was holding myself
to. The FLOP projection was arithmetically correct and the decision it implied
was wrong.

The 30 s is mine, not theirs. The grader's hard cap is **60 s per MLP**. I held
myself to half of it because my machine is faster than the grader, so a local
time reads optimistic. My one calibration point puts the grader about 11%
slower, and the graded #320802 rows land between 26 s and 62 s of wall time, so
the margin was doing real work. If you are timing locally, do not read your own
clock as the grader's.

## Why I stopped looking for a better design

Before any of the competition work I spent a while on the prior question: how
much room does a static rule have at all? For the dimension-256, depth-32
limiting ReLU kernel at a 66,048-node budget:

- among **all** nonnegative mass-one rules on arbitrary spherical nodes,
  complete Kerdock is at most **0.0233242%** above the infimum;
- allowing **arbitrary signed** mass-one weights, no static rule reduces
  Kerdock risk by more than **6.2940%**.

So moving nodes around or reweighting them was close to exhausted before I
started. That is why the remaining budget went into arithmetic rather than into
a cleverer design. The ceiling result is the reason the Winograd work happened
at all. Paper, replayable certificate archive, and an explicit trust boundary
are in `theory/`; five checks replay in about 40 s.

**These are lower bounds on a limiting kernel in a static linear class.** They
say nothing about finite-width, adaptive, nonlinear, or network-dependent
estimators — which is very likely where the leaders here are working. They are
also computer-assisted and replayable, **not** independently verified: the
depth-32 coefficient and curvature interval stack has not been rebuilt in a
second implementation, and human review of the analytic reduction is
outstanding. I would like someone to attack this.

## Negative results

Each of these closes a *tested implementation family*, not a mathematical
class. I am not claiming lower bounds.

1. **Shared-reference Taylor evaluation of a heteroscedastic mixture state.**
   Recentering on the pooled-within covariance shrank covariance offsets a lot
   (K=64, layer 29: 0.574 → 0.357) and did not improve error at all (4.00e-3 →
   5.41e-3). The error is mean-offset dominated, and the offsets are
   structural — increasing K works *by* separating component means, so one
   shared reference necessarily gets worse as the representation gets better.
2. **Low-rank Hermite / direct-diagonal extraction.** Rank-r truncation gave
   relative errors 2.16e-1, 5.44e-2, 6.73e-3, 7.86e-4 at r = 4, 16, 64, 128.
   The accuracy gate is ~1.5e-3, so only r=128 passes; the affordable rank was
   ~4.4. At r=128, `2n^2 r ≈ n^3` — the low-rank route has become the dense
   route.
3. **Partial designs.** Full basis-count curve at 129/96/64/32 bases: adjusted
   ratios 1.0000 / 1.3743 / 1.4096 / 1.5579. MSE scales as roughly `k^-1.21` to
   `k^-1.24`; an exponent above 1 is exactly the condition for the adjusted
   curve to favor the complete design. A partial design only makes sense as a
   cheap host for a correction, and at 96 bases the measured hurdle for such a
   correction was 1.2841x raw gain (vs a 1.0670x projection). Nothing I had
   cleared even the optimistic bar.
4. **Layer-31 anchor corrections.** Propagated centre sits near 0.65% error
   where ~0.45% is break-even after compute cost. ~0.3% of usable headroom is
   not enough for anything I built.
5. **Learned sign/scale models and handcrafted weight features.** No
   complete-score value once actual compute was charged.

## What is public

- `github.com/SkyeNygaard/arc-whitebox` — MIT.
- `arc_whitebox/submissions/production_baseline_320802/` — the estimator behind
  #320802: estimator, Winograd kernel, frozen Kerdock asset, archive
  (sha256 `77be0e88…`, passes `whest validate-package`), plus both development
  result bundles.
- `whestbench/phase1_320802.json` — machine-readable binding record: submission
  id, graded fields, hashes, method, cost, ablations, negative results, limits.
- `scripts/check_competition_release.py` — verifies the public surface, source
  and archive hashes, archive contents.
- `theory/` — a companion lower-bound result on how much room a static cubature
  rule had left at this node budget, with a proof archive whose five checks
  replay in about 40 s. Summarized under "Why I stopped looking for a better
  design" above.

## Caveats I would rather state than have found

- Public-split numbers only; the private 50 are sealed.
- **Layers 1–30 are returned as zeros.** Only layer 0 (closed-form half-normal
  mean) and layer 31 carry estimates. That is a deliberate choice against the
  scored final-layer objective and it is why my all-layers MSE is ~0.74. Read
  as a full activation-profile estimator this submission is poor.
- The published archive is a frozen re-package cut one day after the upload. I
  identify it by its recorded operating point (adjusted 1.55e-7, raw 2.416e-7,
  effective compute 1.745e11, multiplier 0.64154), not by a byte comparison with
  the tarball you hold. That operating point agrees with the graded submission
  page, which reports adjusted 1.550e-7, final-layer 2.416e-7, mean effective
  compute 1.75e11 and 64.19% budget used — agreement to the precision the page
  displays, not an exact field-for-field match on the multiplier. An
  earlier archive in the same lineage shipped a manifest whose declared
  `estimator.py` hash disagreed with the bytes it contained; that is recorded
  in the repo rather than quietly fixed.
- The basis-count curve is an archived exposed-split projection, not an
  official four-arm measurement, and its frozen test weights were float16.

## LLM use

Heavy, and I would rather be blunt about it than hedge. Multiple LLM agents did
ideation, most of the implementation, the experiment scaffolding, proof attempts
in the adjacent theory work, code review, and drafting of the write-up. The
research direction and the promotion/rejection decisions are mine. What I have
personally verified is narrow and listed in §10 of the PDF: archive hashes,
`whest validate-package` results, the graded figures read off the submission
page, and the operating-point match. The mechanistic account is my reading of
code I did not write line by line — consistent with the measured FLOP counts and
frozen manifests, but a reading, not a proof.

Happy to answer questions, and happier to be corrected. If anyone reproduces the
dense-vs-Winograd comparison and gets a different ratio, I would like to know.
