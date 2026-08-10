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

Everything from the `---` down is the post itself. Do not paste this header.

---

**Title:** `[Phase 1 write-up] A fixed grid of directions instead of random sampling — submission #320802, and five things that did not work`

---

Phase 1 algorithmic-contribution write-up for **submission #320802**
(`skye_nygaard`, graded successfully, 50/50 public MLPs, 0 failures, adjusted
**1.55e-7**, 4.2x better than the Monte Carlo reference). PDF attached; all code
is MIT and public.

This is also one of my two entries nominated for the Phase 1 private
re-evaluation. The other is #321366, adjusted 1.64e-7 — the same estimator, just
using more compute.

I finished #47 of 200 ranked entries, so this is not a leaderboard post. It is a
post about a mechanism, plus a pile of things that did not work, which I think
are the more useful half.

I have tried to write this so you can follow it without knowing anything about
cubature or coding theory. Terms are explained the first time they appear.

<!-- CHECK BEFORE POSTING: #47 was the standing at 2026-08-10 14:34 UTC and the
     board was still moving on the final day. Refresh from the closed
     leaderboard before you post. -->


## The problem, in plain terms

You are handed the weights of a neural network and asked: on average, how
strongly does each neuron fire when you feed the network random inputs?

The obvious answer is to try it. Feed in thousands of random inputs, run the
network, average what you see. That is Monte Carlo sampling, and it works, but
it converges slowly — to halve your error you need four times as many samples.
The competition gives you a compute budget about 15,000x smaller than the
reference answer was computed with, so simply sampling harder is not available.

So you have to use the weights themselves. That is the whole challenge.

## What I did

**Replace random sampling with a fixed, carefully chosen set of directions.**

The inputs are drawn from a standard Gaussian — a bell curve that looks the same
in every direction, with no preferred axis. The network has no bias terms, and
ReLU has the property that scaling the input scales the output by the same
factor. Put those together and only the *direction* of the input matters, not
its length. The question "what is the average activation?" becomes "what is the
average over all directions?" — an average over the surface of a sphere in 256
dimensions.

Now, averaging over a sphere is a problem with a long history. You do not need
random points. You can use a fixed list of directions chosen so that their
average is very close to the true average — the same idea as Simpson's rule for
ordinary integrals, but on a sphere. Such a list is called a **cubature rule**.

I used a list of **66,048 directions** with a property called being a **spherical
5-design**: averaging any polynomial of degree 5 or less over these directions
gives *exactly* the right answer, not an approximation. The construction comes
from the **Kerdock code**, an object from coding theory, together with a related
family of bases. Concretely: 128 Kerdock bases plus the ordinary coordinate
basis, 256 directions each, and every direction paired with the direction
opposite to it.

That pairing is free accuracy. Including every direction alongside its opposite
automatically cancels all the odd-degree error terms, so a 5-design costs no
more than a 4-design would.

The list is chosen once, frozen into a file shipped with the submission, and
reused unchanged for every network. It never looks at the weights.

**The part I would actually point at: the same object that gives the accuracy
also gives the speed.**

Applying 66,048 directions to a 256x256 weight matrix should mean a large matrix
multiply. It does not, because Kerdock bases are built from a structured pattern
that can be applied by a **Walsh–Hadamard transform** — a fast transform in the
same family as the FFT, which does the job in eight cheap passes instead of one
expensive multiplication. The accuracy comes from the geometry of the point set;
the speed comes from the algebra underneath it. They are the same object.

## The result I would like people to check

The score is not raw error. It is error multiplied by how much of the compute
budget you burned. So there are two ways to improve it: be more accurate, or be
cheaper. This experiment changed *only* the cost side.

I held everything statistical fixed — same 66,048 directions, same radius, same
rows returned, same 100 development networks — and changed only the arithmetic
used to push the directions through the network's layers:

| arm | raw final MSE | effective compute | budget | adjusted |
|---|---:|---:|---:|---:|
| ordinary matrix multiplication | 2.2826e-7 | 2.689e11 | 98.86% | 2.2565646e-7 |
| Strassen–Winograd, depth 5 | 2.2819e-7 | 1.748e11 | 64.27% | **1.4641716e-7** |

Raw error is unchanged to three significant figures. The score improves
**1.5412x**, entirely because the second arm is cheaper.

Strassen–Winograd is a classical trick for multiplying matrices with fewer
multiplications than the schoolbook method — and it gives the *exact* same
answer, not an approximation. Applied five levels deep it charges
`7^5 = 16,807` multiplications where ordinary blocking charges `32^3 = 32,768`,
a 1.9497x reduction.

The takeaway: when your score charges you for compute, rewriting the arithmetic
was worth as much to me as finding a better statistical method. If you are
pressed against the budget ceiling, that is probably the cheapest headroom
available to you.

## The accounting mistake I made, in case it saves you a day

**Do not choose between candidates by counting FLOPs.**

The competition charges you `tracked FLOPs + 1e11 * residual wall seconds`.
"Tracked FLOPs" are arithmetic operations the instrumentation counts. "Residual
wall time" is real elapsed seconds spent outside those counted operations —
memory allocation, copying, Python overhead — converted back into a compute
charge at a punishing rate. Optimising one of these can quietly wreck the other.

I built a streaming version of my kernel. It saved exactly the 524,123,904
counted operations it was projected to save and cut peak memory by 82.74%, and
its output was bit-for-bit identical to the version I shipped. I still had to
throw it away. Those savings cost about 2.2 extra seconds of real time per
network, which converts to roughly +222e9 on the compute charge — against a
break-even margin of 5.241 ms. The FLOP arithmetic was correct and the decision
it pointed to was wrong.

The 30 s limit I mention below is mine, not the organizers'. The grader's hard
cap is **60 s per MLP**. I held myself to half of it because my machine is
faster than the grader, so a local timing reads optimistically. My one
calibration point puts the grader about 11% slower, and the graded #320802 rows
land between 26 s and 62 s, so the margin was doing real work. If you are timing
locally, do not read your own clock as the grader's.

## Why I stopped looking for a better set of directions

Before any of the competition work I asked the prior question: how much room is
there in this approach at all? If I had spent the whole competition hunting for
a cleverer set of directions, how much better could I possibly have done?

The answer turned out to be: almost none. For this specific problem — 256
dimensions, 32 layers, a budget of 66,048 directions — I proved two bounds:

- Against **every** possible choice of directions with positive weights, the
  Kerdock set is at most **0.0233242%** worse than the best one that exists.
- Even allowing negative weights, no fixed rule can beat Kerdock by more than
  **6.2940%**.

So the set of directions was essentially already optimal, and moving points
around or reweighting them was a dead end before I started. That is why the
remaining budget went into the arithmetic instead. The ceiling result is the
reason the Strassen–Winograd work happened at all.

The paper, a replayable proof archive, and an explicit statement of what the
proof does and does not cover are in `theory/`. Five checks replay in about 40 s.

**Two limits on this, stated plainly.** First, these bounds apply to *fixed*
rules — a set of directions chosen in advance, the same for every network. They
say nothing about methods that look at the specific network they were given,
adapt as they go, or combine results non-linearly. That is very likely where the
leaders here are working, and nothing I proved constrains them. Second, the
proof is computer-assisted and replayable but **not** independently verified: a
key stack of numerical intervals has not been rebuilt by a second
implementation, and nobody has reviewed the analytic reduction by hand. I would
like someone to attack this.

## Five things that did not work

Each of these rules out *a specific thing I built and measured*, not a whole
mathematical class. I am not claiming these are impossible.

1. **Tracking the spread of activations, not just the average, from one shared
   reference point.** As signals pass through layers they spread out into
   something like a mixture of several clusters. I tried describing all of them
   relative to one common centre. Measuring from the pooled centre did shrink
   the spread terms a lot (0.574 → 0.357 at layer 29), and error got *worse*
   anyway (4.00e-3 → 5.41e-3). The reason is structural: the error is dominated
   by how far apart the cluster centres are, and the method only improves by
   separating those centres further. It necessarily degrades as the
   representation gets better.

2. **Compressing the layer-to-layer transformation to a low-rank
   approximation.** Keeping only the r most important components gave relative
   errors of 2.16e-1, 5.44e-2, 6.73e-3 and 7.86e-4 at r = 4, 16, 64 and 128. The
   accuracy I needed was about 1.5e-3, so only r=128 was good enough — but the
   budget only afforded about r=4. And at r=128 the "compressed" version costs
   the same as the uncompressed one. The shortcut had become the long way round.

3. **Using fewer directions.** Dropping from 129 bases to 96, 64 and 32 made the
   score worse by 1.3743x, 1.4096x and 1.5579x respectively. The reason is that
   error grows roughly as `k^-1.21` to `k^-1.24` in the number of bases `k`, and
   any exponent above 1 means error rises faster than the compute saving falls.
   The full set always wins. A reduced set is only worth it as a cheap base for
   some correction on top — and at 96 bases that correction would have needed to
   deliver a 1.2841x accuracy gain to break even, against a projection that said
   1.0670x would do. Nothing I had came close to either number.

4. **Correcting the final layer against an anchor point.** The value I propagate
   sits about 0.65% off, and about 0.45% of that is eaten by the cost of
   computing a correction. Roughly 0.3% of usable headroom is not enough to
   build anything on.

5. **Learned correction models and hand-designed features of the weights.**
   Several variants. None of them paid for their own compute.

## What is public

- `github.com/SkyeNygaard/arc-whitebox` — MIT.
- `arc_whitebox/submissions/production_baseline_320802/` — the estimator behind
  #320802: the estimator, the fast matrix-multiply kernel, the frozen direction
  set, and the submitted archive (sha256 `77be0e88…`, passes
  `whest validate-package`), plus both development result bundles.
- `whestbench/phase1_320802.json` — a machine-readable record tying the
  submission id to its graded numbers, file hashes, method, cost, experiments,
  negative results and limitations.
- `scripts/check_competition_release.py` — verifies the published files and
  their hashes.
- `theory/` — the paper and proof archive behind the ceiling result above.

## Caveats I would rather state than have you find

- All numbers here are from the public 50 networks. The other 50 are sealed.
- **Layers 1 to 30 are returned as zeros.** Only layer 0 and layer 31 carry real
  estimates. Layer 0 has an exact closed-form answer; layer 31 is the one that
  gets scored. This is a deliberate choice to chase the scored objective, and it
  is why my all-layers error is about 0.74 while my final-layer error is
  2.4e-7. Judged as a general tool for estimating a whole network's activation
  profile, this submission is poor. Judged as a final-layer estimator, it is
  what it claims to be.
- The archive I published is a repackage made one day after the upload. I
  identify it by its recorded operating point (adjusted 1.55e-7, raw 2.416e-7,
  effective compute 1.745e11, multiplier 0.64154), not by comparing bytes with
  the tarball ARC holds. That operating point agrees with the graded submission
  page, which shows adjusted 1.550e-7, final-layer 2.416e-7, effective compute
  1.75e11 and 64.19% budget used — agreement to the precision the page displays,
  not an exact match on the multiplier. An earlier archive in the same lineage
  shipped a manifest whose declared `estimator.py` hash did not match the bytes
  beside it. That is recorded in the repo rather than quietly fixed.
- The reduced-direction comparison in point 3 is an archived projection from the
  development split, not an official four-way measurement, and its frozen test
  weights were float16.

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
ordinary-vs-Winograd comparison and gets a different ratio, I would like to know.
