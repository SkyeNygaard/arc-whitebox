# WHestBench Current State v31 — Final Local Write-up

**Audit cutoff:** 2026-07-31 16:42 ET  
**Canonical ledger:** `whestbench_canonical_research_ledger_20260731_reconciled_v31_final_local_writeup.xlsx`  
**Protected evaluation opened:** No

## Executive verdict

The final local write-up materially strengthens the closure of the current analytic-mixture program and validates the shipping package through the official exposed harness.

I agree with four central conclusions:

1. the shared-reference first/second-order Taylor evaluation family is closed for a structural reason;
2. the tested low-rank Hermite/direct-diagonal route is closed by an accuracy–rank–cost squeeze;
3. the shipped 129-basis estimator is the only currently validated runnable package;
4. the root `estimator.py` must be quarantined because it crashes under the current FlopScope API.

I do **not** agree with the unrestricted statements that this was “the last untested space,” that every analytic mixture or compact joint-state representation is closed, or that writing the paper is the only rational next action.

The defensible conclusion is narrower:

> The current high-accuracy heteroscedastic-mixture representation has no affordable evaluator among exact componentwise propagation, shared-reference Taylor approximation, and the tested low-rank Hermite/direct-diagonal construction.

One special mixture structure remains insufficiently isolated: an exact tied/shared-covariance recurrence that can reuse the correlation structure across components. That deserves one independent, tightly scoped test. If it fails, M192 should be closed.

After that, the highest-upside remaining candidate-class escape is M194: one exact, weight-coupled boundary/Walsh phase identity.

## New mixture-evaluation evidence

### T106 / M205 — Taylor approximation is mean-offset dominated

The local model tested whether expanding component covariance maps around the pooled-within covariance could repair the second-order Taylor approximation.

At \(K=64\):

| Layer | Covariance-offset norm, global | Pooled within | Error, global | Pooled within |
|---:|---:|---:|---:|---:|
| 16 | 0.586 | 0.476 | 5.39e-3 | 5.58e-3 |
| 29 | 0.574 | 0.357 | 4.00e-3 | 5.41e-3 |

The covariance offsets shrink substantially, but the approximation error does not improve.

This strongly supports the claimed mechanism:

- increasing \(K\) works by separating component means;
- the component-mean offsets are therefore structural;
- one shared reference becomes increasingly inaccurate as the representation becomes more expressive;
- a hierarchy of references must become dense in mean space, driving the number of references toward the number of components.

I agree that this closes the tested shared-reference first- and second-order Taylor family.

It does not prove that every non-expansion evaluator is impossible.

### T107 / M206 — low-rank direct extraction degenerates to exact cost

The direct/Hermite route asks for the next-layer variances without explicitly materializing every full ReLU covariance matrix.

Measured relative errors from truncating the component covariance to rank \(r\):

| Layer | r=4 | r=16 | r=64 | r=128 |
|---:|---:|---:|---:|---:|
| 16 | 2.16e-1 | 5.44e-2 | 6.73e-3 | 7.86e-4 |
| 29 | 1.74e-1 | 4.00e-2 | 3.76e-3 | 2.23e-4 |

The stated accuracy gate is approximately \(1.5\times10^{-3}\), so only rank 128 passes.

The affordable rank under the local cost budget was estimated as \(r\leq4.4\). At rank 128,

\[
2n^2r \approx 1.7\times10^7 \approx n^3,
\]

so the low-rank computation becomes the full dense computation in cost.

I agree that this closes the tested low-rank Hermite/factorization implementation.

I would not present it as an information-theoretic lower bound on every possible algorithm for the diagonal of \(W^\top C W\).

### M207 — current heteroscedastic-mixture program

The final reported picture is:

| Evaluator | Reported cost | Failure |
|---|---:|---|
| Exact componentwise | 8.68e7 per component | Budget permits only about K=100 |
| Shared-reference Taylor | 2.6e5 per component | Mean offsets grow with K |
| Direct/Hermite plus low-rank covariance | approximately \(n^3\) at required rank | Rank 128 needed |

The representation itself reportedly reaches error \(\delta\approx1.781\times10^{-3}\) at \(K=1536\). Thus representation accuracy exists, but its current evaluator does not.

This is a genuine accuracy–representation–evaluation squeeze.

Canonical interpretation:

- close larger-\(K\) heteroscedastic full-covariance searches;
- close additional Taylor centers and covariance-rank sweeps;
- do not claim all compact analytic states are impossible;
- permit one exact tied/shared-covariance recurrence test exploiting reusable correlation structure.

## Official shipping result

### M208 — official exposed Mini benchmark

The write-up reports a full `whest` subprocess run on all 100 Phase-1 Mini networks:

- adjusted final-layer score: `1.4641716e-7`;
- raw MSE: `2.2819432e-7`;
- mean multiplier: `0.6427`;
- effective compute: `1.748e11`;
- estimator FLOPs: `1.70873e11` per MLP;
- failures: `0 / 100`.

The local prediction of raw MSE was `2.2826e-7`, agreeing within approximately 0.03%.

Subject to independently checking the referenced JSON and package hashes, this is strong validation that the local propagation harness and the shipped estimator agree end to end.

The 129-basis package remains the only validated runnable candidate.

### Timing correction

Observed wall time was reportedly 16.5 seconds per MLP rather than 21.4 seconds.

Applying the stated 11% grader penalty gives approximately 18.3 seconds against the 30-second guard, or about 39% headroom.

Therefore timeout insurance is optional rather than clearly positive expected value. It should be judged only by its adjusted-score cost.

### M209 — root estimator is broken

The root `estimator.py` reportedly fails with:

```text
TypeError: dot() got an unexpected keyword argument 'out'
```

under FlopScope 0.9.1.

Every smoke-test MLP failed, forcing the zeros-baseline result near 0.83.

This package must be quarantined from:

- submission scripts;
- candidate comparisons;
- automatic packaging;
- fallback-selection logic.

A future port must use a new package identity and receive a complete official-like rerun.

## Tail concentration

The reported worst official-Mini network error is `8.52e-7`, about 5.8× the mean, and contributes roughly 6% of total 100-network loss.

That is worth one mechanism audit, but it is unlikely to supply a breakthrough by itself.

The audit must:

- use grouped development networks;
- avoid tuning to the single worst network;
- identify a legal network-independent mechanism;
- report both mean and upper-tail adjusted score;
- preserve the exact shipping package until the alternative passes.

## Where I agree with the final write-up

I agree that:

- the current heteroscedastic analytic-mixture route is now strongly squeezed;
- Taylor recentering fails for a structural mean-spread reason;
- the tested low-rank diagonal extraction loses all computational advantage at the required rank;
- repeated oracle capacity without a legal evaluator should no longer justify a workstream;
- the shipped estimator was much closer to its design-class optimum than the sequence of failed design experiments suggested;
- the official run is a major confidence improvement;
- the broken root estimator finding is immediately actionable.

## Where I disagree or qualify it

### “The last untested space closed”

Too broad.

The final work closes the last untested evaluator inside the local heteroscedastic-mixture implementation program. It does not close:

- exact tied/shared-covariance reuse;
- a non-mixture compact copula state;
- M194’s phase-bearing boundary identity;
- M189’s Kerdock-index QTT possibility;
- M193’s output-weighted boundary-normal compression;
- a genuinely new direct identity for the required diagonal.

These have low priors, but they are different mathematical classes.

### “The analytic mixture route is fully exhausted”

Almost, but not quite.

The tied/shared-covariance recurrence is special because the correlation structure can be reused across components. The final analysis begins by noting that sharing is possible when covariances are tied, but the later cost synthesis does not separately benchmark that exact recurrence.

It deserves one test, not a reopened ladder.

### “Write the paper”

The paper case is strong, but that is not the complete competition decision.

The correct competition sequence is:

1. independently reproduce the final gates;
2. run the one tied/shared-covariance exception;
3. if it fails, promote M194;
4. run M189 and M193 only as cheap falsifiers;
5. perform one low-cost grouped tail audit;
6. keep the validated baseline ready to ship.

The project should not restart broad analytic-mixture optimization, geometric sampling or learned-feature programs.

## Revised priorities

1. **Independent reproduction:** regenerate the final mixture curves and official score from the referenced scripts and JSON.
2. **Final M192 exception:** exact tied/shared-covariance \(K\leq64\) recurrence exploiting reusable correlations.
3. **M194:** one algebraically specified weight-coupled boundary/Walsh identity.
4. **M189:** one existing-array QTT falsifier.
5. **M193:** one output-weighted boundary-normal and gate-current audit.
6. **M210:** one grouped no-fit tail mechanism audit.
7. **M190:** internal contraction engine and proof lane only.
8. **M195:** deferred; no more handcrafted weight features.
9. Quarantine the root estimator and keep protected evaluation sealed.

## Final state

There is no new deployable improvement.

The validated baseline is now on firmer ground:

\[
\text{adjusted score}\approx1.4642\times10^{-7}
\]

with zero failures on the exposed official Mini split.

The former primary analytic program is nearly closed. Its failure is not that the missing joint state does not exist; the local data indicate that it does. The failure is that every tested way of evaluating that state loses either accuracy or cost.

That leaves a narrower and more honest frontier:

> Either exploit the one untested shared-covariance structure, or find a genuinely new exact identity that accesses deep phase/boundary information without reconstructing the full covariance state.

If the tied/shared-covariance test fails, there is no remaining broad candidate-development program—only M194 and a few bounded class escapes with low prior, alongside publication and shipping work.
