# Release status

**Audited revision date:** 2026-08-02  
**Protected evaluation opened:** No  
**Canonical ledger:** v31 final local write-up

## Status legend

- **Included:** present in this repository.
- **Replayed:** exact arithmetic or deterministic check rerun from available artifact.
- **Reported:** described in a source, but the raw rerun bundle is absent.
- **Missing:** required for artifact-complete reproduction and not present.
- **External gate:** requires a second implementation or named human review.

## Theory release

| Item | Status | Notes |
|---|---|---|
| T16 all-degree auxiliary theorem | Replayed / included | Exact reduced-cost generator passes; corrected primal, second interval-stack result, endpoint certificate, and supersession note included |
| Canonical nonnegative one-sided theorem record | Included / locally checked | v5.2 record included; T22 clean-regeneration report included; full external interval reconstruction still open |
| Audited signed frozen-witness inertia floor | Replayed | Exact-rational downstream replay gives fraction 0.9370601683665084 and factor cap 1.0671673322143325; inherited interval endpoints |
| Frozen-witness sign-count hierarchy | Replayed | q=1, 1072, 4160, and 8192 reproduced from the released allocation |
| Block-trace sharpness/equality | Included | Exact theorem |
| Shared-Gram sharpness and nonattainment | Included | Exact theorem; uniform signed epsilon open |
| Independent full T22/kernel coefficient interval reconstruction | Missing / external gate | T16 primal numerics have two stacks, but the complete inherited coefficient archive remains externally unaudited |
| Named human mathematical review | External gate | Focus on Hermite remainder, derivative proof, certificate scope |

## Empirical release

| Item | Status | Notes |
|---|---|---|
| Canonical v31 workbook | Included | Historical and current rows |
| Current-state memo | Included | Final local synthesis |
| Complete-block variance identity | Included in paper/ledger | Exact identity |
| Smoothed-anchor result | Reported | Source narrative retained; full raw bundle not assembled here |
| Mixture K ladder | Missing | Scripts, arrays, metric metadata absent |
| Pooled-within Taylor | Missing | Reported values only |
| Direct/Hermite rank sweep | Missing | Cost arithmetic reproduced; errors reported |
| Exact final 129-basis package | Missing | Do not substitute prior package |
| Official Mini-100 JSON | Missing | Aggregate values reported only |
| Root package failure bundle | Missing | Reported 2/2 smoke failure and API explanation |

## Audited corrections

- The original release headline used the unrecovered reoptimized T70 constant `0.9370605225569535`. The audited headline now uses the fully replayable frozen-witness value `0.9370601683665084`.
- The factor `1.067167...` is not a `6.7167%` reduction in risk. The equivalent maximum reduction relative to Kerdock risk is `6.293983...%`.
- All theorem prose now says fixed node budget rather than equal implementation cost.
- The Can et al. DOI and Kerdock spherical-code related work were corrected.
- Paper A no longer describes the public signed witness as the unrecovered 134-component reoptimization. The audited theorem keeps the released 146-profile allocation frozen and applies an exact positive-index strengthening downstream.
- Strict nonattainment is now explicitly attributed to the older abstract rank/block-trace floor, not silently transferred to the inertia-strengthened headline floor.
- Sign-count consequences are stated as counts of consolidated negative-weight support entries, not as claims about the magnitude of negative mass.

## Practical conclusion

No new deployable estimator is included. The public release is a theorem and research-provenance release, not a claim of a new benchmark submission.
