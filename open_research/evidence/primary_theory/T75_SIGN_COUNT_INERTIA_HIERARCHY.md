# T75 — Sign-count inertia hierarchy for static signed cubature

**Date:** 2026-07-30  
**Status:** Exact spectral theorem plus exact-rational order-320 specializations on a frozen 146-profile grid. The kernel lower endpoints inherit the v21 directed interval stack.

## 1. Theorem

Let a static mass-one rule use at most \(N\) nonzero weights, of which at least \(q\ge1\) are negative. For every comparison profile with feature moment matrix

\[
M=E^TWE,
\]

the number of positive eigenvalues of \(M\) is at most the number of positive entries of \(W\), hence at most \(N-q\). If the profile trace is \(T>0\), then

\[
\boxed{
\|M\|_F^2\ge {T^2\over N-q}.
}
\]

Consequently, for profile diagonal target \(A\) with

\[
S_2=\|A\|_F^2,
\]

we have

\[
\boxed{
\|A-M\|_F^2\ge {T^2\over N-q}-S_2.
}
\]

The proof is the positive-index trace inequality from T73 with \(p=N-q\).

## 2. Certified hierarchy for WHestBench

The v21 146-profile coefficient grid was reoptimized separately for each sign-count class, then every floating witness was shrunk, rounded downward, and verified with exact rational arithmetic against the order-320 directed kernel-coefficient lower endpoints.

| minimum negative weights | maximum positive weights | risk / Kerdock | same-cost gain cap |
|---:|---:|---:|---:|
| 1 | 66,047 | **0.9370605226** | **1.0671669289×** (stronger dedicated T73 certificate) |
| 2 | 66,046 | 0.9370739626 | 1.0671516230× |
| 16 | 66,032 | 0.9372729741 | 1.0669250343× |
| 64 | 65,984 | 0.9379559403 | 1.0661481600× |
| 256 | 65,792 | 0.9406984935 | 1.0630398655× |
| 1,024 | 65,024 | 0.9518267726 | 1.0506113389× |
| 1,072 | 64,976 | 0.9525316552 | **1.0498338765×** |
| 2,048 | 64,000 | 0.9670812292 | 1.0340393029× |
| 4,096 | 61,952 | 0.9991036420 | 1.0008971622× |
| 4,160 | 61,888 | 1.0001384731 | **0.9998615461×** |
| 8,192 | 57,856 | 1.0699484210 | 0.9346244925× |

Thus:

- any rule with at least **1,072** negative nodes cannot improve Kerdock by 1.05× at equal cost;
- any rule with at least **4,160** negative nodes has certified risk strictly **above** Kerdock;
- the only sign pattern compatible with the global 1.06717× boundary is extremely sparse negativity, with the worst class attained at one negative node.

## 3. Interpretation

Negative weights are not a generic source of freedom. The theorem gives a monotone structural tax for every additional negative atom. To approach the universal signed boundary, a candidate must use almost all nodes with positive weights and only a tiny number of negative atoms.

This complements, rather than replaces, the negative-mass theorem:

- T75 controls the **number** of negative atoms, regardless of their magnitudes;
- T64 controls the **total negative mass**, regardless of its support size;
- T61–T64 show that approaching the older abstract boundary also requires unbounded total variation or degenerating evaluation geometry.

Together these results squeeze a hypothetical near-optimal signed construction into a narrow and unstable regime: very few negative atoms, nontrivial negative mass, and increasingly singular cancellations.

## 4. Scope

Covered:

- static, network-independent, mass-one linear rules;
- arbitrary spherical nodes;
- arbitrary real weights;
- at most 66,048 active nodes;
- a declared lower bound on the count of negative weights.

Not covered:

- adaptive or value-dependent weights/nodes;
- nonlinear estimators;
- arbitrary total mass without adding the constant-mode argument;
- finite-width network objectives.

## 5. Verification note

For the one-negative class, T73 is the stronger frozen witness. The hierarchy verifier uses a more conservative universal shrink and is reported only as a consistency check for \(q=1\). All rows from \(q\ge2\) are independently exact-rational certificates on their declared grids.
