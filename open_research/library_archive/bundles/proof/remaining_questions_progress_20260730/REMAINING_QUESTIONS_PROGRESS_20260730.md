# WHestBench remaining questions — continued theorem and downstream progress

**Date:** 2026-07-30

## Executive summary

This round produces five substantive advances and one correction.

1. **The rank/block-trace relaxation is now characterized exactly.** It is sharp over abstract symmetric rank-`N` matrices with the required harmonic block traces. Therefore no argument using only rank and those traces can improve the signed certificate.
2. **Actual equality is much more rigid.** A signed point-evaluation rule can attain the abstract floor only if it uses exactly `N` equal positive weights and its nodes are pairwise orthogonal under the comparison feature kernel. The residual theorem gap is therefore a spherical zero-code / moment-realizability problem.
3. **The continuous-radius search had a real overflow bug.** After repair, the continuous adjacent cone improves the exact released fraction only from `0.9370459569` to `0.9370496015` numerically. General contiguous multiblocks reach `0.9370553398`. Neither explains a material part of the remaining gap.
4. **Finite-width transfer has an exact proof-cost frontier.** A valid 90% infinite-width signed subcertificate requires coefficients only through degree 128; 93% requires degree 194; the full 93.7046% certificate uses degree 280.
5. **The target-free support branch fails at the support-capacity level.** Even target-labeled selection of the best of four tested support families per record captures only `40.53%` on rotation 19 and `48.54%` pooled at `K=32`.
6. **The five-source feature model is not a residual model.** It usually reverses the globally useful first coefficient. This explains why a flexible rule can lose to a constant and motivates a mandatory constant-first residual protocol.

The strongest certified static theorem remains:

\[
R(Q)\ge0.9370459569114724\,R_K
\]

for every static mass-one rule with at most 66,048 arbitrary spherical nodes and arbitrary real weights, under the dimension-256 depth-32 infinite-width normalized ReLU kernel.

---

## 1. Abstract sharpness of the block-trace theorem

For a weighted harmonic feature space

\[
A=\operatorname{diag}(a_jI_{d_j}),
\qquad
T=\sum_ja_jd_j,
\qquad
S_2=\sum_ja_j^2d_j,
\]

any mass-one `N`-node signed rule produces a rank-at-most-`N` moment matrix `M` with the individual block traces fixed. Hence

\[
\|A-M\|_F^2\ge {T^2\over N}-S_2.
\]

### New theorem: abstract sharpness

If `Na_j/T<=1` in every block, there exists a rank-`N` projection with diagonal consisting of `d_j` copies of `Na_j/T`. Scaling this projection by `T/N` attains equality.

Every component in the released signed certificate satisfies this condition with substantial margin. Therefore:

> Rank and all individual harmonic block traces completely exhaust the current abstract matrix relaxation.

A stronger theorem must exploit that `M` is an evaluation moment matrix, not an arbitrary symmetric matrix.

### New theorem: atomic equality characterization

If an actual signed atomic rule attains the rank floor, then the matrix equality condition forces

\[
M^2=(T/N)M.
\]

Writing `M=E^TWE` and `G=EE^T`, full row rank gives

\[
WGW=(T/N)W.
\]

All nonzero weights must therefore equal `1/N`, and

\[
G_{ij}=L_a(x_i,x_j)=0\quad(i\ne j).
\]

Thus equality requires an equal-positive-weight `N`-point zero code for the comparison feature kernel. Negative weights cannot attain the abstract boundary.

### Downstream mathematical targets

The next useful hierarchy should constrain one shared spherical moment functional, rather than independently relaxing each feature block. Candidate tools:

- joint catalecticant/Hankel constraints;
- commutative multiplication operators satisfying the sphere relation;
- bounds on the idempotence defect `||M^2-(T/N)M||`;
- zero-code or few-distance bounds for the active feature kernels;
- a joint SDP over several comparison profiles sharing the same moment sequence.

---

## 2. Corrected comparison-cone audit

### Overflow defect

The original continuous-radius stationary-point calculation formed products of enormous quadratic coefficients. At high degree, double precision overflow omitted interior minima and made the first saturation claim invalid.

The repaired routine rescales numerator and denominator quadratics independently before differentiating. This leaves the stationary points unchanged.

### Corrected numerical frontier

| Cone | Fraction of Kerdock upper |
|---|---:|
| Released exact adjacent grid | `0.9370459569114724` |
| Corrected continuous adjacent cone | `0.9370496015333069` |
| General contiguous multiblock cone | `0.9370553397756372` |

The extra gains are only:

- continuous radii: `0.00036446` percentage point;
- multiblock profiles: `0.00093829` percentage point.

The latter two remain numerical discovery results. They do not replace the exact-rational released theorem constant.

### Conclusion

The remaining roughly 6.3% static theorem gap is not caused by the radius grid or by restricting profiles to adjacent pairs. Realizability is now the dominant mathematical target.

---

## 3. Exact finite-width subcertificate frontier

The released witness can be truncated exactly. Retaining only components supported through degree `M` gives a valid lower certificate requiring finite-width coefficient control only through `M`.

| Certified infinite-width floor | Degree cutoff |
|---:|---:|
| 50% | 22 |
| 60% | 28 |
| 70% | 40 |
| 80% | 62 |
| 85% | 84 |
| 90% | 128 |
| 92% | 164 |
| 93% | 194 |
| 93.5% | 214 |
| 93.7% | 242 |
| Full 93.7046% | 280 |

For cutoff `M`, let `f_M` be the listed subcertificate fraction. If

\[
k_\ell^{(256)}\ge\alpha_Mh_{M,\ell}
\quad(1\le\ell\le M)
\]

and width-256 Kerdock risk is at most `beta` times the infinite-width Kerdock upper endpoint, then

\[
{R_{256}(Q)\over R_{256}(Q_K)}
\ge {\alpha_M f_M\over\beta}.
\]

### New compressed proof program

A finite-width network kernel is an exact Gaussian noise-stability series

\[
K_m(t)=\sum_{n\ge0}a_n^{(m)}t^n,
\qquad a_n^{(m)}\ge0.
\]

Instead of estimating 128–280 harmonic coefficients separately:

1. interval-bound selected finite-width values `K_m(t_j)` or OU/noise-stability functionals;
2. solve a nonnegative Hausdorff-moment LP over `a_n^(m)`;
3. lower-bound only the comparison coefficients required by one subcertificate;
4. combine with an upper bound on finite-width Kerdock MSE.

The 90% cutoff at degree 128 is the best first target.

---

## 4. Target-free layer-31 support capacity

For a fixed support, replay using the true defect amplitudes is an upper ceiling for every coefficient learner restricted to that support.

At `K=32`:

| Rotation | radial-H3 capture | PCA-sensitivity capture | target-labeled best family per record |
|---:|---:|---:|---:|
| 3 | 51.08% | 53.21% | 53.48% |
| 19 | 36.50% | 38.90% | 40.53% |
| Pooled | 45.52% | 47.75% | 48.54% |

The support indices are highly stable across rotations, but their exact-amplitude capacity is not. This proves that support reproducibility and downstream-signed defect containment are separate properties.

### Closed class

The result closes all descendants that:

- use one of the four tested support families;
- keep `K<=32`;
- only change amplitude estimation or choose among those four supports.

Even an oracle family selector fails the 50% gate on rotation 19 and pooled.

### Open escape

- a larger or differently constructed support;
- a multi-checkpoint basis matching the measured rank-four repair space;
- a support built from an independent signed absolute-phase anchor;
- nonlinear source features not confined to coordinate selection.

---

## 5. Constant-first residual coefficients

The frozen five-source global coefficient is

\[
(-0.5591,0.3145,0.1725,0.2405,0.2257).
\]

The feature policy predicts the first coefficient with the global negative sign in only:

- `33%` of development cases;
- `25%` of validation cases;
- `33%` of confirmation cases.

Its mean first coefficient is positive on every split. The other four coefficient signs match the global rule in every case.

On confirmation, the feature rule is worse than the global rule in 8/12 cases and raises the mean case ratio by approximately `0.143`; its pooled ratio and worst case are also worse.

### Diagnosis

The model is not estimating a small network-specific residual. It is repeatedly relearning and reversing the strongest average action.

### Required next protocol

Freeze `a_0` and predict only

\[
a(X)=a_0+\lambda\delta a(X),
\qquad 0\le\lambda\le1.
\]

Whiten the source dictionary by the per-instance Gram matrix, freeze `lambda` and clipping on grouped development, and report only incremental feature value over the global rule. Zero residual must be the default action.

---

## 6. Revised priority order

### A. Joint moment realizability theorem

Prove a nonzero distance between the abstract scaled projection and spherical evaluation moment matrices. This is the only identified route to a materially stronger signed static theorem.

### B. Degree-128 finite-width transfer

Target a 90% signed floor first. Use finite-width noise-stability moment constraints rather than a full harmonic reconstruction.

### C. Four-channel source construction

The Oracle continuation shows a stable effective repair rank near four. Build legal sources for the four major checkpoint channels separately and measure each source ceiling before coefficient learning.

### D. Constant-first orientation-aware residual experiment

Only after a global baseline and source basis are frozen should a small set of independently referenced signed features be tested.

### E. One Poisson conditioning experiment

Reparameterize the existing Poisson span into divided-difference spectral coordinates and whiten it. This tests conditioning, not a new oracle span.

---

## 7. Remaining honest frontier

1. Can shared spherical moment constraints improve the 93.7046% signed floor materially?
2. Can a degree-128 subcertificate be transferred to width 256?
3. Can four major Oracle repair channels be approximated with legal high-ceiling sources?
4. Is there any independently referenced signed feature with positive value beyond the frozen global coefficient?
5. Can a nonlinear exactly integrable surrogate reshape the residual spectrum enough to justify recertifying a new rule?

The first two are theorem questions. The last three are constructive estimator questions. They should not be blended into one broad impossibility claim.
