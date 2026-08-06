# Dual/adjoint contracted third-cumulant propagation

## Objective

The connected-cubic control does not need a full \(256^3\) cumulant or even
the full \(256\times256\) slice. For each control direction it needs

\[
q(M,v;K)
=\sum_{abc}M_{ab}v_cK_{abc}.
\]

At the final post-ReLU layer,

\[
M=\operatorname{diag}(u),\qquad
q=\sum_{ij}u_i v_j K_{iij}.
\]

This audit asks whether reverse/adjoint propagation of that scalar can replace
the 272.6B-FLOP factorized \(K_3\) rollout.

## Exact affine adjoint

For column-vector convention, let

\[
z_\ell=W_\ell h_{\ell-1}.
\]

The affine third-cumulant transport is exact:

\[
K^-_\ell=W_\ell^{\otimes3}K^+_{\ell-1}.
\]

If the ReLU cumulant approximation is decomposed as

\[
K^+_\ell
=T_\ell^{\otimes3}K^+_{\ell-1}+S_\ell,\qquad
T_\ell=D_\ell W_\ell,
\]

where

\[
D_\ell=\operatorname{diag}\left(
E[\operatorname{ReLU}'(Z_{\ell,i})]
\right),
\]

then the exact adjoint recurrence for a symmetric \(M\) is

\[
\begin{aligned}
q_\ell(M_\ell,v_\ell)
&=\langle M_\ell\otimes v_\ell,S_\ell\rangle
 +q_{\ell-1}(M_{\ell-1},v_{\ell-1}),\\
M_{\ell-1}&=T_\ell^\top M_\ell T_\ell,\\
v_{\ell-1}&=T_\ell^\top v_\ell.
\end{aligned}
\]

The input is Gaussian, so its third cumulant is zero and only the accumulated
source contractions remain.

## Contracting a CP source

The factorized implementation represents a symmetric source as

\[
S=\operatorname{Sym}\sum_r A_r\otimes B_r\otimes C_r.
\]

Its contraction with a symmetric \(M\) is

\[
\langle M\otimes v,S\rangle
=\frac13\sum_r\left[
(A_r^\top M B_r)(v^\top C_r)
+(A_r^\top M C_r)(v^\top B_r)
+(B_r^\top M C_r)(v^\top A_r)
\right].
\]

The implemented reverse recurrence reconstructs the direct full-CP
contraction to relative error below \(10^{-15}\) on every tested network.

## Where approximation enters

The affine adjoint is exact. The difficulty is constructing \(S_\ell\).

For weak non-Gaussianity, the leading HOUT/Edgeworth form is

\[
K^+_\ell
\simeq D_\ell^{\otimes3}K^-_\ell
+S^{G}(m^-_\ell,C^-_\ell),
\]

where \(S^G\) is the connected third cumulant created by applying ReLU to a
Gaussian with the supplied mean and covariance.

The vendor SIMPLE/factorized \(k=3\) update contains more:

- terms involving incoming repeated slices \(K^-_{iij}\);
- a pure-radial fourth-order harmonic state;
- repeated-index corrections from the power-cumulant-to-cumulant conversion;
- corresponding corrections to later means and covariances.

Therefore the exact residual source after expected-gate transport is not a
function of mean and covariance alone. Producing that exact source currently
requires the full growing factor state. A contracted deployment must either:

1. carry additional adjoint families for the repeated \(K_3\), radial \(K_4\),
   and lower-moment dependencies; or
2. drop them and use a Gaussian/HOUT local source.

The implemented cheap experiment uses option 2 and obtains its lower moments
from a separate \(k=2\) rollout.

For a Gaussian-only local source, the factorized code produces 768 raw CP
columns per layer, of which 512 are structurally active. The leading block has
the form

\[
A=D_{g_1}C,\qquad B=3I,\qquad
C_{\rm fac}=D_{g_1}C D_{g_2},
\]

plus an active repeated-index correction block.

## Validation protocol

- Networks: IDs 160--167 only.
- Target layer: post-ReLU layer 29.
- Two controls: top two SVD directions of the full factorized C21 artifact.
- Comparisons:
  - exact residual sources extracted from the full rollout;
  - Gaussian sources with full-rollout mean/covariance and gates (“frozen”);
  - Gaussian sources with a cheap \(k=2\) lower-state rollout;
  - rank-32 truncation of the final diagonal \(M\) probe.

No IDs 168+ were opened.

## Contraction accuracy

Against the existing full factorized C21 contraction:

| Method | Cosine | Optimal scale | Scaled relative error |
|---|---:|---:|---:|
| Exact dual, full residual sources | 0.99975 | 0.9680 | 2.26% |
| Frozen lower state + Gaussian sources | 0.98878 | 0.7810 | 14.94% |
| Cheap k=2 state + Gaussian sources | 0.98112 | 1.2238 | 19.34% |
| Cheap source + rank-32 probe | 0.97721 | 1.4727 | 21.23% |

The exact-dual row differs from the exported slice because
`FactoredTensor.get_dslice((2,1))` deliberately zeroes its diagonal; \(K_{iii}\)
is stored in the `(3,)` slice. The adjoint computes the mathematically full
CP contraction. Direct CP and exact dual agree to \(4.0\times10^{-16}\).
The slice convention costs 4.0% raw error or 2.26% after scalar calibration.

Against the 100M-sample oracle connected C21 contraction:

| Method | Cosine | Pearson | Optimal scale | Scaled relative error |
|---|---:|---:|---:|---:|
| Full factorized artifact | 0.99741 | 0.99531 | 1.4160 | 7.20% |
| Frozen Gaussian-source dual | 0.99097 | 0.98334 | 1.1112 | 13.41% |
| Cheap Gaussian-source dual | 0.98373 | 0.96996 | 1.7420 | 17.97% |
| Cheap rank-32 dual | 0.97944 | 0.96242 | 2.0954 | 20.17% |

The cheap dual's per-network optimal scale is unstable:

\[
\text{mean}=1.954,\quad
\text{sd}=0.414,\quad
\text{range}=1.557\text{--}2.616.
\]

Rank-32 truncation worsens it to mean 2.476, standard deviation 0.656.
This matters because the full factor artifact already proved too inaccurate
as a connected-cubic anchor despite its substantially better 7.2% scaled
directional error.

## FLOP accounting for two controls

Let \(d=256\), \(L=30\), and \(r=2\).

### Dense generic adjoint

The \(M,v\) transition costs per layer and control

\[
4d^3+2d^2.
\]

Across 30 layers and two controls this is 4.03B FLOPs.

The cheap Gaussian sources have total active CP rank

\[
30(2d)=15{,}360.
\]

Generic source contraction adds about 8.12B, giving:

\[
\boxed{12.16\text{B}}
\]

for the reverse contractions themselves.

The full cheap operational estimate also includes:

- \(k=2\) lower-state rollout: 1.546B;
- local Gaussian-source construction: approximately 6.324B, estimated as
  30 times the vendor first-layer \(k=3\) minus \(k=2\) polynomial cost.

Total:

\[
\boxed{\approx20.0\text{B}}
\]

This is far below 85B and an order of magnitude below the full 272.6B
factorized rollout.

### Sub-10B low-rank probe

Starting \(M=\operatorname{diag}(u)\), keep only the 32 largest diagonal
entries and represent

\[
M=Q\Lambda Q^\top.
\]

The adjoint preserves this rank:

\[
Q\leftarrow T^\top Q.
\]

For rank 32, transition and source contractions cost about 1.83B. Including
the same lower-state and source-construction costs gives:

\[
\boxed{\approx9.7\text{B}}.
\]

Across the tested directions, the rank-32 terminal probe retains 79.3% of
the diagonal probe's squared Frobenius energy on average. Its final oracle
error, however, rises to 20.17%, and calibration becomes less stable.

Rank 64 retains 93.2% of probe energy but raises the total estimate to roughly
11.5B.

## Decision

The dual formulation is a genuine computational improvement:

- exact contracted evaluation is algebraically valid;
- a generic approximate implementation is around 20B;
- a rank-32 approximation can cross below 10B.

It does **not** currently solve the anchor:

- mean/covariance-only HOUT sources omit state-dependent \(K_3/K_4\) terms;
- the cheap directional error is 18--20%;
- network-specific amplitude varies by more than 20%;
- this is materially worse than the already-unsuccessful full factor anchor.

Do not integrate this approximation into the submission as-is. It becomes
interesting again only if an adjoint closure including the repeated \(K_3\)
and radial \(K_4\) dependencies can preserve a small family of matrix/vector
probes without rank explosion, or if a deterministic contracted-source model
reduces directional error below the full factor artifact's roughly 7% level.
