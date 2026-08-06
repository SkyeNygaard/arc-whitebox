# WHestBench best paths beyond v21 — exhaustive proof continuation

**Date:** 2026-07-30  
**Starting frontier:** canonical v21 proof frontier and the degree-280 signed certificate  
**Disposition:** `NEW_STATIC_BOUNDARY THEOREMS; NEW EXACT REPLAY FORMULA; FINITE-WIDTH SHORTCUT DISPROVED; A51 PATH NARROWED`

## Executive summary

This continuation deliberately starts after the v21 results rather than repeating them. The canonical starting facts are:

- every static, network-independent, mass-one rule with at most 66,048 arbitrary spherical nodes and arbitrary real weights has limiting-kernel risk at least
  \[
  0.9370459569114724\,R_K;
  \]
- rank plus every individual harmonic block trace is sharp over an abstract matrix relaxation;
- equality for one realized profile would force equal positive weights and pairwise zero feature-kernel inner products;
- a 90% finite-width subcertificate needs harmonic coefficient control only through degree 128;
- the retained A51 lower-recentering source depends exactly on the contracted center interface \((Vd,Ed)\), and the checkpoint defect obeys an exact gate-crossing recurrence.

The new conclusions are:

1. **The abstract rank/block-trace relaxation is simultaneously sharp for all harmonic profiles, not merely separately sharp profile by profile.** A single shared block matrix attains every profile's rank floor after diagonal rescaling. Therefore coupling profiles through one shared abstract block system cannot improve the theorem. Any stronger static result must use identities that distinguish spherical evaluation vectors from arbitrary block vectors.

2. **The released 93.7046% lower bound is strictly unattainable by every actual atomic spherical rule.** Two positive certificate components use the same adjacent degrees \(3,4\) but distinct mixing weights. Equality in both would require an off-diagonal inner product to be a common zero of normalized Gegenbauer polynomials \(G_3\) and \(G_4\). Their exact polynomial gcd is one and an explicit Bézout identity equals one, so no such inner product exists.

3. **Approaching the certificate floor requires pathological signed weights.** For every finite total-variation bound \(V\), compactness and strict nonattainment imply a positive, though currently non-explicit, gap \(\epsilon(V)>0\). Hence any sequence of real rules approaching the abstract floor must have total variation tending to infinity. A quantitative conditioned version shows that a near-floor sequence must also lose evaluation-matrix conditioning or drive some weights toward zero.

4. **There is an exact Gaussian gate-crossing formula that is stronger than the generic density-based cubic bound.** Conditional on a forward-measurable source and before drawing the current Gaussian weight row, the first and second moments of the exact ReLU crossing remainder depend only on the angle between the baseline and perturbed activation vectors. This removes the anti-concentration assumption for local ensemble crossing energy in an independent Gaussian suffix.

5. **A universal finite-width coefficientwise monotonicity shortcut is false.** A width-one Gaussian ReLU network keeps the one-layer kernel \(\kappa(t)\) at every later layer, whereas the infinite-width depth-two kernel is \(\kappa(\kappa(t))\). The width-one cubic Maclaurin coefficient is zero while the infinite-width coefficient is strictly positive. Thus actual width-256 transfer requires quantitative architecture-specific coefficient bounds; it cannot follow from a general principle that finite-width coefficients dominate limiting coefficients.

6. **The natural full-128 A51 interface may be equivalent to the full 256-dimensional center defect.** If the 128 selected coordinates are distinct and \(V_{I^c}\) is invertible, then \((Vd,E_Id)\) reconstructs all of \(d\) exactly. The exposed supports do use 128 distinct coordinates, but the frozen direction arrays needed to audit \(\det V_{I^c}\) were not retained in a directly materializable artifact. The correct continuation is therefore not “fit 128 outputs”; it is to derive the few final-output-contracted checkpoint quantities directly.

No deployable estimator is claimed. The work strengthens the theorem boundary and removes several attractive but invalid shortcuts.

---

# 1. Simultaneous sharpness of the abstract moment relaxation

## 1.1 Setup

Let the active harmonic blocks be real Hilbert spaces \(H_\ell\) of dimensions \(d_\ell\), with \(d_\ell\ge N\). A nonnegative harmonic profile is a finitely supported vector

\[
a=(a_\ell)_{\ell\in S},\qquad a_\ell\ge0.
\]

Write

\[
A_a=\bigoplus_{\ell\in S} a_\ell I_{d_\ell},
\qquad
T_a=\sum_\ell a_\ell d_\ell,
\qquad
S_{2,a}=\sum_\ell a_\ell^2d_\ell.
\]

For a mass-one \(N\)-atomic rule, the weighted feature moment matrix has rank at most \(N\) and block trace \(a_\ell d_\ell\). The existing lower bound is

\[
\|A_a-M_a\|_F^2\ge \frac{T_a^2}{N}-S_{2,a}.
\]

## 1.2 New theorem: one universal abstract optimizer works for every profile

For every block choose an isometry

\[
U_\ell:\mathbb R^N\to H_\ell,
\qquad U_\ell^TU_\ell=I_N.
\]

Define one profile-independent block matrix

\[
M^0_{\ell m}
=
\frac{\sqrt{d_\ell d_m}}{N}U_\ell U_m^T.
\]

For a profile \(a\), let

\[
D_a=\bigoplus_\ell \sqrt{a_\ell}I_{d_\ell},
\qquad
M_a=D_aM^0D_a.
\]

Equivalently, vertically concatenate the matrices

\[
V_{a,\ell}=\sqrt{\frac{a_\ell d_\ell}{N}}U_\ell.
\]

Then

\[
M_a=V_aV_a^T,
\]

and

\[
V_a^TV_a
=
\sum_\ell\frac{a_\ell d_\ell}{N}I_N
=
\frac{T_a}{N}I_N.
\]

Therefore:

- \(M_a\) has rank at most \(N\);
- all its nonzero eigenvalues equal \(T_a/N\);
- its \(\ell\)-block trace is exactly \(a_\ell d_\ell\);
- it attains
  \[
  \|A_a-M_a\|_F^2
  =\frac{T_a^2}{N}-S_{2,a}.
  \]

### Consequence

The same underlying abstract block system \(M^0\) simultaneously attains the rank floor for **every** nonnegative profile. Thus the following information is jointly insufficient:

- one shared block matrix;
- rank at most \(N\);
- every individual harmonic block trace;
- every diagonal rescaling corresponding to a comparison profile.

A joint SDP that contains only these constraints cannot improve the signed theorem. It must add a property of genuine spherical evaluation features: polynomial multiplication, commutation, the sphere ideal, a shared atomic moment functional, or a code constraint.

The verification package includes finite-dimensional random-frame checks for several unrelated profiles.

---

# 2. Strict nonattainment of the released signed certificate

## 2.1 Two incompatible positive components

The released certificate contains two positive components with \(s=3\):

\[
r_1=0.005623413251903491,
\qquad
r_2=0.0068129206905796083.
\]

For each \(r\), the feature kernel is proportional to

\[
L_r(t)=d_3G_3(t)+r d_4G_4(t).
\]

In dimension 256 the normalized Gegenbauer polynomials are exactly

\[
G_3(t)=\frac{t(86t^2-1)}{85},
\]

\[
G_4(t)=\frac{22360t^4-516t^2+1}{21845}.
\]

## 2.2 Theorem: no actual atomic rule attains the total floor

Suppose an actual rule attained the released total lower bound. The certificate decomposition is a sum of:

- a positive-semidefinite residual discrepancy;
- positive multiples of component discrepancies;
- for each component, a nonnegative excess above its rank floor.

Total equality therefore forces equality in every positive component. Applying the atomic equality characterization to both \(s=3\) components forces equal positive weights and, for every distinct node pair with inner product \(t\),

\[
d_3G_3(t)+r_1d_4G_4(t)=0,
\]

\[
d_3G_3(t)+r_2d_4G_4(t)=0.
\]

Subtracting the equations gives \(G_4(t)=0\), and then \(G_3(t)=0\).

But the exact numerator gcd is

\[
\gcd\big(t(86t^2-1),\;22360t^4-516t^2+1\big)=1.
\]

An explicit Bézout identity is

\[
172t(16640t^2-319)G_3(t)
-257(11008t^2-85)G_4(t)=1.
\]

Hence \(G_3\) and \(G_4\) have no common real or complex root. Contradiction.

> **Strict nonattainment theorem.** No mass-one atomic spherical rule with at least two active nodes attains the released degree-280 signed certificate floor.

This is stronger than the previous statement that equality would require a zero code for one profile. The released witness itself contains two incompatible zero-code requirements.

## 2.3 What strictness does not prove

Strict pointwise inequality does **not** automatically yield one global numerical improvement over the certificate constant for unrestricted signed rules. The parameter space is noncompact because weights may become arbitrarily large with cancellations and nodes may coalesce. A sequence could, in principle, approach the abstract floor through increasingly ill-conditioned representations even though no finite rule attains it.

## 2.4 Bounded-total-variation gap and blow-up corollary

Fix a finite total-variation bound

\[
\sum_i|w_i|\le V.
\]

After padding with zero-weight atoms, the parameter set

\[
(S^{255})^N\times
\{w:\sum_iw_i=1,\ \|w\|_1\le V\}
\]

is compact. Kernel risk is continuous. Because equality is impossible at every point, the minimum on this compact set is strictly above the released floor.

> **Bounded-TV corollary.** For every finite \(V\), there exists \(\epsilon(V)>0\) such that every rule with total variation at most \(V\) obeys
> \[
> R(Q)\ge L_{\rm cert}+\epsilon(V).
> \]

Consequently, any sequence of actual atomic rules whose risk tends to the abstract certificate floor must satisfy

\[
\|w\|_1\to\infty.
\]

Thus the final 6.3% theorem gap cannot be approached by a stable family of bounded signed weights.

## 2.5 Quantitative conditioned stability theorem

Normalize the two feature kernels to have diagonal one:

\[
K_j(t)=\frac{d_3G_3(t)+r_jd_4G_4(t)}{d_3+r_jd_4}.
\]

Their lack of a common zero implies

\[
m^2:=\min_{t\in[-1,1]}\big(K_1(t)^2+K_2(t)^2\big)>0.
\]

The verifier produces an exact rational Bézout-based lower bound on \(m^2\). It is conservative, but positive without numerical root finding.

For an \(N\)-atom rule, let \(E_j\) be its normalized evaluation matrix for profile \(j\), let \(M_j=E_j^TWE_j\), and define the rank excess

\[
\Delta_j=\|M_j\|_F^2-\frac1N.
\]

Assume

\[
\sigma_{\min}(E_j)\ge s>0,
\qquad
|w_i|\ge\mu>0.
\]

The rank-stability identity gives

\[
\left\|M_j^2-\frac1NM_j\right\|_F^2
\le
\left(\frac1N+\Delta_j\right)\Delta_j.
\]

Writing

\[
M_j^2-\frac1NM_j
=E_j^T\left(WG_jW-\frac1NW\right)E_j
\]

and using \(\sigma_{\min}(E_j)\ge s\) yields

\[
\left\|WG_jW-\frac1NW\right\|_F^2
\le
\frac{(1/N+\Delta_j)\Delta_j}{s^4}.
\]

The off-diagonal entries are \(w_iw_kK_j(t_{ik})\). Summing the two profiles gives

\[
\boxed{
\sum_{j=1}^2
\frac{(1/N+\Delta_j)\Delta_j}{s^4}
\ge
\mu^4N(N-1)m^2.
}
\]

Therefore both profile excesses cannot tend to zero while weights remain bounded away from zero and both evaluation matrices remain uniformly well conditioned.

### Interpretation

A near-attaining sequence must exploit at least one of:

- diverging total variation;
- vanishing active weights;
- coalescing or otherwise ill-conditioned evaluation vectors;
- failure of full active-node rank.

The current explicit numerical constant is too conservative to affect the competition bound. The theorem's value is that it identifies the only possible degeneracies a stronger realizability proof must control.

---

# 3. Exact Gaussian gate-crossing formula

## 3.1 Setup

Let

\[
w\sim N(0,I/d),
\]

and condition on deterministic vectors \(a,b\). Define

\[
z=w^Ta,
\qquad
z'=w^T(a+b),
\qquad
\delta=z'-z.
\]

The exact ReLU linearization remainder is

\[
R
=\operatorname{ReLU}(z')-\operatorname{ReLU}(z)-\mathbf1_{z>0}(z'-z)
=|z'|\mathbf1_{zz'<0}.
\]

Let

\[
\rho
=
\frac{a^T(a+b)}{\|a\|\,\|a+b\|}.
\]

## 3.2 Exact first and second moments

The pair \((z,z')\) is jointly Gaussian. Direct integration of the two opposite-sign quadrants gives

\[
\boxed{
\mathbb E_w R
=
\frac{\|a+b\|}{\sqrt d}
\frac{1-\rho}{\sqrt{2\pi}}.
}
\]

and

\[
\boxed{
\mathbb E_w R^2
=
\frac{\|a+b\|^2}{d}
\frac{\arccos\rho-\rho\sqrt{1-\rho^2}}{\pi}.
}
\]

The verification script independently compares these formulas against direct one-dimensional Gaussian quadrature and Monte Carlo over six correlations. Quadrature agrees to approximately machine precision.

## 3.3 Small-angle behavior

Writing \(\rho=\cos\theta\),

\[
\frac{\arccos\rho-\rho\sqrt{1-\rho^2}}{\pi}
=
\frac{\theta-\sin\theta\cos\theta}{\pi}
=
\frac{2}{3\pi}\theta^3+O(\theta^5).
\]

This gives an exact geometric explanation for cubic gate-crossing behavior. The controlling quantity is the angular change of the activation vector, not merely \(\|b\|\).

Important examples:

- if \(a+b=ca\) with \(c>0\), then \(\rho=1\) and the crossing remainder is exactly zero, regardless of the radial size change;
- an orthogonal perturbation can have substantial crossing error even when its Euclidean norm is modest;
- an approximately antiparallel perturbation is maximally dangerous.

## 3.4 Why this improves the nonlinear replay program

Suppose a checkpoint correction is measurable with respect to weights up to that checkpoint. Every subsequent Gaussian weight row is independent of the current incoming baseline and perturbed activation vectors conditional on the past. Therefore the displayed formula applies **exactly at every suffix layer conditional on previous suffix rows**.

For a layer with independent rows, the conditional expected local squared remainder is the sum of the scalar formulas. This eliminates the need to prove a generic conditional density bound for the local ensemble crossing term in a forward-measurable Gaussian suffix.

It does not yet close deep replay because:

- downstream-propagated local remainders interact;
- operator amplification and cross terms remain;
- a correction that directly inspects the current or future row violates the conditional independence premise;
- the competition evaluates fixed realized networks, whereas this formula is ensemble-conditional.

The exact formula should replace the looser density lemma whenever the source is prefix-measurable.

---

# 4. Finite-width transfer: a universal shortcut is false

## 4.1 Tempting but invalid conjecture

A very attractive route would be to prove

\[
k_\ell^{(m)}\ge k_\ell^{(\infty)}
\]

for all harmonic or power-series coefficients. The degree-128 limiting witness would then transfer with little work.

This monotonicity is false in the standard Gaussian ReLU family.

## 4.2 Width-one counterexample

Let

\[
\kappa(t)
=
\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi}
\]

be the normalized ReLU dual activation.

In a scalar width-one hidden layer, activations are nonnegative. For the next scalar Gaussian weight \(g\),

\[
\operatorname{ReLU}(g\,a(x))
=
\operatorname{ReLU}(g)a(x).
\]

Thus every later layer is only a random positive scalar multiple of the first-layer feature. After normalization, the finite-width kernel remains

\[
K_{m=1,L}(t)=\kappa(t)
\]

at every depth.

The Maclaurin expansion is

\[
\kappa(t)
=
\frac1\pi+rac12t+rac{1}{2\pi}t^2+rac{1}{24\pi}t^4+\cdots,
\]

so its cubic coefficient is zero.

At infinite width and depth two,

\[
K_{\infty,2}(t)=\kappa(\kappa(t)).
\]

Its cubic coefficient is

\[
\boxed{
\frac{-12+13\pi^2}{48\pi(\pi^2-1)^{3/2}}
\approx0.029197816>0.
}
\]

Therefore coefficientwise finite-width domination of the limiting kernel fails.

## 4.3 Consequence for width 256

This does not say that the width-256 coefficients are unfavorable. It says that no theorem based only on “Gaussian ReLU finite width” can supply the needed one-sided coefficient margins. The degree-128 program needs at least one architecture-specific quantitative ingredient:

- a recursive interval bound on low-order chaos energies;
- a finite-width diagrammatic expansion with a controlled remainder;
- a Hausdorff-moment dual using rigorously bounded kernel values;
- or a direct one-sided perturbation theorem specialized to width 256 and depth 32.

The existing qualitative nonnegative-noise-stability representation proves coefficient positivity and fixed-MUB signs, but the width-one example confirms that the missing numerical margins are mathematically essential.

---

# 5. A51: when the natural interface is not a compression

## 5.1 Exact v21 interface

For the retained lower-recentering source, let

\[
d=\mu-m,
\qquad
x=Vd,
\qquad
y=Ed,
\]

where \(E\) selects the 128 probe coordinates. The exact lower anchor is a known nonlinear map

\[
\ell(d)=\Phi(Vd,Ed).
\]

The v21 coefficient-uniform minimality theorem shows that no universal **linear** sufficient statistic can have rank below \([V;E]\).

## 5.2 New conditional equivalence theorem

Let the selected coordinate set \(I\) have size 128 and be distinct. Partition

\[
d=(d_I,d_{I^c}),
\qquad
V=(V_I,V_{I^c}).
\]

If

\[
\det V_{I^c}\ne0,
\]

then

\[
d_I=Ed
\]

and

\[
\boxed{
d_{I^c}=V_{I^c}^{-1}(Vd-V_Id_I).
}
\]

Thus \((Vd,Ed)\) is a bijective linear encoding of the entire 256-dimensional defect.

> Under the invertibility condition, exact recovery of the natural full-128 A51 interface is exactly as informative as exact recovery of the full center defect.

The high-reference sample-row supports use 128 distinct coordinates in every exposed record. However, the corresponding frozen \(V\) arrays were not retained in a directly materializable file, so the actual determinant/rank audit remains open. Generic random row directions would make \(V_{I^c}\) invertible with probability one, but this is not evidence about the frozen construction.

## 5.3 Empirical local-versus-universal compression

The archived lower-structure diagnostic shows:

- local rank-two median anchor relative error: approximately `0.002620`;
- local rank-four median anchor relative error: approximately `0.000843`;
- universal rank-128 median anchor relative error: approximately `0.406740`;
- the universal training spectrum needs ranks 29, 36 and 44 for 90%, 95% and 99% energy;
- a universal 24-dimensional center PCA captures only approximately `10.84%` of median center-defect energy.

This is a characteristic rotating-subspace pattern: each network's anchor matrix is locally compressible, but the useful singular directions are not shared across networks.

The correct response is not a larger universal PCA. It is a **covariant identity** whose directions rotate with the physical network.

## 5.4 Same-cloud no-op corner

Suppose a control anchor is replaced by its value under the same cubature functional \(Q\). Then for any, even data-dependent, coefficient \(\beta\),

\[
Q(f-\beta g)+\beta Q(g)=Q(f)
\]

pathwise. Cross-fitting \(\beta\) prevents regression overfit but does not create an absolute expectation anchor.

The viable escape classes are:

- an independent absolute anchor;
- an exact analytic expectation;
- a white-box identity for the missing contractions;
- added evaluations whose signed values are explicitly included in the information protocol.

## 5.5 Best A51 continuation

Do not estimate \((Vd,Ed)\) generically. Freeze a physical rank-4 or rank-5 final-output source and derive only its scalar normal-equation vector

\[
b=A^Te.
\]

Use the exact checkpoint recurrence to express each entry as a small sum of adjoint-contracted crossing defects. The exact Gaussian crossing formula then gives a theorem-ready local suffix interface whenever the source is prefix-measurable.

The source must first have a final-output oracle ratio below approximately `0.20-0.22`; otherwise no plausible coefficient method can meet the competition score after overhead. Only after this source-capacity gate should a constant-first residual policy be tested.

---

# 6. Exhaustive path assessment

## 6.1 Static realizability

### Proved now

- rank and all block traces are simultaneously sharp in one shared abstract block system;
- the released certificate floor is not attained by any real atomic rule;
- bounded-TV classes have a positive strict gap;
- near-floor sequences must become unbounded or ill conditioned.

### Still open

- an explicit useful \(\epsilon(V)\);
- a TV-free uniform realizability gap;
- a zero-code cardinality theorem for the active kernels;
- a joint spherical moment/catalecticant inequality;
- whether unrestricted ill-conditioned signed rules can approach the abstract floor.

### Best next proof

Quantify the stability theorem by deriving lower bounds on evaluation-matrix singular values from node separation or by using a shared polynomial moment matrix that remains well conditioned under coalescence. A semialgebraic separating polynomial for the two incompatible \(s=3\) profiles would be particularly valuable.

## 6.2 Finite-width degree-128 transfer

### Proved now

- the general coefficientwise monotonicity shortcut is false;
- quantitative low-degree information is logically necessary.

### Still open

- rigorous width-256 chaos-energy intervals;
- a degree-128 Hausdorff-moment dual;
- a rigorous finite-width Kerdock denominator upper bound.

### Best next proof

Start at a lower cutoff, such as degree 62 or 84, to validate the complete moment-dual pipeline. Extend to 128 only after a nontrivial actual-width floor is certified. The verifier should report which finite-width moment constraints are dual-active, so the probabilistic task remains compressed.

## 6.3 Multi-checkpoint source identity

### Proved now

- A51's natural exact interface can generically be full-dimensional;
- local low rank does not transfer globally;
- the local Gaussian crossing moments are exact for prefix-measurable sources.

### Still open

- a frozen physical rank-4/5 source with winning-scale oracle capacity;
- exact or certifiable formulas for its contracted crossing defects;
- complete deep replay with downstream cross terms;
- legal per-network signed coefficient information beyond the global action.

### Best next construction

Use checkpoint repair arrays to choose four physical channels by a target-free, covariant rule. Compute their exact source-span ceiling first. For each channel derive one scalar adjoint integrand. Stop immediately if the union ratio is not below `0.20-0.22`.

## 6.4 Constant-first residual policy

This remains conditional on a new high-capacity source. The old five-source class should remain closed. The correct action is

\[
a(X)=a_0+\lambda\delta a(X),
\qquad0\le\lambda\le1,
\]

with zero residual as the default, Gram whitening, clipping and abstention fixed before grouped evaluation.

The exact comparison target is incremental conditional value beyond \(a_0\), not total oracle span value.

## 6.5 Exactly integrable residual surrogate

No new source was discovered in this continuation. The route remains logically open only if a surrogate supplies all four:

1. exact expectation;
2. live degree-six-plus spectrum;
3. cheap evaluation or replacement of baseline work;
4. a recertified residual kernel with a winning score floor.

Retuning Poisson radii, projected-ReLU dictionaries or low-degree controls is not a new path.

## 6.6 Computational lower bounds

A universal white-box impossibility remains unrealistic because the weights determine the target. A useful theorem must fix a representation or query class. The strongest plausible narrow model is a bounded number of prefix-measurable scalar contractions followed by a bounded-degree/rational action. This should be attempted only after the constructive contraction program identifies the actual sufficient statistics it wants.

---

# 7. Ranked next work

## Priority 1 — physical four-channel contraction source

1. Freeze four or five covariant checkpoint channels.
2. Verify a source-span final-output ratio below `0.20-0.22`.
3. Derive \(A^Te\) as scalar checkpoint crossing contractions.
4. Use the exact Gaussian-row formula at every independent suffix layer.
5. Bound remaining downstream cross terms.
6. Only then test a constant-first residual coefficient policy.

## Priority 2 — quantitative atomic realizability

1. Build a joint moment matrix for the two incompatible degree-3/4 profiles.
2. Add sphere-ideal and multiplication-operator constraints.
3. Seek a certified lower bound on either total variation, evaluation conditioning or off-diagonal common-zero defect.
4. Translate the result into an explicit improvement over `0.9370459569`.

## Priority 3 — actual-width degree-62/84 pilot theorem, then degree 128

1. Derive rigorous low-order finite-width moment intervals.
2. Solve and independently verify the moment dual.
3. Produce the first nontrivial arbitrary-signed width-256 floor.
4. Identify active moments and extend only those to degree 128.

## Priority 4 — exact deep replay composition

Use the local Gaussian formula to replace anti-concentration at each prefix-measurable suffix step. The remaining theorem should control the energy and cross terms of the exact Duhamel sum, preferably through conditional orthogonality or martingale differences rather than worst-case products of operator norms.

---

# 8. Stop and reopen map

## Stop

- any further rank/block-trace optimization without sphere identities;
- continuous-radius or multiblock profile engineering;
- a claim that strict nonattainment alone improves the numeric signed constant;
- generic estimation of all 128 A51 coordinates;
- universal PCA of center defects;
- same-cloud anchor reconstruction;
- coefficientwise finite-width monotonicity assumptions;
- generic zero-centered coefficient learners;
- more tests on the four old \(K\le32\) support menus;
- linearized replay without exact gate crossing.

## Reopen only with a materially new object

- a joint spherical moment inequality;
- a bounded-TV or conditioning theorem with a useful explicit constant;
- a rigorously certified width-256 moment dual;
- a physical rank-4/5 source with ratio below `0.20-0.22`;
- an exact scalar identity for its contracted crossing defects;
- a residual source with exact expectation and a recertified spectrum;
- a declared computational model containing plausible winning algorithms.

---

# 9. Verification and trust status

The package includes four independent scripts:

1. simultaneous abstract sharpness, exact Gegenbauer gcd/Bézout strictness, and the conditioned stability identity;
2. Gaussian crossing formulas checked against direct quadrature and Monte Carlo;
3. an exact symbolic finite-width monotonicity counterexample;
4. the A51 invertibility condition and archived local-versus-universal compression audit.

All scripts pass. The strictness and monotonicity results are ordinary exact algebra. The released `0.9370459569` constant remains computer-assisted and retains its external Arb/FLINT and human-review gates. The new bounded-TV gap is existential; no useful explicit global epsilon is claimed.
