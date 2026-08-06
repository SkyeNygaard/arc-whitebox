# A 66,048-point Kerdock rule is basically optimal - for the static problem it actually solves

I spent a large part of an estimator competition trying to beat a structured Kerdock / mutually-unbiased-basis quadrature rule.

The obvious interpretation of repeated failure would be that I had not found the right static nodes or weights. The mathematical work now supports a much stronger conclusion:

> For the limiting depth-32 ReLU kernel in dimension 256, complete Kerdock is within about **0.0233%** of optimal among all static, network-independent, nonnegative mass-one rules with the same 66,048-node budget.

Even if arbitrary signed weights are allowed, the fully replayable frozen witness proves that the optimum retains at least **93.7060%** of Kerdock risk. That is at most a **6.2940% reduction in Kerdock risk** at the same node budget. It is not a wall-time or equal-cost theorem.

This does not prove that Kerdock is globally optimal for every white-box neural expectation estimator. It proves that the particular problem Kerdock solves - fixed spherical nodes, fixed linear weights, no adaptation to the realized network - is essentially exhausted.

## The integration problem

The target is the expected activation of a bias-free deep ReLU network under a standard Gaussian input. Positive homogeneity gives

\[
\mathbb E[f(X)]=\mathbb E[R]\,\mathbb E[f(U)],
\]

where \(U\) is uniform on the sphere and \(R\) is the independent Gaussian radius. The difficult part is therefore spherical integration.

The baseline uses 129 complete real mutually unbiased bases in dimension 256: 128 Kerdock/Walsh-Hadamard chirp bases plus the coordinate basis. Evaluating all vectors and their antipodes gives 66,048 points.

Why is this design strong? Every complete basis is isotropic, antipodal pairing removes odd structure, and mutual unbiasedness makes all cross-basis inner products have the same magnitude. Low-degree angular components cancel exactly; the remaining error is a high-degree deterministic cubature phase.

## Two different near-optimality statements

### 1. Nonnegative rules: at most 0.0233% room

For the infinite-width depth-32 ReLU kernel, the average squared integration error is an RKHS discrepancy. The kernel has a nonnegative Gegenbauer expansion, so a Delsarte-style positive-definite minorant gives a lower bound for every nonnegative probability rule.

The complete auxiliary optimization turns out to have a unique degree-five solution. It is a Hermite interpolant touching the ReLU kernel at three algebraic points. Exact Gaussian quadrature supplies the dual certificate; exact recurrences eliminate every higher Gegenbauer degree; directed interval arithmetic proves the sixth derivative is positive; and an interval linear-system check proves all five nonconstant coefficients are positive.

The resulting ratio is

\[
1\le
\frac{R(Q_{\mathrm{Kerdock}})}{\inf_QR(Q)}
\le1.0002332417295004.
\]

So Kerdock may be exactly optimal, or may be slightly suboptimal, but the maximum certified excess is only 0.0233242%.

### 2. Signed rules: at most 6.294% risk reduction in the audited replay

Signed weights are much harder because the usual positive-definite argument no longer works. The key replacement is an inertia constraint.

For a harmonic comparison profile, the atomic moment matrix has the form

\[
M=E^TWE,
\]

where \(W\) is diagonal with the cubature weights. The number of positive eigenvalues of \(M\) cannot exceed the number of positive weights. If a symmetric matrix has trace \(T>0\) and at most \(p\) positive eigenvalues, then

\[
\|M\|_F^2\ge T^2/p.
\]

Negative eigenvalues do not help carry positive trace; they only add squared norm.

Combining this lemma with the released frozen degree-280 comparison witness gives

\[
R(Q)\ge0.9370601683665084\,R(Q_{\mathrm{Kerdock}}).
\]

Equivalently,

\[
\frac{R(Q_{\mathrm{Kerdock}})}{R(Q)}\le1.0671673322143325.
\]

The factor minus one is 6.7167%, but that is **not** the percentage reduction in Kerdock risk. The actual maximum reduction is `1 - 0.937060168... = 6.293983...%`. A marginally stronger reoptimized constant was reported later, but its rational witness was not recovered, so it is not the public headline.

## Many negative-weight support entries are especially bad

The same argument strengthens as the number of negative-weight support entries increases. After duplicate locations are consolidated and zero weights removed:

- at least 1,072 negative-weight support entries: less than 1.05x possible Kerdock-to-rule factor;
- at least 4,160 entries: certified worse than Kerdock;
- at least 8,192 entries: substantially worse.

Any static signed rule near the audited 1.067168x boundary would need very few negative-weight support entries. This count does not control the magnitude of total negative mass, and the theorem does not say such a good rule exists.

## The proof method itself hits a wall

The signed certificate originally came from rank and harmonic block-trace constraints. A natural next idea was to force all comparison profiles to share one common harmonic Gram matrix.

That stronger-looking relaxation is still exactly sharp. One can explicitly construct a common rank-\(N\) positive-semidefinite matrix that attains every separate rank floor simultaneously. So more second-moment coupling cannot improve the proof.

Actual point evaluations are more constrained. Equality would force exactly \(N\) equal positive weights and pairwise zeros of the comparison kernel - a spherical zero code. Two active profiles require common zeros of consecutive Gegenbauer polynomials, which is impossible. Therefore no atomic rule attains the older abstract floor.

But strict nonattainment is not automatically a uniform numerical gap for signed rules: positive and negative atoms can coalesce while total variation diverges. A stronger theorem must use genuine point-evaluation realizability, sphere-ideal identities, or an explicit total-variation/conditioning bound.

## What this does and does not close

It closes, at winning scale:

- large gains from better static node sets at the same budget;
- better nonnegative static weights;
- dense signed cancellation;
- more optimization inside the same abstract harmonic rank/trace relaxation.

It does not close:

- finite-width-specific structure;
- nodes or weights chosen from the realized network;
- nonlinear processing of basis outputs;
- network-dependent analytic controls;
- transformed residual problems whose kernel must be recertified;
- a cheaper static rule that accepts slightly worse raw MSE but wins on total compute.

That distinction is the main point. “Kerdock is optimal” is too broad. “Kerdock essentially exhausts static network-independent linear cubature at this node budget for the limiting deep-ReLU kernel” is both strong and defensible.

## Why I am releasing the ledger and proof artifacts

The project generated far more than one theorem: many exact identities, failed estimators, oracle capacity studies, cost corrections, and scope repairs. I am publishing the canonical experiment ledger and the available code/certificates so other people can:

- independently reconstruct the interval stack;
- find an error in the certificate;
- construct a rule closer to the signed floor;
- build an actual finite-width or adaptive escape;
- recover the missing exact baseline package;
- avoid repeating the many closed static branches.

The audited repository distinguishes proved, computer-assisted, clean-checkout replayed, arithmetic-checked, reported-only, oracle, closed, and open claims. It also includes the corrected v5.2 theorem record, endpoint certificate, a second T16 interval implementation result, and a strict release checker. It also states what is missing rather than replacing unavailable artifacts with nearby versions.

The most useful outcome would not be everyone agreeing with the result. It would be someone tying or beating the baseline for a mathematically clear reason.
