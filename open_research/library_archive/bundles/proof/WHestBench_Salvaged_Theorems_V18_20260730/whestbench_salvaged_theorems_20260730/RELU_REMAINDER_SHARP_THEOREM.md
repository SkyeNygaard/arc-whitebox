# ReLU remainder salvage — exact crossing formula and sharp cubic bound

**Status:** analytically proved. This strengthens the hostile-patched `2L|t|^3` bound by a factor of six.

For `phi(z)=max(z,0)`, define the first-order gate remainder

\[
r(z,t)=\phi(z+t)-\phi(z)-\mathbf1_{\{z>0\}}t.
\]

## Exact pointwise formula

The remainder is nonnegative and supported exactly on a gate crossing:

For `t>=0`,

\[
r(z,t)=(z+t)\mathbf1_{\{-t\le z\le0\}}.
\]

For `t<0`,

\[
r(z,t)=-(z+t)\mathbf1_{\{0<z\le -t\}}.
\]

In particular,

\[
0\le r(z,t)\le |t|\mathbf1_{\{|z|\le|t|\}}.
\]

Without any density assumption, this already gives the distribution-free conditional bound

\[
\mathbb E[r(Z,T)^2\mid T]
\le |T|^2\Pr(\text{the segment from }Z\text{ to }Z+T\text{ crosses }0\mid T).
\]

This form remains valid with atoms and is the correct fallback when a density certificate is unavailable.

## Exact second moment

If `Z` has density `p`, then for `t>0`,

\[
\mathbb E r(Z,t)^2
=
\int_{-t}^0(z+t)^2p(z)\,dz,
\]

and for `t<0`,

\[
\mathbb E r(Z,t)^2
=
\int_0^{|t|}(|t|-z)^2p(z)\,dz.
\]

If the density is bounded by `L_t` on the actual crossing interval, then

\[
\mathbb E r(Z,t)^2
\le
\frac{L_t}{3}|t|^3.
\]

The constant `1/3` follows by integrating the squared triangular crossing profile and is sharp to first order.

If `p` is continuous at zero, then

\[
\lim_{t\downarrow0}\frac{\mathbb E r(Z,t)^2}{t^3}
=\frac{p(0^-)}3,
\qquad
\lim_{t\uparrow0}\frac{\mathbb E r(Z,t)^2}{|t|^3}
=\frac{p(0^+)}3.
\]

For a continuous density the common limit is `p(0)/3`.

## Random and dependent perturbations

Let `T` be random. If the conditional density of `Z` given `T` is bounded by `L(T)` on the corresponding crossing interval, then

\[
\mathbb E r(Z,T)^2
\le
\frac13\mathbb E[L(T)|T|^3].
\]

No independence between `Z` and `T` is needed; the condition is explicitly conditional.

For Gaussian `Z` with conditional standard deviation at least `sigma`, the global density bound gives

\[
\mathbb E r(Z,T)^2
\le
\frac{1}{3\sigma\sqrt{2\pi}}\mathbb E|T|^3.
\]

## Vector form

For coordinatewise ReLU with perturbation vector `T`, if each conditional coordinate density has crossing-interval bound at most `L`, then

\[
\mathbb E\|r(Z,T)\|_2^2
\le
\frac L3\mathbb E\sum_j|T_j|^3
\le
\frac L3\mathbb E\|T\|_2^3.
\]

After a downstream linear map `J`,

\[
\mathbb E\|Jr(Z,T)\|^2
\le
\frac{\|J\|_{\rm op}^2L}{3}
\mathbb E\|T\|_2^3.
\]

## Exact nonlinear improvement gate

If the baseline error has risk `R0`, the linearly corrected error has risk `Rlin`, and the downstream gate remainder has second moment at most `delta^2`, then

\[
R_{\rm exact}
\le(\sqrt{R_{\rm lin}}+\delta)^2.
\]

A sufficient exact-ReLU improvement condition is therefore

\[
\delta<\sqrt{R_0}-\sqrt{R_{\rm lin}}.
\]

The sharp cubic bound above supplies a legal, direction-sensitive way to certify `delta`.
