# Pair-target derivation

For frozen probe $p$, let $u_p=e_{i_p}$, right direction $v_p$, sample pointwise center $m$, Gaussian mean $\mu$, raw second moment $M=\mathbb E[hh^T]$, and connected cubic contraction $c_p=e_{i_p}^T C_3 v_p$. Define

\[
d=\mu-m,\quad a_p=v_p^Td,\quad z_p=v_p^T\mu,\quad s_p=M_{i_pi_p},\quad t_p=M_{i_p,:}v_p.
\]

The exact pointwise-centered radial-Hermite anchor contraction is

\[
A_p(m)=\frac{c_p+s_pa_p+2d_{i_p}t_p+2(m_{i_p}^2-\mu_{i_p}^2)z_p}{D+1}.
\]

Dropping the empirically neutral connected defect leaves the lower recentering target

\[
\ell_p=\frac{s_pa_p+2d_{i_p}t_p+2(m_{i_p}^2-\mu_{i_p}^2)z_p}{D+1}.
\]

## Required decomposition

- **Mean-projection term:** $s_pa_p/(D+1)$, requiring $a_p=v_p^T(\mu-m)$.
- **Marginal-second-moment term:** the same product viewed through $s_p=M_{ii}$.
- **Row-direction pair term:** $2d_it_p/(D+1)$.
- **Center-induced linear term:** $2d_it_p/(D+1)-4m_id_iz_p/(D+1)$ after expanding $m_i^2-\mu_i^2=-2m_id_i-d_i^2$.
- **Center-induced diagonal-quadratic term:** $-2d_i^2z_p/(D+1)$.
- **Optional connected cubic:** $c_p/(D+1)$; excluded from the retained estimator.

The local scalar-error coefficients are

\[
\partial_{a_p}\ell_p=\frac{s_p}{D+1},\quad
\partial_{s_p}\ell_p=\frac{a_p}{D+1},\quad
\partial_{t_p}\ell_p=\frac{2d_i}{D+1},
\]
\[
\partial_{d_i}\ell_p=\frac{2t_p-4\mu_i z_p}{D+1},\quad
\partial_{z_p}\ell_p=\frac{2(m_i^2-\mu_i^2)}{D+1}.
\]

For frozen cross-fit coefficient row $\beta_p\in\mathbb R^{256}$, every scalar anchor error $e_p$ maps exactly to

\[
c_{\rm out}=\sum_p e_p\beta_p=e^T\beta.
\]

The authoritative metric is therefore output-space quadratic loss. For baseline error $r$,

\[
\|r+c_{\rm out}\|^2=\|r\|^2+2\langle r,c_{\rm out}\rangle+\|c_{\rm out}\|^2.
\]

## Consequence

Pair moments do not create an absolute correction by themselves: their coefficients $a_p$ and $d_i$ vanish when the center defect is unavailable. They modulate the center-driven correction. This is why pair terms are algebraically necessary but need not be independently estimated.
