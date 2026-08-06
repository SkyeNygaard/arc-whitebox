#!/usr/bin/env python3
"""Independent numerical/property checks for the Agent 6 theorem package."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('/mnt/data/agent6_anchor_review/THEOREM_CHECKS.json'))
    ap.add_argument('--seed', type=int, default=6020260729)
    ap.add_argument('--relu-cases', type=int, default=5_000_000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {}

    # 1. Correction-risk identity in a high-dimensional random ensemble.
    n, d = 20000, 37
    e = rng.standard_normal((n, d))
    u = 0.3 * e + rng.standard_normal((n, d))
    R0 = np.mean(np.sum(e*e, axis=1))
    C = np.mean(np.sum(e*u, axis=1))
    U = np.mean(np.sum(u*u, axis=1))
    alphas = np.linspace(-2, 2, 401)
    direct = np.array([np.mean(np.sum((e-a*u)**2, axis=1)) for a in alphas])
    formula = R0 - 2*alphas*C + alphas**2*U
    out['correction_risk_max_abs_error'] = float(np.max(np.abs(direct-formula)))
    out['correction_alpha_formula'] = float(C/U)
    out['correction_alpha_grid'] = float(alphas[np.argmin(direct)])

    # 2. Conditional selector, including the positive-only refinement.
    g = rng.integers(0, 11, size=n)
    group_coeff = np.linspace(-0.8, 0.8, 11)
    u_select = group_coeff[g, None] * e + 0.8 * rng.standard_normal((n, d))
    cond_records = []
    gain_unconstrained = 0.0
    gain_positive = 0.0
    gain_bounded = 0.0
    A = 0.7
    for k in range(11):
        m = g == k
        ck = float(np.mean(np.sum(e[m]*u_select[m], axis=1)))
        uk = float(np.mean(np.sum(u_select[m]*u_select[m], axis=1)))
        p = float(np.mean(m))
        a_un = ck/uk
        a_pos = max(0.0, a_un)
        a_bound = min(A, a_pos)
        gain_unconstrained += p*(2*a_un*ck-a_un*a_un*uk)
        gain_positive += p*(2*a_pos*ck-a_pos*a_pos*uk)
        gain_bounded += p*(2*a_bound*ck-a_bound*a_bound*uk)
        cond_records.append({'group':k,'C':ck,'U':uk,'a_un':a_un,'a_pos':a_pos,'a_bound':a_bound})
    # Directly verify with per-row selected alpha.
    amap_un = np.array([cond_records[k]['a_un'] for k in g])
    amap_pos = np.array([cond_records[k]['a_pos'] for k in g])
    amap_bound = np.array([cond_records[k]['a_bound'] for k in g])
    out['selector_gain_formula_unconstrained'] = gain_unconstrained
    out['selector_gain_direct_unconstrained'] = float(R0-np.mean(np.sum((e-amap_un[:,None]*u_select)**2,axis=1)))
    out['selector_gain_formula_positive'] = gain_positive
    out['selector_gain_direct_positive'] = float(R0-np.mean(np.sum((e-amap_pos[:,None]*u_select)**2,axis=1)))
    out['selector_gain_formula_bounded'] = gain_bounded
    out['selector_gain_direct_bounded'] = float(R0-np.mean(np.sum((e-amap_bound[:,None]*u_select)**2,axis=1)))

    # 3. General replacement and correlated-noise shrinkage.
    # Create S and S-perpendicular coordinates explicitly.
    m, ds, dr = 30000, 13, 9
    s = rng.standard_normal((m, ds))
    r = rng.standard_normal((m, dr))
    eta = rng.standard_normal((m, ds))
    nvec = 0.4*s + 0.8*eta
    S = float(np.mean(np.sum(s*s,axis=1)))
    N = float(np.mean(np.sum(nvec*nvec,axis=1)))
    K = float(np.mean(np.sum(s*nvec,axis=1)))
    Rr = float(np.mean(np.sum(r*r,axis=1)))
    denom = S+N+2*K
    astar = (S+K)/denom
    alpha_grid = np.linspace(-0.2,1.4,1601)
    risks = []
    for a in alpha_grid:
        rems = (1-a)*s-a*nvec
        risks.append(Rr+float(np.mean(np.sum(rems*rems,axis=1))))
    out['correlated_shrinkage_alpha_formula'] = astar
    out['correlated_shrinkage_alpha_grid'] = float(alpha_grid[int(np.argmin(risks))])
    out['correlated_shrinkage_risk_formula'] = Rr+S-(S+K)**2/denom
    out['correlated_shrinkage_risk_grid'] = float(np.min(risks))
    out['full_replacement_threshold_lhs_N'] = N
    out['full_replacement_threshold_rhs_S'] = S
    out['full_replacement_direct_risk'] = Rr+N

    # General n not in S: check N - 2<r,n_r> < S.
    n_s = 0.25*s + 0.4*rng.standard_normal((m,ds))
    n_r = 0.30*r + 0.6*rng.standard_normal((m,dr))
    Ngen = float(np.mean(np.sum(n_s*n_s,axis=1)+np.sum(n_r*n_r,axis=1)))
    B = float(np.mean(np.sum(r*n_r,axis=1)))
    direct_gen = float(np.mean(np.sum(n_s*n_s,axis=1)+np.sum((r-n_r)**2,axis=1)))
    formula_gen = Rr + Ngen - 2*B
    out['general_replacement_max_formula_error'] = abs(direct_gen-formula_gen)
    out['general_replacement_improves_direct'] = direct_gen < (Rr+S)
    out['general_replacement_improves_criterion'] = (Ngen-2*B) < S

    # 4. Common-bias observational equivalence and two-point risk identity.
    krep, p = 8, 17
    mu = rng.standard_normal(p)
    b = rng.standard_normal(p)
    shift = rng.standard_normal(p)
    noise = rng.standard_normal((krep,p))
    z1 = mu+b+noise
    z2 = (mu+shift)+(b-shift)+noise
    out['common_bias_max_observation_difference'] = float(np.max(np.abs(z1-z2)))
    x = rng.standard_normal(p)
    lhs = float(np.sum((x-b)**2)+np.sum((x-(b-shift))**2))
    lower = float(np.sum(shift**2)/2)
    out['two_point_deterministic_inequality_slack'] = lhs-lower

    # 5. ReLU scalar crossing lemma: broad randomized property test.
    max_violation = -math.inf
    max_identity_error = 0.0
    remaining = args.relu_cases
    chunk = 500_000
    cases = 0
    while remaining:
        qn = min(chunk, remaining)
        z = rng.standard_normal(qn)*rng.lognormal(mean=0.0,sigma=1.0,size=qn)
        t = rng.standard_normal(qn)*rng.lognormal(mean=-4.0,sigma=2.0,size=qn)
        phi_z = np.maximum(z,0.0)
        phi_zt = np.maximum(z+t,0.0)
        lin = (z>0)*t
        rem = phi_zt-phi_z-lin
        bound = np.abs(t)*(np.abs(z)<=np.abs(t))
        max_violation = max(max_violation,float(np.max(np.abs(rem)-bound)))
        max_identity_error = max(max_identity_error,float(np.max(np.abs(phi_zt-(phi_z+lin+rem)))))
        remaining -= qn
        cases += qn
    edges_z = np.array([0.,0.,1.,-1.,1.,-1.])
    edges_t = np.array([1.,-1.,-2.,2.,-0.5,0.5])
    rem = np.maximum(edges_z+edges_t,0)-np.maximum(edges_z,0)-(edges_z>0)*edges_t
    bound = np.abs(edges_t)*(np.abs(edges_z)<=np.abs(edges_t))
    max_violation=max(max_violation,float(np.max(np.abs(rem)-bound)))
    out['relu_cases_checked'] = cases+len(edges_z)
    out['relu_max_bound_violation'] = max_violation
    out['relu_max_identity_error'] = max_identity_error

    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
