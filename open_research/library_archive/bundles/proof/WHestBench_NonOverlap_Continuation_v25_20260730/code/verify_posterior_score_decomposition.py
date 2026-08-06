#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi

D=256; B=129
MULT=np.array([1,128,32895,256,32768],dtype=np.float64)
LAM_C=np.array([
64293.0127076907740,
0.0199556322060796393,
0.0251746480145940471,
0.796825266924534112,
0.0219825945011625946,
],dtype=np.float64)
LAM_D=np.array([
62584.6253844085647,
2.9882e-14,
4.2248036872e-10,
9.1827509366e-6,
2.7655590677e-12,
],dtype=np.float64)
RK=2.4336603575430052e-7


def kappa(t: np.ndarray) -> np.ndarray:
    t=np.clip(t,-1.0,1.0)
    return (np.sqrt(np.maximum(0.0,1.0-t*t))+t*(np.pi-np.arccos(t)))/np.pi

def c31(t: np.ndarray) -> np.ndarray:
    y=np.asarray(t,dtype=np.float64)
    for _ in range(31): y=kappa(y)
    return y

def hdim(l:int)->int:
    return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)

def g_sequence(t:np.ndarray,L:int)->list[np.ndarray]:
    out=[np.ones_like(t)]
    if L:
        out.append(t.copy())
    for n in range(1,L):
        out.append(((2*n+D-2)/(n+D-2))*t*out[n]-(n/(n+D-2))*out[n-1])
    return out

def assoc_eigs(vals:np.ndarray)->np.ndarray:
    f1,fm1,f0,fa,fma=vals
    fe1=(f1+fm1)/2; fo1=(f1-fm1)/2
    fea=(fa+fma)/2; foa=(fa-fma)/2
    return np.array([
        2*(fe1+(D-1)*f0+(B-1)*D*fea),
        2*(fe1+(D-1)*f0-D*fea),
        2*(fe1-f0),
        2*(fo1+2048*foa),
        2*(fo1-16*foa),
    ],dtype=np.float64)

def main()->None:
    L=10
    qx,qw=roots_jacobi(800,(D-3)/2,(D-3)/2)
    qw=qw/qw.sum()
    gs=g_sequence(qx,L)
    cv=c31(qx)
    coeff=np.array([hdim(l)*np.sum(qw*cv*gs[l]) for l in range(L+1)])
    pts=np.array([1.0,-1.0,0.0,1/16,-1/16])
    gp=g_sequence(pts,L)
    dvals=sum((coeff[l]**2/hdim(l))*gp[l] for l in range(L+1))
    lam_e=np.zeros(5)
    for l in range(L+1):
        lam_e+=(coeff[l]**3/hdim(l)**2)*assoc_eigs(gp[l])
    lam_h=lam_e-LAM_D**2/LAM_C
    pap2=float(np.sum(MULT*(LAM_D/LAM_C)**2))
    d1=float(sum(coeff[l]**2/hdim(l) for l in range(L+1)))
    tr_pa2=float(np.sum(MULT*lam_e/LAM_C))
    b2norm=tr_pa2-pap2
    c2norm=d1-2*tr_pa2+pap2
    residual=2*b2norm+c2norm
    sector=np.asarray(MULT*lam_h/LAM_C)
    global_b=float(sector[0])
    remaining_after_global=residual-2*global_b
    relu_b2_sq=1/(2*np.pi)
    result={
        'title':'Independent posterior-score decomposition reproduction',
        'dimension':D,
        'bases':B,
        'gegenbauer_cutoff':L,
        'coefficients_c31':coeff.tolist(),
        'D_values':dvals.tolist(),
        'lambda_E':lam_e.tolist(),
        'lambda_H':lam_h.tolist(),
        'pap_frobenius_squared':pap2,
        'trace_P_A2':tr_pa2,
        'A_frobenius_squared':d1,
        'B_frobenius_squared':b2norm,
        'C_frobenius_squared':c2norm,
        'posterior_tensor_residual':residual,
        'reported_posterior_tensor_residual':4.9365176285223093e-7,
        'cross_fraction':2*b2norm/residual,
        'pure_hidden_fraction':c2norm/residual,
        'B_sector_contributions':sector.tolist(),
        'global_fraction_of_B':global_b/b2norm,
        'remaining_tensor_after_global_matched_query_upper_bound':remaining_after_global,
        'remaining_relu_second_chaos_risk_upper_bound':relu_b2_sq*remaining_after_global,
        'remaining_ratio_to_Kerdock_upper_bound':relu_b2_sq*remaining_after_global/RK,
        'naive_residual_sample_second_moment_estimate':0.022452655155701946,
        'global_score_variance':global_b,
        'samples_for_0p5_percent_relative_mse':0.022452655155701946/(0.005*global_b),
        'checks':{},
    }
    checks={
        'C31_point_values_match': bool(np.max(np.abs(c31(pts)-np.array([
            1.0,0.9720108731544701844,0.9734181125699380687,
            0.9736160100959505490,0.9732406018244275977])))<5e-15),
        'D_point_values_match': bool(np.max(np.abs(dvals-np.array([
            0.9475627990125502951,0.9475627278256903869,0.9475627632078801566,
            0.9475627654332082036,0.9475627609842023068])))<5e-15),
        'residual_matches_report': bool(abs(residual-4.9365176285223093e-7)<2e-16),
        'all_lambda_H_nonnegative_numerically': bool(np.min(lam_h)>-1e-22),
        'cross_dominates_99p9_percent': bool(2*b2norm/residual>0.999),
        'global_sector_dominates_99_percent_B': bool(global_b/b2norm>0.99),
        'naive_sampling_exceeds_10_million_for_0p5pct': bool(result['samples_for_0p5_percent_relative_mse']>1e7),
    }
    result['checks']=checks
    result['passed']=all(checks.values())
    out=Path(__file__).resolve().parents[1]/'results'/'posterior_score_decomposition.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
