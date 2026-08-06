from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=256; DEPTH=32; ROWS=192; EPS=1e-3; NETWORKS=12
LAYERS=[0,4,8,12,16,20,24,26,28,30,31]
MASTER_SEED=202607301

def suffix(state: np.ndarray, k: int, weights: list[np.ndarray]) -> np.ndarray:
    z=state.astype(np.float32,copy=True)
    for w in weights[k+1:]:
        z=np.maximum(z@w,0).astype(np.float32,copy=False)
    return z.mean(0,dtype=np.float64)

all_rows=[]
for ni in range(NETWORKS):
    seed=MASTER_SEED+1009*ni
    rng=np.random.default_rng(seed)
    weights=[(rng.standard_normal((D,D))*math.sqrt(2/D)).astype(np.float32) for _ in range(DEPTH)]
    x=rng.standard_normal((ROWS,D)).astype(np.float32)
    acts=[]; a=x
    for w in weights:
        a=np.maximum(a@w,0).astype(np.float32,copy=False); acts.append(a.copy())
    base_mean=acts[-1].mean(0,dtype=np.float64)
    base_norm=np.linalg.norm(base_mean)
    for k in LAYERS:
        ak=acts[k]; muk=ak.mean(0,dtype=np.float64); mu_norm=np.linalg.norm(muk)
        out_scale=suffix(ak*(1+EPS),k,weights) if k<31 else (ak*(1+EPS)).mean(0,dtype=np.float64)
        scale_expected=(1+EPS)*base_mean
        scale_err=np.linalg.norm(out_scale-scale_expected)/max(np.linalg.norm(scale_expected),1e-30)
        dmu=EPS*muk
        out_mu=suffix(ak+dmu[None,:],k,weights) if k<31 else (ak+dmu[None,:]).mean(0,dtype=np.float64)
        input_rel=np.linalg.norm(dmu)/max(mu_norm,1e-30)
        output_rel=np.linalg.norm(out_mu-base_mean)/max(base_norm,1e-30)
        gain_mu=output_rel/max(input_rel,1e-30)
        v=rng.standard_normal(D); v/=np.linalg.norm(v); dr=v*(EPS*mu_norm)
        out_r=suffix(ak+dr[None,:],k,weights) if k<31 else (ak+dr[None,:]).mean(0,dtype=np.float64)
        gain_r=(np.linalg.norm(out_r-base_mean)/max(base_norm,1e-30))/EPS
        centered=ak-muk[None,:]
        v2=rng.standard_normal(D); v2/=np.linalg.norm(v2)
        delta=EPS*np.outer(centered@v2,v2)
        pert=ak+delta.astype(np.float32)
        rms0=np.linalg.norm(centered)/math.sqrt(centered.size)
        rmsd=np.linalg.norm(delta)/math.sqrt(delta.size)
        shape_input_rel=rmsd/max(rms0,1e-30)
        out_shape=suffix(pert,k,weights) if k<31 else pert.mean(0,dtype=np.float64)
        gain_shape=(np.linalg.norm(out_shape-base_mean)/max(base_norm,1e-30))/max(shape_input_rel,1e-30)
        all_rows.append({'network_index':ni,'seed':seed,'layer':k+1,'scale_relative_error':float(scale_err),'mean_shift_gain':float(gain_mu),'random_shift_gain':float(gain_r),'rank1_shape_gain':float(gain_shape),'rank1_shape_input_rel':float(shape_input_rel)})

metrics=['scale_relative_error','mean_shift_gain','random_shift_gain','rank1_shape_gain']
summary={}
for metric in metrics:
    vals=np.array([r[metric] for r in all_rows],dtype=float)
    summary[metric]={'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9)),'p99':float(np.quantile(vals,.99)),'max':float(vals.max())}
layer_summary=[]
for layer in sorted(set(r['layer'] for r in all_rows)):
    rr=[r for r in all_rows if r['layer']==layer]
    row={'layer':layer}
    for metric in metrics:
        vals=np.array([r[metric] for r in rr],dtype=float)
        row[metric+'_median']=float(np.median(vals)); row[metric+'_max']=float(vals.max())
    layer_summary.append(row)
res={'scope':f'{NETWORKS} fresh architecture-matched width-256 depth-32 synthetic networks; {ROWS} Gaussian rows/network; diagnostic only','master_seed':MASTER_SEED,'networks':NETWORKS,'rows_per_network':ROWS,'epsilon':EPS,'records':all_rows,'summary':summary,'layer_summary':layer_summary,'gate_any_tested_gain_over_1_5':bool(max(r['mean_shift_gain'] for r in all_rows)>1.5 or max(r['random_shift_gain'] for r in all_rows)>1.5 or max(r['rank1_shape_gain'] for r in all_rows)>1.5)}
(ROOT/'results'/'TEST3_TRANSFER_PROBE.json').write_text(json.dumps(res,indent=2))
print(json.dumps({k:v for k,v in res.items() if k!='records'},indent=2))
