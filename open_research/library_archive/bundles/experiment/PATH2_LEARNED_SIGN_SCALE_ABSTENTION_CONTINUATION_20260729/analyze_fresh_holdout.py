from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent; P=ROOT/'fresh_scale_holdout_v1'
D=pd.read_csv(P/'fresh_screen_rows.csv').sort_values(['network_seed','rotation_seed']).reset_index(drop=True)
rows=[]
for _,z in D.iterrows():
 v=np.load(P/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz')
 base=v['base'];truth=.5*(v['truth_half1']+v['truth_half2']);e=base-truth
 rec={}
 for p in [32,128]:
  dd=v[f'direct{p}']-base;a0=float(np.mean(e*e));lin=float(2*np.mean(e*dd));q=float(np.mean(dd*dd));rec|={f'a0_{p}':a0,f'lin_{p}':lin,f'q_{p}':q,f'alpha_opt_{p}':float(-lin/(2*q))}
 rows.append(rec)
D=pd.concat([D,pd.DataFrame(rows)],axis=1)
assert len(D)==36 and D.network_seed.nunique()==12

def metrics(p,alpha,boot=True):
 a=np.asarray(alpha,float);b=D[f'a0_{p}'].to_numpy();m=b+a*D[f'lin_{p}'].to_numpy()+a*a*D[f'q_{p}'].to_numpy();r=m/b
 out={'pooled_ratio':float(m.sum()/b.sum()),'mean_ratio':float(r.mean()),'median_ratio':float(np.median(r)),'worst':float(r.max()),'p90':float(np.quantile(r,.9)),'wins':int((r<1).sum()),'alpha_mean':float(a.mean()),'alpha_min':float(a.min()),'alpha_max':float(a.max()),'ratios':r.tolist(),'alphas':a.tolist()}
 out['per_rotation']={str(rot):{'pooled_ratio':float(m[D.rotation_seed.eq(rot)].sum()/b[D.rotation_seed.eq(rot)].sum()),'worst':float(r[D.rotation_seed.eq(rot)].max()),'wins':int((r[D.rotation_seed.eq(rot)]<1).sum())} for rot in sorted(D.rotation_seed.unique())}
 if boot:
  groups=sorted(D.network_seed.unique());gb=np.array([b[D.network_seed.eq(g)].sum() for g in groups]);gm=np.array([m[D.network_seed.eq(g)].sum() for g in groups]);rng=np.random.default_rng(2026072902);ix=rng.integers(0,len(groups),size=(100000,len(groups)));vals=gm[ix].sum(1)/gb[ix].sum(1)
  out['grouped_bootstrap_ci95']=[float(x) for x in np.quantile(vals,[.025,.5,.975])];out['prob_below_0_595']=float(np.mean(vals<=.595));out['prob_below_1']=float(np.mean(vals<1))
 return out
risk=D.risk_n129.to_numpy()
pol={
 'k32_full_alpha1':metrics(32,np.ones(len(D))),
 'k32_fixed_045':metrics(32,np.full(len(D),.45)),
 'k32_fixed_05':metrics(32,np.full(len(D),.5)),
 'k32_frozen_l08_04_05':metrics(32,np.where(risk>0.00279066317598334,.4,.5)),
 'k32_bounded_q925_045_055':metrics(32,np.where(risk>0.0027637032840478275,.45,.55)),
 'k32_oracle_scale':metrics(32,D.alpha_opt_32.to_numpy(),False),
 'k128_fixed_05':metrics(128,np.full(len(D),.5)),
 'k128_full_alpha1':metrics(128,np.ones(len(D))),
 'k128_oracle_scale':metrics(128,D.alpha_opt_128.to_numpy(),False),
}
out={'protocol':json.load(open(P/'IMMUTABLE_HOLDOUT_PROTOCOL.json')),'n_rows':len(D),'n_groups':int(D.network_seed.nunique()),'risk_summary':D.risk_n129.describe().to_dict(),'alpha32_summary':D.alpha_opt_32.describe().to_dict(),'alpha128_summary':D.alpha_opt_128.describe().to_dict(),'policies':pol,'rows':D[['network_seed','rotation_seed','risk_n129','k32_ratio','k128_ratio','alpha_opt_32','alpha_opt_128']].to_dict('records')}
(ROOT/'results'/'FRESH_SCALE_HOLDOUT_RESULTS.json').write_text(json.dumps(out,indent=2))
print(pd.DataFrame([{'policy':k,**{x:v[x] for x in ['pooled_ratio','worst','p90','wins','alpha_mean']},'ci_lo':v.get('grouped_bootstrap_ci95',[None,None,None])[0],'ci_hi':v.get('grouped_bootstrap_ci95',[None,None,None])[-1]} for k,v in pol.items()]).sort_values('pooled_ratio').to_string(index=False))
