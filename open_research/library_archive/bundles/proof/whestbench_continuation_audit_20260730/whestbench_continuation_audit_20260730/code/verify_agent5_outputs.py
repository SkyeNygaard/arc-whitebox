from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
A=Path('/mnt/data/whestbench_continuation_20260730/agent5/compact_bundle')
res=json.load(open(A/'report/FINAL_RESULTS.json'))
checks={}

def close(a,b,tol=2e-12):return abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b)))
# Stein validation directly from 16 row-level JSONs.
rows=[json.load(open(p)) for p in sorted((A/'results/stein').glob('validation_network_*.json'))]
rng=np.random.default_rng(20260730)
def method(label,mode,alpha):
 b=np.array([r['base_unbiased_mse'] for r in rows]); inn=np.array([r['candidates'][label][mode]['inner'] for r in rows]); nn=np.array([r['candidates'][label][mode]['norm_sq'] for r in rows]); m=b+2*alpha*inn+alpha*alpha*nn
 ratios=m/b; ix=rng.integers(0,len(rows),(100000,len(rows))); gain=b[ix].sum(1)/m[ix].sum(1)
 return {'candidate_over_base':float(m.sum()/b.sum()),'gain_base_over_candidate':float(b.sum()/m.sum()),'bootstrap_gain_ci95':[float(np.quantile(gain,.025)),float(np.quantile(gain,.975))], 'wins':int((m<b).sum()),'median_ratio':float(np.median(ratios)),'worst_ratio':float(np.max(ratios)),'ratios':ratios.tolist()}
specs=[('stein_tanh_k8_stable','full',2.0),('stein_tanh_k8_stable','full',1.0),('harmonic_d68_k8','cf4',1.0),('harmonic_d68_k8','full',1.0)]
for spec,exp in zip(specs,res['stein_validation']['methods']):
 got=method(*spec)
 for k in ['candidate_over_base','gain_base_over_candidate','median_ratio','worst_ratio']:
  assert close(got[k],exp[k]),(spec,k,got[k],exp[k])
 assert got['wins']==exp['wins']; assert np.allclose(got['bootstrap_gain_ci95'],exp['bootstrap_gain_ci95'],rtol=2e-12,atol=2e-12); assert np.allclose(got['ratios'],exp['ratios'],rtol=2e-12,atol=2e-12)
checks['stein_validation']={'n':len(rows),'primary':res['stein_validation']['methods'][0],'recomputed_all_methods':True}
# Fresh anchor aggregates from published row-level CSV (vectors omitted from compact bundle).
df=pd.read_csv(A/'results/anchor/FRESH_VALIDATION_ROWS.csv')
def summary(col):
 B=df.baseline_unbiased_mse.to_numpy();C=df[col].to_numpy(); ratios=[]
 for _,g in df.groupby('network_id'):ratios.append(g[col].sum()/g.baseline_unbiased_mse.sum())
 rng=np.random.default_rng(20260730); bn=np.array([g.baseline_unbiased_mse.sum() for _,g in df.groupby('network_id')]); cn=np.array([g[col].sum() for _,g in df.groupby('network_id')]);ix=rng.integers(0,len(bn),(100000,len(bn)));gain=bn[ix].sum(1)/cn[ix].sum(1)
 return {'candidate_over_base':float(C.sum()/B.sum()),'gain':float(B.sum()/C.sum()),'row_wins':int((C<B).sum()),'network_wins':int((np.array(ratios)<1).sum()),'median_network_ratio':float(np.median(ratios)),'worst_network_ratio':float(np.max(ratios)),'network_bootstrap_gain_ci95':[float(np.quantile(gain,.025)),float(np.quantile(gain,.975))]}
for col,name in [('candidate_unbiased_mse','frozen_policy'),('original_policy_unbiased_mse','original_unscaled_policy'),('oracle_clip_unbiased_mse','oracle_scalar_same_direction')]:
 got=summary(col); exp=res['anchor_fresh_validation'][name]
 for k in ['candidate_over_base','gain','median_network_ratio','worst_network_ratio']:
  assert close(got[k],exp[k]),(name,k,got[k],exp[k])
 assert got['row_wins']==exp['row_wins'] and got['network_wins']==exp['network_wins']; assert np.allclose(got['network_bootstrap_gain_ci95'],exp['network_bootstrap_gain_ci95'],rtol=2e-12,atol=2e-12)
checks['anchor_fresh']={'n_rows':len(df),'frozen':summary('candidate_unbiased_mse'),'oracle':summary('oracle_clip_unbiased_mse'),'raw_vectors_in_bundle':False}
# Signed weight aggregate is direct table audit.
sw=json.load(open(A/'results/signed/SIGNED_WEIGHT_AUDIT.json'))
checks['signed_weight_audit']={'keys':list(sw)[:10],'source':'saved per-network weight audit; full network rows not included'}
out={'passed':True,'checks':checks,'reproducibility_note':'Stein and anchor aggregate headlines recomputed from saved row-level outputs. Compact bundle omits fresh-anchor vector NPZs and upstream T4/ARC dependencies, so generation/evaluation is not fully self-contained.'}
P=Path('/mnt/data/whestbench_continuation_20260730/local_verification/agent5_independent_verification.json');P.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
