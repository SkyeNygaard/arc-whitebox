import json,glob,joblib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/work/new_opportunities/t4_fresh_exact');POL=Path('/mnt/data/work/new_opportunities/anchor_fresh');spec=json.load(open(POL/'FROZEN_ANCHOR_SALVAGE_POLICY.json'));clf=joblib.load(POL/'anchor_rf_classifier.joblib');F=spec['features'];p=spec['action_policy'];alpha=spec['active_scale'];pth=spec['probability_threshold']
def mse(x,y):return float(np.mean((x-y)**2))
def unb(x,a,b):return float(np.mean((x-a)*(x-b)))
rows=[]
for rp in sorted((ROOT/'results/records').glob('network_74*_rotation_*.json')):
 r=json.load(open(rp));nid=int(r['network_id']);
 if nid not in spec['fresh_validation_networks']:continue
 z=np.load(ROOT/'results/vectors'/r['vectors_file']);base=z['basefull'].astype(float);zero=z['reduced_base'].astype(float);c17=z['p128_projection'].astype(float);p4=z['p4_correction'].astype(float);truth=z['truth_y'].astype(float);A=z['refA_y'].astype(float);B=z['refB_y'].astype(float);f=r['features']
 agree=min(f['cos_c17_p2'],f['cos_c17_p4']);stable=f['cos_p2_p4']>=p['t_pair'] and p['norm_lo']<=f['norm_p4_c17']<=p['norm_hi']
 action='absolute' if stable and agree>=p['t_abs'] else 'paired_p4' if stable else 'zero'
 target=zero+c17 if action=='absolute' else zero+p['beta']*p4 if action=='paired_p4' else zero
 corr=target-base;X=np.array([[f['cos_c17_p2'],f['cos_c17_p4'],f['cos_p2_p4'],f['norm_p2_c17'],f['norm_p4_c17'],f['min_nested_cos'],f['max_loo_angle_sin'],f['cos_p32_p128'],f['norm_p32_p128']]])
 prob=float(clf.predict_proba(X)[0,1]);a=alpha if prob>=pth else 0.;pred=base+a*corr
 inn=float(np.mean((base-truth)*corr));nn=float(np.mean(corr*corr));ao=float(np.clip(-inn/max(nn,1e-30),0,2));oracle=base+ao*corr
 rows.append({'network_id':nid,'rotation_index':int(r['rotation_index']),'action':action,'probability':prob,'active':bool(a),'alpha':a,'baseline_mse':mse(base,truth),'candidate_mse':mse(pred,truth),'baseline_unbiased_mse':unb(base,A,B),'candidate_unbiased_mse':unb(pred,A,B),'original_policy_unbiased_mse':unb(target,A,B),'oracle_clip_unbiased_mse':unb(oracle,A,B),'oracle_alpha':ao,'reference_noise_mse':mse(A,B)/4})
d=pd.DataFrame(rows);d.to_csv(POL/'FRESH_VALIDATION_ROWS.csv',index=False)
def summary(col):
 B=d.baseline_unbiased_mse.to_numpy();C=d[col].to_numpy();rat=[]
 for n,g in d.groupby('network_id'):rat.append(g[col].sum()/g.baseline_unbiased_mse.sum())
 rng=np.random.default_rng(20260730);bn=np.array([g.baseline_unbiased_mse.sum() for _,g in d.groupby('network_id')]);cn=np.array([g[col].sum() for _,g in d.groupby('network_id')]);ix=rng.integers(0,len(bn),(100000,len(bn)));gain=bn[ix].sum(1)/cn[ix].sum(1)
 return {'candidate_over_base':float(C.sum()/B.sum()),'gain':float(B.sum()/C.sum()),'row_wins':int((C<B).sum()),'network_wins':int((np.array(rat)<1).sum()),'median_network_ratio':float(np.median(rat)),'worst_network_ratio':float(np.max(rat)),'network_bootstrap_gain_ci95':[float(np.quantile(gain,.025)),float(np.quantile(gain,.975))]}
out={'policy_sha256':spec['classifier']['sha256'],'network_ids':sorted(d.network_id.unique().tolist()),'n_rows':len(d),'active_rows':int(d.active.sum()),'mean_probability':float(d.probability.mean()),'frozen_policy':summary('candidate_unbiased_mse'),'original_unscaled_policy':summary('original_policy_unbiased_mse'),'oracle_scalar_same_direction':summary('oracle_clip_unbiased_mse'),'reference_noise_fraction_pooled':float(d.reference_noise_mse.sum()/d.baseline_mse.sum())}
gate={'ratio':out['frozen_policy']['candidate_over_base']<=.98,'ci':out['frozen_policy']['network_bootstrap_gain_ci95'][0]>1,'wins':out['frozen_policy']['network_wins']>=5,'worst':out['frozen_policy']['worst_network_ratio']<=1.10};out['gate_components']=gate;out['passed']=all(gate.values());(POL/'FRESH_VALIDATION_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print(d[['network_id','rotation_index','action','probability','active','candidate_unbiased_mse','baseline_unbiased_mse']].to_string(index=False))
