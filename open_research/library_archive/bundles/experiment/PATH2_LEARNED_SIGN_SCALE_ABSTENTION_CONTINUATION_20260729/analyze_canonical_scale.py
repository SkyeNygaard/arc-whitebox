from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
ROOT=Path(__file__).resolve().parent
CAN=ROOT/'canonical24_quadratics_audited'; HARD=ROOT/'hardpanel_quadratics_audited'

def load_panel(panel:Path):
 d=pd.read_csv(panel/'fresh_screen_rows.csv').sort_values(['network_seed','rotation_seed']).reset_index(drop=True)
 raw=[]
 for _,z in d.iterrows():
  p=panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz';v=np.load(p)
  base=v['base']; truth=.5*(v['truth_half1']+v['truth_half2']); pred=v['direct32'];e=base-truth;dd=pred-base
  raw.append({'raw_a0':float(np.mean(e*e)),'raw_lin':float(2*np.mean(e*dd)),'raw_q':float(np.mean(dd*dd)),
              'raw_alpha_opt':float(-np.mean(e*dd)/max(np.mean(dd*dd),1e-30))})
 return pd.concat([d,pd.DataFrame(raw)],axis=1)

def policy_metrics(d,alpha,bootstrap=0,seed=260729):
 a=np.asarray(alpha,float);b=d.raw_a0.to_numpy();m=b+a*d.raw_lin.to_numpy()+a*a*d.raw_q.to_numpy();rat=m/b
 bu=d.baseline_unbiased_mse.to_numpy();qu=[]
 # exact unbiased quadratic values are in label JSON, recover directly below if columns supplied.
 out={'pooled_ratio':float(m.sum()/b.sum()),'mean_ratio':float(rat.mean()),'median_ratio':float(np.median(rat)),
      'worst':float(rat.max()),'p90':float(np.quantile(rat,.9)),'wins':int(np.sum(rat<1)),
      'alpha_mean':float(a.mean()),'alpha_min':float(a.min()),'alpha_max':float(a.max()),'ratios':rat.tolist(),'alphas':a.tolist()}
 if bootstrap:
  rng=np.random.default_rng(seed);n=len(d);ix=rng.integers(0,n,size=(bootstrap,n));vals=m[ix].sum(axis=1)/b[ix].sum(axis=1)
  out['bootstrap_ci95']=[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
  out['bootstrap_prob_below_0_595']=float(np.mean(vals<=.595));out['bootstrap_prob_below_1']=float(np.mean(vals<1))
 return out

def attach_unbiased(d,panel):
 vals=[]
 for _,z in d.iterrows():
  p=panel/'cache'/f'label_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.json';r=json.load(open(p));q=r['k32']['quadratic']
  vals.append({'unb_a0':q['unbiased_a0'],'unb_lin':q['unbiased_linear'],'unb_q':q['correction_norm2'],'unb_alpha_opt':q['optimal_alpha_unconstrained']})
 return pd.concat([d.reset_index(drop=True),pd.DataFrame(vals)],axis=1)

def unb_metrics(d,alpha):
 a=np.asarray(alpha,float);b=d.unb_a0.to_numpy();m=b+a*d.unb_lin.to_numpy()+a*a*d.unb_q.to_numpy();rat=m/b
 return {'pooled_ratio':float(m.sum()/b.sum()),'mean_ratio':float(rat.mean()),'median_ratio':float(np.median(rat)),'worst':float(rat.max()),'p90':float(np.quantile(rat,.9)),'wins':int(np.sum(rat<1))}

def policies(d,canonical=False):
 P={}
 for a in [0,.15,.2,.25,.3,.35,.4,.45,.5,.6,.75,1.0]:
  P[f'fixed_{a:g}']=policy_metrics(d,np.full(len(d),a),50000 if canonical else 0)
  P[f'fixed_{a:g}']['unbiased']=unb_metrics(d,np.full(len(d),a))
 P['oracle_raw_alpha']=policy_metrics(d,d.raw_alpha_opt.to_numpy(),0);P['oracle_raw_alpha']['unbiased']=unb_metrics(d,d.raw_alpha_opt.to_numpy())
 P['oracle_unbiased_alpha']=policy_metrics(d,d.unb_alpha_opt.to_numpy(),0);P['oracle_unbiased_alpha']['unbiased']=unb_metrics(d,d.unb_alpha_opt.to_numpy())
 P['l08_abstain']=policy_metrics(d,np.where(d.frozen_gate_apply_n129,1.,0.),50000 if canonical else 0);P['l08_abstain']['unbiased']=unb_metrics(d,np.where(d.frozen_gate_apply_n129,1.,0.))
 # Frozen after hard-panel discovery: conservative .4 everywhere, and .25 only above original risk threshold.
 for name,a in {
  'fixed04_l08_high_to025':np.where(d.frozen_gate_apply_n129,.4,.25),
  'fixed05_l08_high_to025':np.where(d.frozen_gate_apply_n129,.5,.25),
  'fixed05_l08_high_to04':np.where(d.frozen_gate_apply_n129,.5,.4),
 }.items():
  P[name]=policy_metrics(d,a,50000 if canonical else 0);P[name]['unbiased']=unb_metrics(d,a)
 return P

can=attach_unbiased(load_panel(CAN),CAN); hard=attach_unbiased(load_panel(HARD),HARD)
# Canonical panel should be one rotation per 24 network groups.
assert len(can)==24 and can.network_seed.nunique()==24 and set(can.rotation_seed)=={3}
cp=policies(can,True); hp=policies(hard,False)
out={
 'canonical_n':len(can),'hard_n':len(hard),'groups':int(can.network_seed.nunique()),
 'fast_reference_audit':{'target_vectors_identical':True,'anchor_moment_relative_error_max':7e-8,'max_ratio_shift':2.3e-5},
 'canonical_alpha_summary':{k:float(v) for k,v in can.raw_alpha_opt.describe().items()},
 'canonical_unbiased_alpha_summary':{k:float(v) for k,v in can.unb_alpha_opt.describe().items()},
 'hard_alpha_summary':{k:float(v) for k,v in hard.raw_alpha_opt.describe().items()},
 'canonical_correlations':{
  'risk_vs_raw_alpha_spearman':float(spearmanr(can.risk_n129,can.raw_alpha_opt).statistic),
  'risk_vs_full_ratio_spearman':float(spearmanr(can.risk_n129,can.k32_ratio).statistic),
  'risk_vs_raw_alpha_pearson':float(pearsonr(can.risk_n129,can.raw_alpha_opt).statistic),
 },
 'canonical_policies':cp,'hard_policies':hp,
 'canonical_rows':can[['network_seed','rotation_seed','risk_n129','k32_ratio','raw_alpha_opt','unb_alpha_opt']].to_dict('records')
}
(ROOT/'results'/'CANONICAL_SCALE_RESULTS.json').write_text(json.dumps(out,indent=2))
cols=['pooled_ratio','worst','p90','wins','alpha_mean']
print('CANONICAL')
print(pd.DataFrame([{'policy':k,**{c:v[c] for c in cols}} for k,v in cp.items()]).sort_values(['pooled_ratio']).to_string(index=False))
print('\nHARD')
print(pd.DataFrame([{'policy':k,**{c:v[c] for c in cols}} for k,v in hp.items()]).sort_values(['worst','pooled_ratio']).to_string(index=False))
print('\nalpha canonical',can.raw_alpha_opt.describe().to_dict())
