#!/usr/bin/env python3
import json,glob
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/mnt/data/work/new_opportunities/selected')
rows=[json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(P/'validation_network_*.json')))]
rng=np.random.default_rng(20260730)
def eval_method(label,mode,alpha):
 b=np.array([r['base_unbiased_mse'] for r in rows]); inn=np.array([r['candidates'][label][mode]['inner'] for r in rows]);nn=np.array([r['candidates'][label][mode]['norm_sq'] for r in rows]);m=b+2*alpha*inn+alpha*alpha*nn
 ratios=m/b
 ix=rng.integers(0,len(rows),(100000,len(rows))); agg=b[ix].sum(1)/m[ix].sum(1)
 return {'label':label,'mode':mode,'alpha':alpha,'candidate_over_base':float(m.sum()/b.sum()),'gain_base_over_candidate':float(b.sum()/m.sum()),'bootstrap_gain_ci95':[float(np.quantile(agg,.025)),float(np.quantile(agg,.975))],
  'wins':int((m<b).sum()),'networks':len(rows),'median_ratio':float(np.median(ratios)),'worst_ratio':float(np.max(ratios)),'ratios':ratios.tolist()}
methods=[eval_method('stein_tanh_k8_stable','full',2.0),eval_method('stein_tanh_k8_stable','full',1.0),eval_method('harmonic_d68_k8','cf4',1.0),eval_method('harmonic_d68_k8','full',1.0)]
primary=methods[0];gate={'candidate_over_base':primary['candidate_over_base']<=.98,'ci_upper_gain':primary['bootstrap_gain_ci95'][0]>1.0,'wins':primary['wins']>=9,'worst':primary['worst_ratio']<=1.10};passed=all(gate.values())
out={'preregistration_sha256':'b3b842a2cc94368f72a7c260365ed143605f98d625b0176c0ffaa3701a4583c8','network_ids':[r['network_id'] for r in rows],'methods':methods,'primary_gate_components':gate,'primary_passed_all_gates':passed,'holdout_opened':False}
(P/'VALIDATION_PREREGISTERED_RESULTS.json').write_text(json.dumps(out,indent=2))
pd.DataFrame([{k:v for k,v in m.items() if k!='ratios'} for m in methods]).to_csv(P/'VALIDATION_PREREGISTERED_RESULTS.csv',index=False)
print(json.dumps(out,indent=2))
