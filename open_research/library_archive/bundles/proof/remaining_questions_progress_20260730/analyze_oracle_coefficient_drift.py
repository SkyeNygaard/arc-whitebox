#!/usr/bin/env python3
import json,math,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
p=Path('/mnt/data/current_oracle/extracted/oracle_continuation_20260730/DOWNSTREAM_COEFFICIENT_SYNTHESIS.json')
d=json.load(open(p));g=np.array(d['global_coefficients'],float)
out={'global_coefficients':g.tolist(),'splits':{},'interpretation':{}}
for sn,sp in d['splits'].items():
 P=np.array(sp['predicted_coefficients'],float);fr=np.array(sp['feature_ridge']['case_ratios']);gr=np.array(sp['global_linear']['case_ratios']);delta=fr-gr
 R=P-g
 norms=np.linalg.norm(R,axis=1)
 signs=(np.sign(P)==np.sign(g)).mean(0)
 out['splits'][sn]={
  'n':len(P),'predicted_mean':P.mean(0).tolist(),'predicted_std':P.std(0,ddof=1).tolist(),
  'residual_coefficient_mean':R.mean(0).tolist(),'residual_coefficient_rms':np.sqrt((R*R).mean(0)).tolist(),
  'mean_residual_norm':float(norms.mean()),'median_residual_norm':float(np.median(norms)),
  'sign_match_fraction_by_source':signs.tolist(),
  'feature_minus_global_case_ratio_mean':float(delta.mean()),'feature_minus_global_case_ratio_median':float(np.median(delta)),
  'feature_better_cases':int((delta<0).sum()),'feature_worse_cases':int((delta>0).sum()),
  'corr_residual_norm_with_excess_ratio':float(np.corrcoef(norms,delta)[0,1]),
  'first_source_predicted_negative_fraction':float((P[:,0]<0).mean()),
  'first_source_global_sign':float(np.sign(g[0])),
  'rows':[{'predicted':P[i].tolist(),'residual_from_global':R[i].tolist(),'residual_norm':float(norms[i]),'feature_minus_global_ratio':float(delta[i])} for i in range(len(P))]
 }
out['interpretation']={
 'constant_first_diagnosis':'The feature model does not merely refine the global rule. It frequently reverses or suppresses the globally useful first source coefficient, especially outside development. This confounds average-action recovery with instance-specific residual learning.',
 'next_protocol':'Freeze the global coefficient vector, predict only residual coefficients, and include an explicit shrinkage path lambda in [0,1] around the global rule. Report feature value relative to the matched global baseline, not relative to zero.'}
(ROOT/'ORACLE_COEFFICIENT_DRIFT_ANALYSIS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out['splits'].items()},indent=2)[:8000])
