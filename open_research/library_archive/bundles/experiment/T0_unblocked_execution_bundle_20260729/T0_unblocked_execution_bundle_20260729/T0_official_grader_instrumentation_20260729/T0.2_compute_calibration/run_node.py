from pathlib import Path
import json,sys,csv
ALIASES={
 'raw_mse':['raw_final_layer_mse','final_layer_mse','raw_mse','mean_raw_mse'],
 'score':['adjusted_final_layer_score','adjusted_score','score','mean_adjusted_score'],
 'tracked':['flops_used','tracked_flops_per_network','tracked_flops'],
 'effective':['effective_compute','mean_effective_compute'],
 'residual':['residual_wall_time_s','mean_residual_wall_time_s'],
}
def find_key(x,names):
 if isinstance(x,dict):
  for n in names:
   if n in x and isinstance(x[n],(int,float)):return float(x[n])
  for v in x.values():
   z=find_key(v,names)
   if z is not None:return z
 if isinstance(x,list):
  for v in x:
   z=find_key(v,names)
   if z is not None:return z
 return None
def metrics(p):
 d=json.load(open(p));return {k:find_key(d,v) for k,v in ALIASES.items()}
if len(sys.argv)!=3:raise SystemExit('usage: run_node.py CLEAN.json DELTA.json')
c=metrics(sys.argv[1]);d=metrics(sys.argv[2]);expected=2147483648.0
out={'clean':c,'delta':d,'expected_tracked_delta':expected}
if c['tracked'] is not None and d['tracked'] is not None:out['observed_tracked_delta']=d['tracked']-c['tracked']
if c['effective'] is not None and d['effective'] is not None:
 out['effective_delta']=d['effective']-c['effective'];out['implied_residual_delta_s']=(out['effective_delta']-out.get('observed_tracked_delta',expected))/1e11
if c['score'] is not None and d['score'] is not None and out.get('effective_delta'):
 out['score_sensitivity_per_effective_flop']=(d['score']-c['score'])/out['effective_delta']
if c['raw_mse'] is not None and d['raw_mse'] is not None:out['raw_mse_difference']=d['raw_mse']-c['raw_mse']
print(json.dumps(out,indent=2))
