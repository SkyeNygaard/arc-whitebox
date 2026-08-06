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
if len(sys.argv)!=4:raise SystemExit('usage: run_node.py PRODUCTION.json A42.json A43.json')
names=['production','A42','A43'];rows={n:metrics(p) for n,p in zip(names,sys.argv[1:])};base=rows['production']
for n,r in rows.items():
 r['score_ratio_vs_production']=None if r['score'] is None or base['score'] is None else r['score']/base['score']
 r['raw_ratio_vs_production']=None if r['raw_mse'] is None or base['raw_mse'] is None else r['raw_mse']/base['raw_mse']
 r['effective_ratio_vs_production']=None if r['effective'] is None or base['effective'] is None else r['effective']/base['effective']
print(json.dumps(rows,indent=2))
