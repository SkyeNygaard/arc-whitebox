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
if len(sys.argv)!=2:raise SystemExit('usage: run_node.py OFFICIAL_RESULT_DIR')
d=Path(sys.argv[1]); arms=[129,96,64,32];rows=[]
for k in arms:
 candidates=[d/f'A43_basis{k:03d}.json',d/f'basis_{k}.json',d/('A43.json' if k==129 else f'A43_basis{k:03d}.json')]
 p=next((x for x in candidates if x.exists()),None)
 if p is None:raise SystemExit(f'missing result for {k} bases')
 r={'bases':k,'file':str(p),**metrics(p)};rows.append(r)
base=rows[0]
for r in rows:
 r['adjusted_ratio_vs_129']=None if r['score'] is None or base['score'] is None else r['score']/base['score']
 r['raw_ratio_vs_129']=None if r['raw_mse'] is None or base['raw_mse'] is None else r['raw_mse']/base['raw_mse']
with open(d/'T0.1_OFFICIAL_RESULTS.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
