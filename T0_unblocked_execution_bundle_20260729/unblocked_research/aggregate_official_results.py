#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
import numpy as np
ARMS=['production_baseline','A42','A43','A43_delta64','A43_basis096','A43_basis064','A43_basis032']
ALIASES={
 'raw_mse':['raw_final_layer_mse','final_layer_mse','final_mse'],
 'adjusted_score':['adjusted_score','adjusted_final_layer_score','score'],
 'tracked_flops':['tracked_flops','flops_used','instrumented_flops'],
 'residual_wall_s':['residual_wall_time_s','residual_wall_s'],
 'effective_compute':['effective_compute','effective_flops'],
 'wall_time_s':['wall_time_s','wall_elapsed_s'],
 'failures':['failure_count','failures']}
def norm(x):return re.sub('[^a-z0-9]+','_',str(x).lower()).strip('_')
def walk(o,p=''):
 if isinstance(o,dict):
  for k,v in o.items():yield from walk(v,f'{p}.{k}' if p else str(k))
 elif isinstance(o,list):
  for i,v in enumerate(o):yield from walk(v,f'{p}[{i}]')
 else:yield p,o
def extract(o,names):
 for p,v in walk(o):
  if norm(p.split('.')[-1].split('[')[0]) in {norm(x) for x in names} and isinstance(v,(int,float)):
   return float(v)
def json_objects(text):
 # whole-file JSON first, then line JSON.
 try:yield json.loads(text);return
 except Exception:pass
 for line in text.splitlines():
  line=line.strip()
  if line.startswith('{'):
   try:yield json.loads(line)
   except Exception:pass
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=[]
 for arm in ARMS:
  d=a.input/arm;objs=[]
  for p in list(d.rglob('*.json'))+[d/'stdout.log']:
   if p.exists():objs.extend(json_objects(p.read_text(errors='replace')))
  rec={'arm':arm}
  for key,names in ALIASES.items():
   vals=[extract(o,names) for o in objs];vals=[v for v in vals if v is not None]
   if vals:rec[key]=vals[-1]
  rows.append(rec)
 with (a.output/'aggregate.csv').open('w',newline='') as f:
  keys=sorted(set().union(*(r.keys() for r in rows)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 # Deterministic T0 calculations when metrics are present.
 by={r['arm']:r for r in rows};calc={}
 if all('effective_compute' in by[x] for x in ['A43','A43_delta64']):
  calc['t0_2_residual_delta_s']=(by['A43_delta64']['effective_compute']-by['A43']['effective_compute']-2147483648)/1e11
 if all('residual_wall_s' in by[x] for x in ['production_baseline','A43']):
  calc['t0_3_a43_residual_delta_s']=by['A43']['residual_wall_s']-by['production_baseline']['residual_wall_s'];calc['t0_3_a43_compute_pass']=calc['t0_3_a43_residual_delta_s']<.00524123904
 if all('adjusted_score' in by[x] for x in ['A43','A43_basis096','A43_basis064','A43_basis032']):
  calc['t0_1_adjusted_ratios_vs_129']={x:by[x]['adjusted_score']/by['A43']['adjusted_score'] for x in ['A43','A43_basis096','A43_basis064','A43_basis032']}
 (a.output/'T0_CALCULATIONS.json').write_text(json.dumps({'rows':rows,'calculations':calc},indent=2))
 print(json.dumps(calc,indent=2))
if __name__=='__main__':main()
