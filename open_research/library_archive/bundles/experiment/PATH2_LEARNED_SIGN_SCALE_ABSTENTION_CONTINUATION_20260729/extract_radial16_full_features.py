from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import extract_legal_features as e

def labels():
 r=json.load(open('/mnt/data/whest_path2/additional/RADIAL31_CONTINUATION_BUNDLE/fixed04_merged16.json')); rows=[]
 for z in r['records']:
  m0=float(z['mse']['baseline']); m1=float(z['mse']['direct_exact']); ma=float(z['mse']['direct_exact_shrunk']); a=.4
  y1=m1-m0; ya=ma-m0; c=(a*y1-ya)/(a-a*a); b=y1-c; opt=-b/(2*c) if c>0 else np.nan
  rows.append({'network_seed':int(z['seed']),'rotation_seed':3,'baseline_mse':m0,'ratio_alpha1':m1/m0,'ratio_alpha04':ma/m0,
               'alpha_opt':opt,'ratio_opt':(m0+b*opt+c*opt*opt)/m0,'harm_alpha1':int(m1>m0),'harm_alpha04':int(ma>m0),
               'oracle_ratio':float(z['ratio']['full_oracle'])})
 return pd.DataFrame(rows)

def main():
 torch.set_num_threads(5); torch.set_num_interop_threads(1); d=labels(); xk=e.make_kerdock(3); cache=ROOT/'cache_radial16_full'; cache.mkdir(exist_ok=True); rows=[]
 for i,z in d.iterrows():
  p=cache/f'features_{int(z.network_seed)}.json'
  if p.exists(): f=json.load(open(p))
  else: f=e.extract(int(z.network_seed),3,xk); p.write_text(json.dumps(f,sort_keys=True))
  rows.append({**z.to_dict(),**f}); print(json.dumps({'done':i+1,'n':len(d),'seed':int(z.network_seed),'runtime':f['feature_runtime_seconds']}),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'radial16_full_features.csv',index=False)
if __name__=='__main__':main()
