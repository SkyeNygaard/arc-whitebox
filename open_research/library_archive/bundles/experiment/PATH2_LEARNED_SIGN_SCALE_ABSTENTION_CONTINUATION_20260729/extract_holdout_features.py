from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from extract_legal_features import extract,make_kerdock
ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,default=ROOT/'fresh_scale_holdout_v1');ap.add_argument('--threads',type=int,default=5);a=ap.parse_args()
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
proto=json.load(open(a.outdir/'IMMUTABLE_HOLDOUT_PROTOCOL.json'));seeds=proto['network_seeds'];rots=proto['rotation_seeds'];cache=a.outdir/'features';cache.mkdir(exist_ok=True)
X={r:make_kerdock(r) for r in rots};rows=[]
for i,s in enumerate(seeds):
 for r in rots:
  p=cache/f'features_{s}_r{r}.json'
  if p.exists():z=json.load(open(p))
  else:z=extract(s,r,X[r]);p.write_text(json.dumps(z,sort_keys=True))
  rows.append(z);print({'done':len(rows),'total':len(seeds)*len(rots),'seed':s,'rot':r,'runtime':z['feature_runtime_seconds']},flush=True)
pd.DataFrame(rows).to_csv(a.outdir/'legal_features.csv',index=False)
print({'rows':len(rows),'features':len(rows[0])},flush=True)
