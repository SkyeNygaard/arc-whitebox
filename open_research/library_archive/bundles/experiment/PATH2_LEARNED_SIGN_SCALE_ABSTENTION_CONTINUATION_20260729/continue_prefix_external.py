from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
REPRO=Path('/mnt/data/whest_path2/additional/agent3_agent4_repro/agent3_agent4_repro')
sys.path.insert(0,str(REPRO))
from sampling import full_real_kerdock_bases, haar_rotation, chi_mean
from agent34_screen_fast import make_weights

D=256; ROWS=512; N_BASES=129
COUNTS=[8,16,24,32,48,64,80,96,112,129]
ROTATIONS=[3,11,97]

def make_kerdock(rotation:int)->torch.Tensor:
    q=haar_rotation(D,rotation); r=chi_mean(D)
    return torch.stack([torch.cat([b@q,-(b@q)],0)*r for b in full_real_kerdock_bases(D)],0).reshape(-1,D).contiguous()

def risk_from_blocks(bm:np.ndarray)->dict[str,float]:
    out={}
    for n in COUNTS:
        z=bm[:n]
        f=np.stack([z[idx].mean(0) for idx in np.array_split(np.arange(n),min(6,n))])
        c=f.mean(0)
        rel=np.linalg.norm(f-c,axis=1)/(np.linalg.norm(c)+1e-12)
        out[f'n{n:03d}_risk_mean']=float(rel.mean())
        out[f'n{n:03d}_risk_max']=float(rel.max())
        out[f'n{n:03d}_center_norm']=float(np.linalg.norm(c))
    return out

def extract(seed:int,rotation:int,xk:torch.Tensor)->dict:
    t=time.time(); ws=make_weights(seed); x=xk
    with torch.no_grad():
        for li,w in enumerate(ws):
            x=torch.relu(x@w)
            if li==8: break
    a=x.numpy().astype(np.float32,copy=False)
    bm=a.reshape(N_BASES,ROWS,D).mean(1,dtype=np.float64)
    return {'network_seed':seed,'rotation_seed':rotation,'runtime_seconds':time.time()-t,**risk_from_blocks(bm)}

def canonical_examples():
    d=pd.read_csv(ROOT/'legal_features_and_labels.csv')
    return d[['network_seed','rotation_seed','domain','in_hard_panel','baseline_mse','oracle_ratio','candidate_ratio']].copy()

def radial_examples():
    p=Path('/mnt/data/whest_path2/additional/RADIAL31_CONTINUATION_BUNDLE/fixed04_merged16.json')
    r=json.load(open(p)); rows=[]
    for z in r['records']:
        rows.append({'network_seed':int(z['seed']),'rotation_seed':3,'domain':'radial16',
                     'baseline_mse':float(z['mse']['baseline']),
                     'oracle_ratio':float(z['ratio']['full_oracle']),
                     'candidate_ratio':float(z['ratio']['direct_exact_shrunk'])})
    return pd.DataFrame(rows)

def sparse_examples():
    base=Path('/mnt/data/whest_path2/additional/SPARSE_RADIAL_VALIDATION_CORE/records'); rows=[]
    for p in sorted(base.glob('network_*.json')):
        z=json.load(open(p)); m=z['independent_65k_metrics']
        rows.append({'network_seed':int(z['weight_seed']),'rotation_seed':3,'domain':'sparse24',
                     'baseline_mse':float(m['baseline']['pooled_mse']),
                     'oracle_ratio':float(m['complete_exact']['pooled_ratio']),
                     'candidate_ratio':float(m['complete_exact']['pooled_ratio']),
                     'connected_ratio':float(m['connected_only']['pooled_ratio']),
                     'lower_ratio':float(m['lower_only']['pooled_ratio'])})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--threads',type=int,default=5); ap.add_argument('--cohorts',nargs='*',default=['panel','radial','sparse']); args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    frames=[]
    if 'panel' in args.cohorts: frames.append(canonical_examples())
    if 'radial' in args.cohorts: frames.append(radial_examples())
    if 'sparse' in args.cohorts: frames.append(sparse_examples())
    d=pd.concat(frames,ignore_index=True,sort=False)
    cache=ROOT/'cache_prefix'; cache.mkdir(exist_ok=True)
    rots={int(r):make_kerdock(int(r)) for r in sorted(d.rotation_seed.unique())}
    rows=[]
    for i,z in d.iterrows():
        seed=int(z.network_seed); rot=int(z.rotation_seed); p=cache/f'prefix_{seed}_r{rot}.json'
        if p.exists(): f=json.load(open(p))
        else:
            f=extract(seed,rot,rots[rot]); p.write_text(json.dumps(f,sort_keys=True))
        rows.append({**z.to_dict(),**f})
        print(json.dumps({'done':i+1,'n':len(d),'seed':seed,'rot':rot,'domain':z.domain,'runtime':round(f['runtime_seconds'],3)}),flush=True)
    out=pd.DataFrame(rows); out.to_csv(ROOT/'prefix_external_features.csv',index=False)
    print(json.dumps({'rows':len(out),'domains':out.domain.value_counts().to_dict()},indent=2))
if __name__=='__main__': main()
