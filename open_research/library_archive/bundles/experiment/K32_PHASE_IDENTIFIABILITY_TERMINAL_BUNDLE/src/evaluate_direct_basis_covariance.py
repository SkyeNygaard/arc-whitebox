#!/usr/bin/env python3
from pathlib import Path
import glob,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BP=ROOT/'basis_phase'
# fixed template
vals=[]
for p in sorted(glob.glob(str(ROOT/'data/train_network_*.npz'))):
 with np.load(p) as z:
  vals.extend([z['target_delta'][j].astype(float) for j in range(len(z['rotation_seeds']))])
t=np.mean(vals,0);t/=np.linalg.norm(t)

def load(split):
 rows=[]
 for p in sorted(glob.glob(str(ROOT/f'data/{split}_network_*.npz'))):
  with np.load(p) as z:
   nid=int(z['network_id'])
   for j,rot in enumerate(z['rotation_seeds']):
    with np.load(BP/f'data/{nid}_{j}.npz') as b:dx=b['dx'].astype(float);yb=b['yb'].astype(float)
    yc=yb-yb.mean(0);C=dx.T@yc/len(dx);v=t@C
    rows.append({'id':nid,'rot':int(rot),'v':v,'sample':z['sample_prediction'][j].astype(float),'truth':.5*(z['truth_half1'][j]+z['truth_half2'][j]),'base':float(z['base_mse'][j])})
 return rows
tr=load('train');va=load('validation')
# grouped 8-fold OOF scalar calibration for the direct vector
ids=sorted(set(r['id'] for r in tr));folds=np.array_split(np.arange(len(ids)),8);oof=np.zeros(len(tr))
for hold in folds:
 keepids={ids[k] for k in range(len(ids)) if k not in set(hold)};testids={ids[k] for k in hold}
 num=den=0.
 for r in tr:
  if r['id'] in keepids:
   e=r['truth']-r['sample'];num+=r['v']@e;den+=r['v']@r['v']
 g=num/max(den,1e-30)
 for i,r in enumerate(tr):
  if r['id'] in testids:oof[i]=g
# shrink OOF replay once
num=den=0.
for r,g in zip(tr,oof):
 d=g*r['v'];e=r['truth']-r['sample'];num+=d@e;den+=d@d
sh=float(np.clip(num/max(den,1e-30),0,1))
# final scalar on all training
num=den=0.
for r in tr:
 e=r['truth']-r['sample'];num+=r['v']@e;den+=r['v']@r['v']
gamma=num/max(den,1e-30);g=gamma*sh

def metrics(rows):
 mse=[];base=[];rr=[];cos=[]
 for r in rows:
  pred=r['sample']+g*r['v'];m=np.mean((pred-r['truth'])**2);mse.append(m);base.append(r['base']);rr.append(m/r['base']);e=r['truth']-r['sample'];cos.append((r['v']@e)/max(np.linalg.norm(r['v'])*np.linalg.norm(e),1e-30))
 mse=np.array(mse);base=np.array(base);rr=np.array(rr);rng=np.random.default_rng(20260729);uids=sorted(set(r['id'] for r in rows));groups=[[i for i,r in enumerate(rows) if r['id']==u] for u in uids];bs=[]
 for _ in range(20000):
  ds=rng.integers(0,len(groups),len(groups));ix=np.concatenate([groups[k] for k in ds]);bs.append(mse[ix].sum()/base[ix].sum())
 return {'ratio':float(mse.sum()/base.sum()),'bootstrap95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max()),'mean_direction_cosine':float(np.mean(cos))}
out={'formula':'v = template^T E_b[(X_b-q)^T(Y_b-Ybar)]; candidate = sample + gamma v','gamma_train':float(gamma),'oof_shrink':sh,'effective_gamma':float(g),'validation':metrics(va),'incremental_ops':129*32*256*2}
(BP/'DIRECT_BASIS_COVARIANCE_DEVELOPMENT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
