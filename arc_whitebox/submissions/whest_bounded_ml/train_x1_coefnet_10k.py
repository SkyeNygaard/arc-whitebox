#!/usr/bin/env python3
"""Streaming trainer for the compact symmetry-constrained x1/x1a closure.

Unlike the HistGradientBoosting baseline, this model is small enough to plausibly
run inside the WhestBench compute envelope. It predicts

    residual_ij = c_s(layer,a_i,a_j,rho) * x1_ij
                + (a_i-a_j) * c_a(layer,a_i,a_j,rho) * x1a_ij

where c_s and c_a are produced by a small MLP from exchange-invariant inputs.
The construction guarantees residual_ij == residual_ji.

Data files expected in --data-dir:
  a_train.npy, rho_train.npy, rn_train.npy, x1_train.npy, x1a_train.npy
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

WIDTH=256; DEPTH=32; PAIRS=WIDTH*(WIDTH-1)//2
IU,JU=np.triu_indices(WIDTH,1)

class CoefNet(nn.Module):
    def __init__(self, hidden:int=32):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,2))
    def forward(self,b,d,x1,x1a):
        c=self.body(b)
        return c[:,0]*x1 + d*c[:,1]*x1a

def require(root:Path):
    names=['a_train.npy','rho_train.npy','rn_train.npy','x1_train.npy','x1a_train.npy']
    miss=[str(root/n) for n in names if not (root/n).exists()]
    if miss: raise FileNotFoundError('missing:\n  '+'\n  '.join(miss))
    return {n[:-10]:np.load(root/n,mmap_mode='r') for n in names}

def block_features(a,rho,x1,x1a,idx,layer):
    ai=np.asarray(a[IU[idx]],dtype=np.float32);aj=np.asarray(a[JU[idx]],dtype=np.float32);d=ai-aj
    b=np.column_stack([np.full(len(idx),(layer+1)/DEPTH,np.float32),ai+aj,ai*aj,np.abs(d),np.asarray(rho[idx],np.float32)]).astype(np.float32)
    return b,d.astype(np.float32),np.asarray(x1[idx],np.float32),np.asarray(x1a[idx],np.float32)

def sample_batch(A,nets,layers,blocks,pairs_per_block,rng,mean=None,std=None):
    bs=[];ds=[];xs=[];xas=[];ys=[]
    for _ in range(blocks):
        n=int(rng.choice(nets));l=int(rng.choice(layers));idx=rng.choice(PAIRS,pairs_per_block,replace=False)
        b,d,x,xa=block_features(A['a'][n,l],A['rho'][n,l],A['x1'][n,l],A['x1a'][n,l],idx,l)
        bs.append(b);ds.append(d);xs.append(x);xas.append(xa);ys.append(np.asarray(A['rn'][n,l,idx],np.float32))
    b=np.concatenate(bs)
    if mean is not None:b=(b-mean)/std
    return tuple(torch.from_numpy(q) for q in [b,np.concatenate(ds),np.concatenate(xs),np.concatenate(xas),np.concatenate(ys)])

def fixed_eval_rows(A,nets,layers,rows_per_case,seed,mean,std):
    rng=np.random.default_rng(seed);parts=[[] for _ in range(5)];groups=[]
    for gi,n in enumerate(nets):
        for l in layers:
            idx=rng.choice(PAIRS,rows_per_case,replace=False)
            b,d,x,xa=block_features(A['a'][n,l],A['rho'][n,l],A['x1'][n,l],A['x1a'][n,l],idx,int(l))
            for dst,src in zip(parts,[b,d,x,xa,np.asarray(A['rn'][n,l,idx],np.float32)]):dst.append(src)
            groups.append((gi,rows_per_case))
    b=(np.concatenate(parts[0])-mean)/std
    tensors=tuple(torch.from_numpy(q) for q in [b,np.concatenate(parts[1]),np.concatenate(parts[2]),np.concatenate(parts[3]),np.concatenate(parts[4])])
    return tensors,groups

def metrics(model,tensors,groups,device):
    model.eval();pred=[];B=131072
    with torch.no_grad():
        for s in range(0,len(tensors[-1]),B):pred.append(model(*(q[s:s+B].to(device) for q in tensors[:4])).cpu())
    p=torch.cat(pred).numpy().astype(np.float64);y=tensors[-1].numpy().astype(np.float64)
    bm=float(np.mean(y*y));mm=float(np.mean((p-y)**2));pos=0;wins=[];per=[]
    for gi,count in groups:
        yy=y[pos:pos+count];pp=p[pos:pos+count];b=float(np.mean(yy*yy));m=float(np.mean((pp-yy)**2));wins.append(m<b);per.append({'group':int(gi),'base_mse':b,'model_mse':m,'gain':b/max(m,1e-30)});pos+=count
    return {'base_mse':bm,'model_mse':mm,'gain':bm/max(mm,1e-30),'r2':1-mm/max(bm,1e-30),'fraction_groups_improved':float(np.mean(wins)),'per_group':per}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--hidden',type=int,default=32);ap.add_argument('--steps',type=int,default=12000);ap.add_argument('--blocks-per-step',type=int,default=16);ap.add_argument('--pairs-per-block',type=int,default=2048)
    ap.add_argument('--train-networks',type=int,default=8000);ap.add_argument('--valid-networks',type=int,default=1000);ap.add_argument('--test-networks',type=int,default=1000)
    ap.add_argument('--layers',default='all');ap.add_argument('--lr',type=float,default=2e-3);ap.add_argument('--seed',type=int,default=20260810);ap.add_argument('--eval-every',type=int,default=250);ap.add_argument('--eval-rows-per-case',type=int,default=256);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True);A=require(args.data_dir);n=A['a'].shape[0]
    layers=np.arange(31,dtype=np.int64) if args.layers.lower()=='all' else np.array([int(x) for x in args.layers.split(',')],dtype=np.int64)
    rng=np.random.default_rng(args.seed);perm=rng.permutation(n);nt=args.train_networks;nv=args.valid_networks;ne=args.test_networks
    if nt+nv+ne>n:raise ValueError('split larger than dataset')
    tr,va,te=perm[:nt],perm[nt:nt+nv],perm[nt+nv:nt+nv+ne]
    # Robust standardization pilot.
    pilot=sample_batch(A,tr,layers,64,512,rng);base=pilot[0].numpy();mean=base.mean(0).astype(np.float32);std=(base.std(0)+1e-6).astype(np.float32);del pilot,base
    va_t,va_g=fixed_eval_rows(A,va[:min(100,len(va))],layers[::3],args.eval_rows_per_case,args.seed+1,mean,std)
    te_t,te_g=fixed_eval_rows(A,te[:min(200,len(te))],layers[::2],args.eval_rows_per_case,args.seed+2,mean,std)
    device=torch.device(args.device);torch.manual_seed(args.seed);model=CoefNet(args.hidden).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=3e-5);best=None;best_gain=-1;best_step=0;history=[];t0=time.time()
    for step in range(1,args.steps+1):
        model.train();q=sample_batch(A,tr,layers,args.blocks_per_step,args.pairs_per_block,rng,mean,std);q=tuple(x.to(device) for x in q);pred=model(*q[:4]);loss=((pred-q[4])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%args.eval_every==0 or step==1:
            v=metrics(model,va_t,va_g,device);history.append({'step':step,'train_loss':float(loss),'valid':{k:v[k] for k in ['base_mse','model_mse','gain','r2','fraction_groups_improved']}});print(json.dumps(history[-1]),flush=True)
            if v['gain']>best_gain:best_gain=v['gain'];best_step=step;best={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};torch.save({'state_dict':best,'mean':mean,'std':std,'hidden':args.hidden,'step':step},args.out_dir/'x1_coefnet.pt')
    model.load_state_dict(best);test=metrics(model,te_t,te_g,device)
    result={'config':{**vars(args),'data_dir':str(args.data_dir),'out_dir':str(args.out_dir)},'feature_names':['layer','a_sum','a_product','abs_a_difference','rho','x1','a_difference_times_x1a'],'parameters':sum(p.numel() for p in model.parameters()),'best_step':best_step,'best_valid_gain':best_gain,'seconds':time.time()-t0,'history':history,'test':test,'split':{'train':tr.tolist(),'valid':va.tolist(),'test':te.tolist()}}
    (args.out_dir/'x1_coefnet_results.json').write_text(json.dumps(result,indent=2,default=str));print(json.dumps({'test_gain':test['gain'],'test_r2':test['r2'],'fraction':test['fraction_groups_improved'],'params':result['parameters']}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:print(f'ERROR {type(e).__name__}: {e}',file=sys.stderr);raise
