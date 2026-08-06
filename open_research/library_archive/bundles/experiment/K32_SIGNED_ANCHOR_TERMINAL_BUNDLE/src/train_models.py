#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ProbeSetModel(nn.Module):
    def __init__(self, token_dim: int, global_dim: int, hidden: int = 96):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(token_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.global_enc = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
    def forward(self, token: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        h = self.token(token)
        context = h.mean(dim=1, keepdim=True).expand_as(h)
        g = self.global_enc(glob)[:, None, :].expand_as(h)
        return self.decoder(torch.cat([h, context, g], dim=-1)).squeeze(-1)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load_split(data_dir: Path, split: str) -> dict[str,np.ndarray]:
    rows=[]
    for path in sorted(data_dir.glob(f'{split}_network_*.npz')):
        with np.load(path) as z:
            nr=len(z['rotation_seeds'])
            for j in range(nr):
                rows.append({k:z[k][j].copy() for k in ['global_features','token_features','target_delta','beta_bar','sample_prediction','baseline_prediction','oracle_prediction','truth_half1','truth_half2','base_mse','sample_mse','oracle_mse','base_unbiased_mse','oracle_unbiased_mse','oracle_ratio','target_correction'] } | {'network_id':int(z['network_id']),'rotation_seed':int(z['rotation_seeds'][j]),'source_file':path.name})
    out={}
    for k in rows[0]:
        if k in ('source_file',): out[k]=np.asarray([r[k] for r in rows])
        elif k in ('network_id','rotation_seed'): out[k]=np.asarray([r[k] for r in rows],dtype=np.int64)
        else: out[k]=np.stack([np.asarray(r[k]) for r in rows])
    return out


def pooled_ratio(pred: np.ndarray, data: dict[str,np.ndarray]) -> tuple[float,float,list[float]]:
    truth=.5*(data['truth_half1']+data['truth_half2'])
    mse=np.mean((pred-truth)**2,axis=1)
    base=np.asarray(data['base_mse']).reshape(-1)
    ratios=(mse/np.maximum(base,1e-300)).tolist()
    return float(mse.sum()/base.sum()), float(np.max(ratios)), ratios


def predict_candidate(delta: np.ndarray, data: dict[str,np.ndarray]) -> np.ndarray:
    corr=np.einsum('np,npd->nd',delta,data['beta_bar'])
    return data['sample_prediction']+corr


def bootstrap_base_ratio(pred: np.ndarray, data: dict[str,np.ndarray], reps=5000, seed=20260729):
    truth=.5*(data['truth_half1']+data['truth_half2'])
    mse=np.mean((pred-truth)**2,axis=1)
    base=np.asarray(data['base_mse']).reshape(-1)
    ids=np.unique(data['network_id']); rng=np.random.default_rng(seed); vals=[]
    groups={i:np.flatnonzero(data['network_id']==i) for i in ids}
    for _ in range(reps):
        chosen=rng.choice(ids,size=len(ids),replace=True); num=den=0.0
        for i in chosen:
            ix=groups[int(i)]; num+=float(mse[ix].sum()); den+=float(base[ix].sum())
        vals.append(num/max(den,1e-300))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def standardize(train, val):
    gm=train['global_features'].mean(0); gs=train['global_features'].std(0); gs=np.maximum(gs,1e-6)
    tm=train['token_features'].reshape(-1,train['token_features'].shape[-1]).mean(0)
    ts=train['token_features'].reshape(-1,train['token_features'].shape[-1]).std(0); ts=np.maximum(ts,1e-6)
    ys=float(train['target_delta'].std()); ys=max(ys,1e-8)
    def tx(d):
        return ((d['global_features']-gm)/gs).astype(np.float32), ((d['token_features']-tm)/ts).astype(np.float32), (d['target_delta']/ys).astype(np.float32)
    return tx(train),tx(val),{'global_mean':gm,'global_std':gs,'token_mean':tm,'token_std':ts,'target_scale':ys}


def train_one(seed:int, train, val, norm, cfg, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    (gtr,ttr,ytr),(gva,tva,yva)=train,val
    model=ProbeSetModel(ttr.shape[-1],gtr.shape[-1],cfg['token_hidden']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    n=len(gtr); bs=cfg['batch_size']; best=None; best_state=None; patience=0; history=[]
    tensors={
      'gtr':torch.from_numpy(gtr).to(device),'ttr':torch.from_numpy(ttr).to(device),'ytr':torch.from_numpy(ytr).to(device),
      'btr':torch.from_numpy(train_raw['beta_bar']).to(device),'ctr':torch.from_numpy(train_raw['target_correction']).to(device),
      'gva':torch.from_numpy(gva).to(device),'tva':torch.from_numpy(tva).to(device),'yva':torch.from_numpy(yva).to(device),
    }
    beta_w=np.sum(train_raw['beta_bar']**2,axis=2); beta_w=beta_w/np.maximum(beta_w.mean(),1e-12); bw=torch.from_numpy(beta_w.astype(np.float32)).to(device)
    scale=float(norm['target_scale'])
    for epoch in range(cfg['max_epochs']):
        model.train(); order=np.random.permutation(n); total=0.0
        for st in range(0,n,bs):
            ix=torch.as_tensor(order[st:st+bs],device=device)
            pred=model(tensors['ttr'][ix],tensors['gtr'][ix])
            anchor=((pred-tensors['ytr'][ix])**2*bw[ix]).mean()
            pred_delta=pred*scale
            pcorr=torch.einsum('bp,bpd->bd',pred_delta,tensors['btr'][ix])
            tcorr=tensors['ctr'][ix]
            final=((pcorr-tcorr)**2).mean()/(tcorr.square().mean().detach()+1e-12)
            loss=final+0.15*anchor
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); total+=float(loss)*len(ix)
        model.eval()
        with torch.no_grad():
            pv=model(tensors['tva'],tensors['gva']).cpu().numpy()*scale
        cand=predict_candidate(pv,val_raw); vr,_,_=pooled_ratio(cand,val_raw)
        history.append({'epoch':epoch,'train_loss':total/n,'validation_candidate_ratio':vr})
        if best is None or vr<best-1e-5:
            best=vr; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=cfg['patience']: break
    model.load_state_dict(best_state)
    return model,best,history


def ensemble_predict(models, data_std, norm, data_raw, device):
    g,t,_=data_std; gt=torch.from_numpy(g).to(device); tt=torch.from_numpy(t).to(device); preds=[]; corrs=[]
    with torch.no_grad():
        for m in models:
            m.eval(); d=m(tt,gt).cpu().numpy()*float(norm['target_scale']); preds.append(d); corrs.append(np.einsum('np,npd->nd',d,data_raw['beta_bar']))
    ds=np.stack(preds); cs=np.stack(corrs); dm=ds.mean(0); cm=cs.mean(0)
    disp=np.sqrt(np.mean(np.sum((cs-cm[None])**2,axis=2),axis=0)); mag=np.linalg.norm(cm,axis=1)
    consistency=mag/(mag+disp+1e-12)
    # Fixed pre-registered agreement gate: every ensemble correction must have cosine >0.5 with its mean.
    den=np.maximum(np.linalg.norm(cs,axis=2)*mag[None],1e-12); cos=np.sum(cs*cm[None],axis=2)/den
    agree=(np.min(cos,axis=0)>0.5).astype(np.float64)
    scale=consistency*agree
    gated_delta=dm*scale[:,None]
    return {'member_delta':ds,'mean_delta':dm,'member_corrections':cs,'mean_correction':cm,'consistency':consistency,'min_member_cosine':np.min(cos,axis=0),'agreement':agree,'scale':scale,'gated_delta':gated_delta}


def ridge_baseline(train_raw,val_raw,test_raw,norm):
    from sklearn.linear_model import Ridge
    tm,ts=norm['token_mean'],norm['token_std']; gm,gs=norm['global_mean'],norm['global_std']; ys=norm['target_scale']
    def flat(d):
        t=(d['token_features']-tm)/ts; g=(d['global_features']-gm)/gs; gr=np.repeat(g[:,None,:],t.shape[1],axis=1); return np.concatenate([t,gr],axis=2).reshape(-1,t.shape[2]+g.shape[1])
    X=flat(train_raw); y=(train_raw['target_delta']/ys).reshape(-1); w=np.sum(train_raw['beta_bar']**2,axis=2).reshape(-1); w=w/np.maximum(w.mean(),1e-12)
    model=Ridge(alpha=10.0,fit_intercept=True).fit(X,y,sample_weight=w)
    def pred(d): return (model.predict(flat(d)).reshape(len(d['network_id']),-1)*ys).astype(np.float32)
    return model,pred(val_raw),pred(test_raw)


def write_models(models, outdir, norm, cfg, histories):
    outdir.mkdir(parents=True,exist_ok=True); hashes=[]
    for i,m in enumerate(models):
        p=outdir/f'model_{i}.pt'; torch.save({'state_dict':m.state_dict(),'token_dim':m.token[0].in_features,'global_dim':m.global_enc[0].in_features,'hidden':cfg['token_hidden']},p); hashes.append(sha256(p))
    np.savez_compressed(outdir/'normalization.npz',**norm)
    (outdir/'training_histories.json').write_text(json.dumps(histories,indent=2))
    return hashes


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); args=ap.parse_args()
    global train_raw,val_raw
    cfg_all=json.loads((args.experiment/'frozen_config.json').read_text()); cfg=cfg_all['model']
    train_raw=load_split(args.experiment/'data','train'); val_raw=load_split(args.experiment/'data','validation')
    train_std,val_std,norm=standardize(train_raw,val_raw)
    device=torch.device('cpu'); torch.set_num_threads(min(5,os.cpu_count() or 1)); torch.set_num_interop_threads(1)
    models=[]; bests=[]; histories=[]; t0=time.time()
    for seed in cfg['ensemble_seeds']:
        m,b,h=train_one(seed,train_std,val_std,norm,cfg,device); models.append(m); bests.append(b); histories.append({'seed':seed,'best_validation_ratio':b,'history':h}); print('seed',seed,'best',b,flush=True)
    hashes=write_models(models,args.experiment/'models',norm,cfg,histories)
    ens=ensemble_predict(models,val_std,norm,val_raw,device)
    pred=predict_candidate(ens['gated_delta'],val_raw); ratio,worst,ratios=pooled_ratio(pred,val_raw)
    ung=predict_candidate(ens['mean_delta'],val_raw); ur,uw,_=pooled_ratio(ung,val_raw)
    oracle_ratio=float(np.asarray(val_raw['oracle_mse']).sum()/np.asarray(val_raw['base_mse']).sum())
    sample_ratio=float(np.asarray(val_raw['sample_mse']).sum()/np.asarray(val_raw['base_mse']).sum())
    result={
      'freeze_sha256':cfg_all['freeze_sha256'],'model_hashes':hashes,'training_seconds':time.time()-t0,
      'train_examples':len(train_raw['network_id']),'validation_examples':len(val_raw['network_id']),
      'validation':{'oracle_ratio':oracle_ratio,'sample_anchor_ratio':sample_ratio,'ensemble_ungated_ratio':ur,'ensemble_ungated_worst':uw,'ensemble_gated_ratio':ratio,'ensemble_gated_worst':worst,'bootstrap_95':bootstrap_base_ratio(pred,val_raw),'wins':sum(x<1 for x in ratios),'mean_scale':float(ens['scale'].mean()),'abstentions':int(np.sum(ens['agreement']==0)),'min_member_cosine':float(np.min(ens['min_member_cosine']))},
      'test_not_opened':True,
    }
    (args.experiment/'results'/'training_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
