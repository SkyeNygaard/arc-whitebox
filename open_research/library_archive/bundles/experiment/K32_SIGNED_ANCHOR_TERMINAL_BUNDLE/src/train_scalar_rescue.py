#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import train_models as tm

class ScalarMLP(nn.Module):
    def __init__(self,d:int):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def features(data,template):
    tok=data['token_features']; glob=data['global_features']; beta=data['beta_bar']; d=np.einsum('p,npd->nd',template,beta)
    summaries=np.concatenate([tok.mean(1),tok.std(1),tok.min(1),tok.max(1)],axis=1)
    bn=np.linalg.norm(beta,axis=2)
    extra=np.column_stack([
      np.linalg.norm(d,axis=1),d.mean(1),d.std(1),np.max(np.abs(d),axis=1),np.mean(d>0,axis=1),
      bn.mean(1),bn.std(1),bn.min(1),bn.max(1),
      data['sample_prediction'].mean(1),data['sample_prediction'].std(1),np.linalg.norm(data['sample_prediction'],axis=1),
      data['baseline_prediction'].mean(1),data['baseline_prediction'].std(1),np.linalg.norm(data['baseline_prediction'],axis=1),
      np.linalg.norm(data['sample_prediction']-data['baseline_prediction'],axis=1),
    ])
    return np.concatenate([glob,summaries,extra],axis=1).astype(np.float32),d.astype(np.float32)

def ratios(scale,d,data):
    pred=data['sample_prediction']+scale[:,None]*d;truth=.5*(data['truth_half1']+data['truth_half2']);mse=np.mean((pred-truth)**2,axis=1);base=np.asarray(data['base_mse']).reshape(-1);rr=mse/base
    return float(mse.sum()/base.sum()),float(rr.max()),rr,pred

def train_one(seed,xtr,xv,dtr,dv,tr,va,cfg,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);m=ScalarMLP(xtr.shape[1]).to(device);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);n=len(xtr);best=1e99;state=None;pat=0;hist=[]
    tx=torch.from_numpy(xtr).to(device);vx=torch.from_numpy(xv).to(device);td=torch.from_numpy(dtr).to(device);vd=torch.from_numpy(dv).to(device)
    ts=torch.from_numpy(tr['sample_prediction'].astype(np.float32)).to(device);tt=torch.from_numpy((.5*(tr['truth_half1']+tr['truth_half2'])).astype(np.float32)).to(device);tb=torch.from_numpy(np.asarray(tr['base_mse']).reshape(-1).astype(np.float32)).to(device)
    bs=cfg['batch_size']
    for ep in range(cfg['max_epochs']):
      m.train();order=np.random.permutation(n);tot=0
      for st in range(0,n,bs):
        ix=torch.as_tensor(order[st:st+bs],device=device);s=m(tx[ix]);pred=ts[ix]+s[:,None]*td[ix];per=(pred-tt[ix]).square().mean(1)/torch.clamp(tb[ix],min=1e-12);loss=per.mean()+1e-4*s.square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();tot+=float(loss.detach())*len(ix)
      m.eval();
      with torch.no_grad():sv=m(vx).cpu().numpy()
      vr,_,_,_=ratios(sv,dv,va);hist.append({'epoch':ep,'train_loss':tot/n,'validation_ratio':vr})
      if vr<best-1e-5:best=vr;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};pat=0
      else:
        pat+=1
        if pat>=cfg['patience']:break
    m.load_state_dict(state);return m,best,hist

def ensemble(models,x,d,data,device):
    xx=torch.from_numpy(x).to(device);sc=[]
    with torch.no_grad():
      for m in models:m.eval();sc.append(m(xx).cpu().numpy())
    sc=np.stack(sc);mean=sc.mean(0);std=sc.std(0);same=(np.all(sc>0,axis=0)|np.all(sc<0,axis=0));shrink=np.abs(mean)/(np.abs(mean)+std+1e-12);final=mean*shrink*same
    return sc,final,shrink,same

def sh(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args();exp=a.experiment;rc=json.loads((exp/'rescue/rescue_config.json').read_text());tr=tm.load_split(exp/'data','train');va=tm.load_split(exp/'data','validation');template=tr['target_delta'].mean(0).astype(np.float32);xtr,dtr=features(tr,template);xv,dv=features(va,template);mu=xtr.mean(0);sd=np.maximum(xtr.std(0),1e-6);xtr=(xtr-mu)/sd;xv=(xv-mu)/sd
 device=torch.device('cpu');torch.set_num_threads(5);torch.set_num_interop_threads(1);models=[];hist=[];bests=[];t0=time.time()
 for seed in rc['model']['seeds']:
  m,b,h=train_one(seed,xtr,xv,dtr,dv,tr,va,rc['model'],device);models.append(m);bests.append(b);hist.append({'seed':seed,'best_validation_ratio':b,'history':h});print(seed,b,flush=True)
 out=exp/'rescue/models';out.mkdir(exist_ok=True);hashes=[]
 for i,m in enumerate(models):
  p=out/f'scalar_model_{i}.pt';torch.save({'state_dict':m.state_dict(),'input_dim':xtr.shape[1]},p);hashes.append(sh(p))
 np.savez_compressed(out/'scalar_normalization_and_template.npz',mean=mu,std=sd,template=template)
 sc,final,shr,same=ensemble(models,xv,dv,va,device);vr,vw,vrr,_=ratios(final,dv,va);ur,uw,_,_=ratios(sc.mean(0),dv,va)
 # train-optimal scale diagnostics only
 truth=.5*(va['truth_half1']+va['truth_half2']);e=truth-va['sample_prediction'];os=np.sum(dv*e,1)/np.maximum(np.sum(dv*dv,1),1e-30);orr,ow,_,_=ratios(os,dv,va)
 res={'rescue_freeze_sha256':rc['freeze_sha256'],'model_hashes':hashes,'normalization_template_sha256':sh(out/'scalar_normalization_and_template.npz'),'training_seconds':time.time()-t0,'validation':{'candidate_ratio':vr,'worst':vw,'wins':int(np.sum(vrr<1)),'ungated_ratio':ur,'ungated_worst':uw,'oracle_scale_along_template_ratio':orr,'oracle_scale_worst':ow,'mean_scale':float(final.mean()),'mean_abs_scale':float(np.mean(np.abs(final))),'abstentions':int(np.sum(~same)),'member_best_ratios':bests},'terminal_test_not_opened':True}
 (exp/'rescue/results/training_validation.json').write_text(json.dumps(res,indent=2));(exp/'rescue/results/training_histories.json').write_text(json.dumps(hist,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
