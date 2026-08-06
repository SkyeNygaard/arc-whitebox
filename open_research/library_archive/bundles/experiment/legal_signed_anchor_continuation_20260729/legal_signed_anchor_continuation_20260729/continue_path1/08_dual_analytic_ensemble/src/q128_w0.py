#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,observed_covariance
D=fr.D;TARGET=fr.TARGET

def run(n,xk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);x=torch.from_numpy(xk.copy());gates=[];first=None;Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);gates.append((pre>0).double().mean(0).cpu().numpy())
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET:Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa)
 p=len(idx);U=np.eye(D)[:,idx];V=dirs.T
 for l in range(TARGET-1,-1,-1):
  A=ws[l+1].double().numpy()*gates[l+1][None,:];U=A@U;V=A@V
 dm0,dc0=first;dmi=dm0@U;dmv=dm0@V;dcii=np.einsum('ip,ij,jp->p',U,dc0,U);dciv=np.einsum('ip,ij,jp->p',U,dc0,V)
 mi=m[idx];mv=dirs@m;meani=mi+dmi;meanv=mv+dmv;scale=D/(rho*rho);sd=(H*H).mean(0)*scale;sr=(H[:,idx]*(H@dirs.T)).mean(0)*scale;diag=sd[idx]+dcii+meani*meani-mi*mi;row=sr+dciv+meani*meanv-mi*mv;anchor=(diag*dmv+2*dmi*row+2*(mi*mi-meani*meani)*meanv)/(D+1);pred,_=fr.estimate_from_fit(fit,sa+anchor)
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'baseline_pred':base.tolist(),'delta_output':(pred-base).tolist(),'anchor_norm':float(np.linalg.norm(anchor)),'runtime_seconds':time.perf_counter()-t}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int,required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(a.threads);xk,_=fr.make_kerdock();z=run(a.network,xk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds']}))
if __name__=='__main__':main()
