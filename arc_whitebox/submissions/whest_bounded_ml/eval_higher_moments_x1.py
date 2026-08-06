#!/usr/bin/env python3
"""Evaluate an x1/x1a closure on real `arc-whestbench-higher-moments-2026` files.

This needs only the ~63 MB per-MLP higher-moment files, not the 2.2 GB full K3
tensors. pre_M21 contains exactly the k21 diagonal slice used by x1/x1a.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parent))
from coefnet_numpy_runtime import NumpyCoefNet

def phi(x):return np.exp(-.5*x*x)/math.sqrt(2*math.pi)
def bvn(a,b,r,nodes=16):
 a,b,r=np.broadcast_arrays(a,b,r);r=np.clip(r.astype(float),-.999999,.999999);x,w=np.polynomial.legendre.leggauss(nodes);rr=.5*r[...,None]*(x+1);den=np.maximum(1-rr*rr,1e-14);e=-(a[...,None]**2-2*rr*a[...,None]*b[...,None]+b[...,None]**2)/(2*den);dens=np.exp(e)/(2*math.pi*np.sqrt(den));return np.clip(ndtr(a)*ndtr(b)+.5*r*np.sum(dens*w,axis=-1),0,1)
def gaussian_cov(mu,s,r):
 s=np.maximum(s,1e-12);a=mu/s;root=np.sqrt(np.maximum(1-r*r,1e-14));P=bvn(a[:,None],a[None,:],r);p2=np.exp(-(a[:,None]**2-2*r*a[:,None]*a[None,:]+a[None,:]**2)/(2*np.maximum(1-r*r,1e-14)))/(2*math.pi*root);sec=mu[:,None]*mu[None,:]*P+mu[:,None]*s[None,:]*phi(a[None,:])*ndtr((a[:,None]-r*a[None,:])/root)+mu[None,:]*s[:,None]*phi(a[:,None])*ndtr((a[None,:]-r*a[:,None])/root)+s[:,None]*s[None,:]*(r*P+(1-r*r)*p2);m=mu*ndtr(a)+s*phi(a);diag=(mu*mu+s*s)*ndtr(a)+mu*s*phi(a);np.fill_diagonal(sec,diag);C=sec-np.outer(m,m);return (C+C.T)/2

def connected21(mu,M11,M21,m2):
 return M21-2*mu[:,None]*M11-m2[:,None]*mu[None,:]+2*(mu[:,None]**2)*mu[None,:]
def case(d,layer,model):
 pm=np.asarray(d['pre_mean'][layer],float);P11=np.asarray(d['pre_M11'][layer],float);P21=np.asarray(d['pre_M21'][layer],float);pm2=np.asarray(d['pre_m2'][layer],float);Cpre=P11-np.outer(pm,pm);v=np.maximum(np.diag(Cpre),1e-12);s=np.sqrt(v);rho=np.clip(Cpre/np.outer(s,s),-1,1);np.fill_diagonal(rho,1);k21=connected21(pm,P11,P21,pm2);den=s[:,None]**3+s[None,:]**3;x1=(k21+k21.T)/np.maximum(den,1e-12);x1a=(k21-k21.T)/np.maximum(den,1e-12);mu_post=np.asarray(d['mean'][layer],float);Ctrue=np.asarray(d['M11'][layer],float)-np.outer(mu_post,mu_post);Cbase=gaussian_cov(pm,s,rho);iu,ju=np.triu_indices(len(pm),1);a=pm/s;dif=a[iu]-a[ju];base=np.column_stack([np.full(len(iu),(layer+1)/32),a[iu]+a[ju],a[iu]*a[ju],np.abs(dif),rho[iu,ju]]).astype('f4');pred=model.predict_invariant(base,dif.astype('f4'),x1[iu,ju].astype('f4'),x1a[iu,ju].astype('f4'));scale=s[iu]*s[ju];target=(Ctrue-Cbase)[iu,ju]/scale;bm=float(np.mean(target**2));mm=float(np.mean((pred-target)**2));return {'layer':layer,'base_mse':bm,'model_mse':mm,'gain':bm/max(mm,1e-30),'r2':1-mm/max(bm,1e-30)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('model',type=Path);ap.add_argument('files',type=Path,nargs='+');ap.add_argument('--layers',default='1,4,8,12,16,20,24,28,30');ap.add_argument('--out',type=Path,default=Path('higher_moments_x1_eval.json'));a=ap.parse_args();m=NumpyCoefNet(a.model);layers=[int(x) for x in a.layers.split(',')];rows=[]
 for f in a.files:
  d=np.load(f);fr=[]
  for l in layers:fr.append(case(d,l,m))
  rows.append({'file':str(f),'global_index':int(d['global_index']),'rows':fr});print(json.dumps({'file':str(f),'gain':float(np.mean([x['base_mse'] for x in fr])/np.mean([x['model_mse'] for x in fr]))}),flush=True)
 b=np.array([x['base_mse'] for z in rows for x in z['rows']]);q=np.array([x['model_mse'] for z in rows for x in z['rows']]);res={'rows':rows,'summary':{'base_mse':float(b.mean()),'model_mse':float(q.mean()),'gain':float(b.mean()/q.mean()),'fraction':float(np.mean(q<b))}};a.out.write_text(json.dumps(res,indent=2));print(json.dumps(res['summary']))
if __name__=='__main__':main()
