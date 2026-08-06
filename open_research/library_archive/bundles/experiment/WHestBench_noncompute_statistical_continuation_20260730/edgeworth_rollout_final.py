from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path('/mnt/data/whest_reopened');A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle';WHITE=ROOT/'arc_code/arc_whitebox';CEIL=ROOT/'arc_code/arc_ceiling'
sys.path[:0]=[str(A9),str(WHITE/'src'),str(CEIL),'/mnt/data/competition_relevance_20260730']
import arc_experiments as ae
import whest.gaussmath as gm
from edgeworth34_rank1_eval import moments34
from bivariate_edgeworth4 import edgeworth4_mean_correction
from sample_edgeworth34_lowrank import cumulants
D=256
OUT=Path('/mnt/data/competition_relevance_20260730/edgeworth_rollout_final');OUT.mkdir(parents=True,exist_ok=True)

def mc_final(w,n,seed,chunk=4096):
 rng=np.random.default_rng(seed);acc=np.zeros(D);acc2=np.zeros(D);done=0
 while done<n:
  b=min(chunk,n-done);a=rng.standard_normal((b,D),dtype=np.float32)
  for W in w:a=np.maximum(a@W,0)
  acc+=a.sum(0,dtype=np.float64);acc2+=(a.astype(np.float64)**2).sum(0)
  done+=b
 return acc/n,acc2/n-(acc/n)**2

def kerdock_to_layer(w,layer,rot):
 a=ae.first_activation(w[0],rot)
 if layer==0:raise ValueError('layer0 unsupported')
 for li,W in enumerate(w[1:],start=1):
  h=a@W
  if li==layer:return h.astype(np.float64),a
  a=np.maximum(h,0)
 raise ValueError(layer)

def continue_exact(H,w,start_layer):
 a=np.maximum(H,0).astype(np.float32)
 for li in range(start_layer+1,len(w)):a=np.maximum(a@w[li],0)
 return a.mean(0,dtype=np.float64)

def psd(C):
 C=(C+C.T)*.5;e,V=np.linalg.eigh(C);neg=float(np.sum(np.maximum(-e,0)));pos=float(np.sum(np.maximum(e,0)));e=np.maximum(e,1e-12);return (V*e)@V.T,neg/max(pos,1e-30),float(np.min(e))

def propagate(m,C,w,start_layer):
 repairs=[]
 for li in range(start_layer+1,len(w)):
  W=w[li].astype(np.float64);mu=m@W;S=W.T@C@W;S,neg,_=psd(S);repairs.append(neg);m,C=gm.relu_cov_from_gauss(mu,S,n_nodes=12)
 return m,repairs

def start_states(H,kernel_rank):
 mu=H.mean(0);z=H-mu;sig=z.T@z/len(z);c21,c31,c22=cumulants(H)
 gm0,gc0=gm.relu_cov_from_gauss(mu,sig,n_nodes=12);gsec=gc0+np.outer(gm0,gm0)
 c3,c4=moments34(mu,sig,c21,c31,c22,gsec,kernel_rank)
 sd=np.sqrt(np.maximum(np.diag(sig),1e-300));t=mu/sd;k3=np.diag(c21);k4=np.diag(c22)
 gmean=mu*norm.cdf(t)+sd*norm.pdf(t)
 m3=gmean-(k3/6)*t*norm.pdf(t)/(sd*sd)
 m4=m3+edgeworth4_mean_correction(mu,sig,k4)
 emp=np.maximum(H,0);em=emp.mean(0);ec=np.cov(emp,rowvar=False,bias=True)
 gc0,ng,_=psd(gc0);c3,n3,_=psd(c3);c4,n4,_=psd(c4);ec,ne,_=psd(ec)
 return {'gaussian':(gmean,gc0,ng),'edgeworth3':(m3,c3,n3),'edgeworth4':(m4,c4,n4),'empirical_state':(em,ec,ne)}

def mse(a,b):return float(np.mean((a-b)**2))
def run(seed,layer,nref,rot,kernel_rank):
 t=time.time();w=ae.make_weights(seed);r1,_=mc_final(w,nref,2_000_000+seed+layer);r2,_=mc_final(w,nref,2_100_000+seed+layer);truth=.5*(r1+r2);noise=mse(r1,r2)/4
 H,_=kerdock_to_layer(w,layer,rot);base=continue_exact(H,w,layer);bm=mse(base,truth);states=start_states(H,kernel_rank);rows={}
 for name,(m,C,startneg) in states.items():
  pred,reps=propagate(m,C,w,layer);mm=mse(pred,truth);rows[name]={'ratio':mm/bm,'mse':mm,'start_negative_fraction':startneg,'max_propagation_negative_fraction':max(reps) if reps else 0.0,'prediction_norm':float(np.linalg.norm(pred))}
 out={'seed':seed,'layer':layer+1,'rot':rot,'nref_each':nref,'kernel_rank':kernel_rank,'base_mse':bm,'reference_noise':noise,'rows':rows,'seconds':time.time()-t}
 (OUT/f'seed{seed}_layer{layer+1}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--layers',type=int,nargs='+',default=[15,23]);ap.add_argument('--nref',type=int,default=32768);ap.add_argument('--rot',type=int,default=3);ap.add_argument('--kernel-rank',type=int,default=4);a=ap.parse_args()
 for s in a.seeds:
  for l in a.layers:run(s,l,a.nref,a.rot,a.kernel_rank)
