from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/whest_reopened');A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle';WHITE=ROOT/'arc_code/arc_whitebox';CEIL=ROOT/'arc_code/arc_ceiling'
sys.path[:0]=[str(A9),str(WHITE/'src'),str(CEIL),'/mnt/data/competition_relevance_20260730']
import arc_experiments as ae
import whest.gaussmath as gm
from rederive_edgeworth4 import pass4,rel
from edgeworth34_rank1_eval import moments34
from sample_edgeworth34_lowrank import cumulants,trunc
D=256;OUT=Path('/mnt/data/competition_relevance_20260730/kerdock_edgeworth34');OUT.mkdir(parents=True,exist_ok=True)
def kerdock_h(w,layer,rot):
 A=ae.first_activation(w[0],rot)
 if layer==0:return None
 B=np.empty_like(A)
 for li,W in enumerate(w[1:],start=1):
  H=A@W
  if li==layer:return H.astype(float)
  np.maximum(H,0,out=H);A=H

def run(seed,layer,nref,rot):
 t=time.time();w=ae.make_weights(seed);mu,sig,_,_,_,r1=pass4(w,layer,nref,1_500_000+seed+layer);_,_,_,_,_,r2=pass4(w,layer,nref,1_600_000+seed+layer);ref=.5*(r1+r2);floor=.5*rel(r1,r2);wn=w[layer+1].astype(float);gm0,gc0=gm.relu_cov_from_gauss(mu,sig,n_nodes=12);gsec=gc0+np.outer(gm0,gm0);H=kerdock_h(w,layer,rot);c21,c31,c22=cumulants(H)
 rows={}
 for p in [1,2,4,8]:
  _,c4=moments34(mu,sig,c21,c31,c22,gsec,p);dc=c4-gc0;item={'full_correction':rel(np.sum((c4@wn)*wn,axis=0),ref)}
  for r in [8,16,32,64,96,128]:item[f'corr_rank{r}']=rel(np.sum(((gc0+trunc(dc,r))@wn)*wn,axis=0),ref)
  rows[f'kernel_rank{p}']=item
 out={'seed':seed,'layer':layer+1,'rot':rot,'nref_each':nref,'floor':floor,'gaussian':rel(np.sum((gc0@wn)*wn,axis=0),ref),'rows':rows,'seconds':time.time()-t};(OUT/f'seed{seed}_layer{layer+1}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--layers',type=int,nargs='+',default=[15,23]);ap.add_argument('--nref',type=int,default=131072);ap.add_argument('--rot',type=int,default=3);a=ap.parse_args()
 for s in a.seeds:
  for l in a.layers:run(s,l,a.nref,a.rot)
if __name__=='__main__':main()
