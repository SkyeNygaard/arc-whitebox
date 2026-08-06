from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path('/mnt/data/whest_reopened');A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle';WHITE=ROOT/'arc_code/arc_whitebox';CEIL=ROOT/'arc_code/arc_ceiling'
sys.path[:0]=[str(A9),str(WHITE/'src'),str(CEIL),'/mnt/data/competition_relevance_20260730']
import arc_experiments as ae
import whest.gaussmath as gm
from bivariate_edgeworth import _pair_terms
from bivariate_edgeworth4 import pair_terms4,edgeworth4_mean_correction
from rederive_edgeworth4 import pass4,rel
OUT=Path('/mnt/data/competition_relevance_20260730/edgeworth34_rank1');OUT.mkdir(parents=True,exist_ok=True)
def rankp_offdiag(M,p):
 X=M.copy();np.fill_diagonal(X,0);u,s,vt=np.linalg.svd(X,full_matrices=False);return (u[:,:p]*s[:p])@vt[:p]
def moments34(mu,sig,c21,c31,c22,gsec,kernel_rank=None):
 sd=np.sqrt(np.maximum(np.diag(sig),1e-300));t=mu/sd;rho=np.clip(sig/np.outer(sd,sd),-1+1e-12,1-1e-12);A21,A30=_pair_terms(mu,sd,rho);A31,A22,A40=pair_terms4(mu,sd,rho)
 if kernel_rank is not None:
  A21=rankp_offdiag(A21,kernel_rank);A30=rankp_offdiag(A30,kernel_rank);A31=rankp_offdiag(A31,kernel_rank);A22=rankp_offdiag(A22,kernel_rank);A40=rankp_offdiag(A40,kernel_rank)
 k3=np.diag(c21);k4=np.diag(c22)
 corr3=(k3[:,None]*A30+3*c21*A21+3*c21.T*A21.T+k3[None,:]*A30.T)/6
 corr4=(k4[:,None]*A40+4*c31*A31+6*c22*A22+4*c31.T*A31.T+k4[None,:]*A40.T)/24
 # exact univariate diagonal limits for F(x)=ReLU(x)^2
 p0=norm.pdf(t)/sd
 np.fill_diagonal(corr3,k3*p0/3.0)
 np.fill_diagonal(corr4,-k4*t*norm.pdf(t)/(12.0*sd*sd))
 gmean=mu*norm.cdf(t)+sd*norm.pdf(t)
 m3=gmean-(k3/6)*t*norm.pdf(t)/(sd*sd)
 m4=m3+edgeworth4_mean_correction(mu,sig,k4)
 cov3=gsec+corr3-np.outer(m3,m3)
 cov4=gsec+corr3+corr4-np.outer(m4,m4)
 return cov3,cov4

def run(seed,layer,n):
 t0=time.time();w=ae.make_weights(seed);mu,sig,c21,c31,c22,r1=pass4(w,layer,n,1_000_000+seed+layer);_,_,_,_,_,r2=pass4(w,layer,n,1_100_000+seed+layer);ref=.5*(r1+r2);floor=.5*rel(r1,r2);wn=w[layer+1].astype(float)
 gm0,gc0=gm.relu_cov_from_gauss(mu,sig,n_nodes=12);gsec=gc0+np.outer(gm0,gm0);vg=np.sum((gc0@wn)*wn,axis=0)
 rows={}
 for p in [None,1,2,4,8]:
  c3,c4=moments34(mu,sig,c21,c31,c22,gsec,p);rows['full' if p is None else f'rank{p}']={'e3':rel(np.sum((c3@wn)*wn,axis=0),ref),'e4':rel(np.sum((c4@wn)*wn,axis=0),ref)}
 out={'seed':seed,'layer':layer+1,'nref_each':n,'floor':floor,'gaussian':rel(vg,ref),'rows':rows,'seconds':time.time()-t0};(OUT/f'seed{seed}_layer{layer+1}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--layers',type=int,nargs='+',default=[15,23]);ap.add_argument('--nref',type=int,default=131072);a=ap.parse_args()
 for s in a.seeds:
  for l in a.layers:run(s,l,a.nref)
if __name__=='__main__':main()
