from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/whest_reopened');A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle';WHITE=ROOT/'arc_code/arc_whitebox';CEIL=ROOT/'arc_code/arc_ceiling'
sys.path[:0]=[str(A9),str(WHITE/'src'),str(CEIL)]
import arc_experiments as ae
import whest.gaussmath as gm
from bivariate_edgeworth import edgeworth3_mean,edgeworth3_second_moment,_pair_terms
from edgeworth_correction_rank import moments,relvar
D=256;OUT=Path('/mnt/data/competition_relevance_20260730/edgeworth_coefficient_rank');OUT.mkdir(parents=True,exist_ok=True)
def run(seed,layer,nref):
 t0=time.time();w=ae.make_weights(seed);mu,sig,c,ref1=moments(w,layer,nref,300000+seed+layer);_,_,_,ref2=moments(w,layer,nref,400000+seed+layer);ref=.5*(ref1+ref2);floor=.5*relvar(ref1,ref2);wn=w[layer+1].astype(float)
 sd=np.sqrt(np.maximum(np.diag(sig),1e-300));rho=np.clip(sig/np.outer(sd,sd),-1+1e-12,1-1e-12);A,B=_pair_terms(mu,sd,rho) # A=a_iij, B=a_iii
 gm0,gc0=gm.relu_cov_from_gauss(mu,sig,n_nodes=12);gsec=gc0+np.outer(gm0,gm0);em=edgeworth3_mean(mu,sig,c);fullsec=edgeworth3_second_moment(mu,sig,c,gsec);fullcov=fullsec-np.outer(em,em);fullvar=np.sum((fullcov@wn)*wn,axis=0);gvar=np.sum((gc0@wn)*wn,axis=0)
 k=np.diag(c);mean_delta=em-gm0
 # marginal-only covariance correction: k_i B_ij + k_j B_ji, plus mean correction
 margsec=(k[:,None]*B+k[None,:]*B.T)/6
 margcov=margsec-(gm0[:,None]*mean_delta[None,:]+mean_delta[:,None]*gm0[None,:]+mean_delta[:,None]*mean_delta[None,:])
 margvar=np.sum(((gc0+margcov)@wn)*wn,axis=0)
 UA,SA,VTA=np.linalg.svd(A,full_matrices=False);UB,SB,VTB=np.linalg.svd(B,full_matrices=False)
 ea=np.cumsum(SA*SA)/np.sum(SA*SA);eb=np.cumsum(SB*SB)/np.sum(SB*SB)
 rows={}
 for p in [1,2,4,8,16,32,64]:
  Ap=(UA[:,:p]*SA[:p])@VTA[:p];Bp=(UB[:,:p]*SB[:p])@VTB[:p]
  sec=(k[:,None]*Bp+3*c*Ap+3*c.T*Ap.T+k[None,:]*Bp.T)/6
  cov=gsec+sec-np.outer(em,em)
  var=np.sum((cov@wn)*wn,axis=0)
  rows[str(p)]={'A_energy':float(ea[p-1]),'B_energy':float(eb[p-1]),'var_error':relvar(var,ref)}
 out={'seed':seed,'layer':layer+1,'nref_each':nref,'floor':floor,'gaussian':relvar(gvar,ref),'marginal_only':relvar(margvar,ref),'full':relvar(fullvar,ref),'ranks':rows,'seconds':time.time()-t0}
 (OUT/f'seed{seed}_layer{layer+1}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--layers',type=int,nargs='+',default=[15,23]);ap.add_argument('--nref',type=int,default=131072);a=ap.parse_args()
 for s in a.seeds:
  for l in a.layers:run(s,l,a.nref)
if __name__=='__main__':main()
