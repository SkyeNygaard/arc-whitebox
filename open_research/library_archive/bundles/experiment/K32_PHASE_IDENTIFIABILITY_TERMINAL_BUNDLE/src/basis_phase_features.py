#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys,math,time
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import radial_core as r;import generate_dataset as gd

def skew_kurt(z):
 z=np.asarray(z,float);m=z.mean();s=z.std();u=(z-m)/max(s,1e-12);return [s,float(np.mean(u**3)),float(np.mean(u**4)-3),float(np.min(u)),float(np.max(u)),float(np.quantile(u,.1)),float(np.quantile(u,.25)),float(np.quantile(u,.75)),float(np.quantile(u,.9)),float(np.mean(np.abs(u)))]

def features_for(network_id:int,rotation_seed:int,datafile:Path,rot_index:int):
 with np.load(datafile,allow_pickle=True) as z:
  beta=z['beta_bar'][rot_index].astype(np.float64);delta=z['target_delta'][rot_index].astype(np.float64);idx=z['probe_indices'][rot_index];dirs=z['probe_directions'][rot_index].astype(np.float64);q=z['q_anchor'][rot_index].astype(np.float64);sample=z['sample_prediction'][rot_index].astype(np.float64);base=z['baseline_prediction'][rot_index].astype(np.float64)
 ws,_,_=r.make_weights(network_id);x=gd.make_kerdock(rotation_seed);H,Y=r.forward_target_final(torch.from_numpy(x),ws);H=H.double().numpy();Y=Y.double().numpy();m=H.mean(0);rho=r.chi_mean(r.D)
 X=r.radial_features_sample_rows(H,m,idx,dirs,rho)
 Hb=H.reshape(r.N_BASES,r.ROWS_PER_BASIS,r.D).mean(1);H2b=(H*H).reshape(r.N_BASES,r.ROWS_PER_BASIS,r.D).mean(1);Yb=Y.reshape(r.N_BASES,r.ROWS_PER_BASIS,r.D).mean(1);Xb=X.reshape(r.N_BASES,r.ROWS_PER_BASIS,len(idx)).mean(1)
 tcorr=delta@beta # oracle only? do NOT use in features. use fixed template outside.
 # Runtime legal directions: sample-baseline and beta row-space modes.
 dirs_out=[]
 sb=sample-base
 for v in [sample,base,sb]:
  n=np.linalg.norm(v);dirs_out.append(v/max(n,1e-12))
 # right singular modes of beta are legal and low-dimensional
 _,_,vh=np.linalg.svd(beta,full_matrices=False)
 dirs_out.extend(vh[:8])
 f=[];names=[]
 def add(prefix,z):
  vals=skew_kurt(z);f.extend(vals);names.extend([prefix+':'+k for k in ['std','skew','excess_kurt','minz','maxz','q10','q25','q75','q90','mean_absz']])
 # basis output projections and cross moments
 for j,v in enumerate(dirs_out):
  zy=(Yb-Yb.mean(0))@v;add(f'yproj{j}',zy)
  zh=(Hb-Hb.mean(0))@(np.abs(v) if len(v)==r.D else v);add(f'hproxy{j}',zh)
  f.extend([float(np.mean(zy*zh)),float(np.mean((zy**2)*zh)),float(np.mean(zy*(zh**2)))]);names.extend([f'cross{j}:11',f'cross{j}:21',f'cross{j}:12'])
 # per-probe basis feature deviations, pooled over probe ranks
 DX=Xb-q[None,:]
 # fixed rank-template unavailable in this worker; save rank summaries and covariance spectrum
 for stat,arr in [('mean',DX.mean(1)),('l2',np.linalg.norm(DX,axis=1)),('max',np.max(np.abs(DX),axis=1))]:add('xblock_'+stat,arr)
 # cross-output contraction between block feature deviations and block output deviations
 C=DX.T@(Yb-Yb.mean(0))/r.N_BASES
 s=np.linalg.svd(C,compute_uv=False)
 f.extend(s[:16].tolist()+[float(np.sum(s*s)),float(s[0]/max(np.sum(s),1e-12))]);names.extend([f'xy_s{i}' for i in range(16)]+['xy_frob2','xy_top_fraction'])
 # basis covariance spectra in output and target spaces
 for pref,A in [('Yb',Yb-Yb.mean(0)),('Hb',Hb-Hb.mean(0)),('H2b',H2b-H2b.mean(0))]:
  sv=np.linalg.svd(A,compute_uv=False);f.extend(sv[:16].tolist()+[float(np.sum(sv*sv)),float(sv[0]/max(np.sum(sv),1e-12))]);names.extend([f'{pref}_s{i}' for i in range(16)]+[f'{pref}_frob2',f'{pref}_top_fraction'])
 return np.asarray(f,np.float32),np.asarray(names),DX.astype(np.float32),Yb.astype(np.float32),Hb.astype(np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--datafile',type=Path,required=True);ap.add_argument('--rot-index',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 with np.load(a.datafile) as z:nid=int(z['network_id']);rot=int(z['rotation_seeds'][a.rot_index])
 torch.set_num_threads(2);torch.set_num_interop_threads(1);t=time.time();f,n,dx,yb,hb=features_for(nid,rot,a.datafile,a.rot_index);a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,network_id=nid,rotation_seed=rot,features=f,names=n,dx=dx,yb=yb,hb=hb,runtime=time.time()-t);print(nid,rot,len(f),time.time()-t)
if __name__=='__main__':main()
