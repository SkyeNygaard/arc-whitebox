#!/usr/bin/env python3
import argparse,json,sys,time
from pathlib import Path
import numpy as np
T4=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code');sys.path.insert(0,str(T4));import frozen_reference_impl as fr
from run_stein_grid import build_master
from run_stein_screen import effective_j,forward_kerdock
OUT=Path('/mnt/data/work/new_opportunities/selected/signed_weights');OUT.mkdir(parents=True,exist_ok=True)
CONFIGS={'primary_tanh_k8_alpha2':('tanh_k8_hp',1.0,2.0),'primary_tanh_k8_alpha1':('tanh_k8_hp',1.0,1.0),'harmonic_d68_alpha1':('harm_k8_d6_8',1e-8,1.0)}
def one(net,xk):
 t=time.time();ws,_,_=fr.make_weights(net);Y,gates=forward_kerdock(xk,ws);U=np.linalg.svd(effective_j(ws,gates),full_matrices=False)[0];G,sets=build_master(xk,U);out={'network_id':net,'candidates':{}}
 n=len(G)
 for label,(name,ridge,alpha) in CONFIGS.items():
  X=G[:,sets[name]];gm=X.mean(0);Xc=X-gm;gram=Xc.T@Xc;scale=max(np.trace(gram)/len(gm),1e-30);a=np.linalg.solve(gram+ridge*scale*np.eye(len(gm)),gm);delta=-Xc@a;w=1/n+alpha*delta
  out['candidates'][label]={'negative_mass':float(np.maximum(-w,0).sum()),'l1':float(np.abs(w).sum()),'min_weight':float(w.min()),'max_weight':float(w.max()),'ess':float(1/np.sum(w*w)),'sum_weight':float(w.sum()),'control_residual_norm':float(np.linalg.norm(w@X))}
 out['seconds']=time.time()-t;(OUT/f'weights_{net}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('nets',nargs='+',type=int);a=ap.parse_args();xk,_=fr.make_kerdock();
 for n in a.nets:one(n,xk)
if __name__=='__main__':main()
