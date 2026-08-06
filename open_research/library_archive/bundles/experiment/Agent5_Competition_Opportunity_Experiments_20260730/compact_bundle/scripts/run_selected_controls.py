#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np
T4=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(T4)); import frozen_reference_impl as fr
from run_stein_grid import build_master,raw_stats,subtract,centered,solve,metric
from run_stein_screen import effective_j,qmc_final,forward_kerdock,mse_obs,mse_unb
D=256;NB=129;RPB=512
OUT=Path('/mnt/data/work/new_opportunities/selected');OUT.mkdir(parents=True,exist_ok=True)
CONFIGS={
'harmonic_d68_k8':('harm_k8_d6_8',1e-8,16),
'stein_sigtanh_k2_rr2':('sigtanh_k2_hp',1e-8,2),
'stein_tanh_k8_stable':('tanh_k8_hp',1.0,0),
'stein_tanh_k4_stable':('tanh_k4_hp',1.0,0),
'stein_sigmoid_k4_mid':('sigmoid_k4_hp',0.1,0),
'stein_sigtanh_k4_heavy':('sigtanh_k4_hp',10.0,0),
}

def run(net,nref,xk,bid,stage):
 t0=time.time();ws,whash,_=fr.make_weights(net);a=qmc_final(ws,nref,2000000+2*net);b=qmc_final(ws,nref,2000001+2*net);truth=(a+b)/2
 Y,gates=forward_kerdock(xk,ws);Y=Y.astype(np.float64,copy=False);base=Y.mean(0);U=np.linalg.svd(effective_j(ws,gates),full_matrices=False)[0]
 G,sets=build_master(xk,U);fullst=raw_stats(G,Y,np.ones(len(G),bool));tests=[];trains=[]
 for f in range(4):
  te=(bid%4)==f;ts=raw_stats(G,Y,te);tests.append(ts);trains.append(subtract(fullst,ts))
 row={'stage':stage,'network_id':net,'nref_per_half':nref,'weight_sha256':whash,'base_observed_mse':mse_obs(base,truth),'base_unbiased_mse':mse_unb(base,a,b),'reference_noise_mse':float(np.mean((a-b)**2)/4),'candidates':{}}
 for label,(fname,ridge,rank) in CONFIGS.items():
  idx=sets[fname];gg,gy,mg,my=centered(fullst,idx);B=solve(gg,gy,ridge,rank);full=my-mg@B
  vals=[];ns=[]
  for tr,te in zip(trains,tests):
   gg,gy,_,_=centered(tr,idx);Bt=solve(gg,gy,ridge,rank);mgt=te['sumg'][idx]/te['n'];myt=te['sumy']/te['n'];vals.append(myt-mgt@Bt);ns.append(te['n'])
  cf=np.average(np.stack(vals),axis=0,weights=ns)
  row['candidates'][label]={'feature_set':fname,'ridge':ridge,'rank':rank or 'full','full':metric(base,full,truth,a,b),'cf4':metric(base,cf,truth,a,b)}
 row['seconds']=time.time()-t0;(OUT/f'{stage}_network_{net}.json').write_text(json.dumps(row,sort_keys=True))
 best=min((v['cf4']['unbiased_mse'],k) for k,v in row['candidates'].items());print(json.dumps({'stage':stage,'network':net,'seconds':row['seconds'],'base_unb':row['base_unbiased_mse'],'noise_fraction':row['reference_noise_mse']/max(row['base_observed_mse'],1e-30),'best_cf4_unb':best}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('networks',nargs='+',type=int);ap.add_argument('--nref',type=int,default=262144);ap.add_argument('--stage',default='screen_hp');args=ap.parse_args();xk,_=fr.make_kerdock();bid=np.repeat(np.arange(NB),RPB)
 for n in args.networks:run(n,args.nref,xk,bid,args.stage);gc.collect()
if __name__=='__main__':main()
