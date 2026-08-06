#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
import orientation_codebook_experiment as ex
EPS=1e-30
EXTRA=list(range(1170016,1170032))

def update_case(r,truth,noise):
 y0=np.asarray(r['y0']);base=np.asarray(r['base_output']);err=truth-y0
 r['truth']=truth.tolist();r['truth_noise_mse']=float(noise);r['baseline_mse']=float(np.mean((base-truth)**2));r['baseline_mse_nc']=max(r['baseline_mse']-noise,1e-20);r['y0_mse']=float(np.mean((y0-truth)**2))
 for o in r['orientations']:
  corr=np.asarray(o['correction']);pred=y0+corr;mse=float(np.mean((pred-truth)**2));mn=max(mse-noise,1e-20);ip=float(err@corr)
  o['mse']=mse;o['mse_nc']=mn;o['ratio']=mse/r['baseline_mse'];o['ratio_nc']=mn/r['baseline_mse_nc'];o['error_correction_ip']=ip;o['correction_cosine']=float(ip/(np.linalg.norm(err)*np.linalg.norm(corr)+EPS));opt=float(np.clip(ip/(corr@corr+EPS),-2,2));o['oracle_scale']=opt;o['oracle_scale_mse']=float(np.mean((y0+opt*corr-truth)**2))
 return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--src',type=Path,required=True);ap.add_argument('--dst',type=Path,required=True);ap.add_argument('--asset',type=Path,required=True);a=ap.parse_args()
 if a.dst.exists():shutil.rmtree(a.dst)
 shutil.copytree(a.src,a.dst)
 z=np.load(a.asset);chirps=z['chirps'].astype(np.float32)
 m=json.loads((a.dst/'freeze_manifest.json').read_text());m['reference_rotation_seeds']=list(range(1170000,1170032));m['reference_complete_designs_per_group']=8;m['reference_groups']=4;m['reference_extension']='frozen candidate outputs; reference-only precision extension after all candidate rules were preregistered'
 (a.dst/'freeze_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 for bi,b in enumerate(m['base_ids']):
  old=np.load(a.src/f'truth_{b}.npz');g1=old['t1'];g2=old['t2'];w=ex.make_weights(730000+b);vals=[]
  for s in EXTRA:vals.append(ex.complete_design_mean(w,chirps,ex.haar(s)))
  vals=np.asarray(vals);g3=vals[:8].mean(0);g4=vals[8:].mean(0);groups=np.asarray([g1,g2,g3,g4]);truth=groups.mean(0);noise=float(np.mean(np.var(groups,axis=0,ddof=1)/4))
  np.savez_compressed(a.dst/f'truth_{b}.npz',truth=truth,t1=g1,t2=g2,t3=g3,t4=g4,groups=groups)
  for v in range(m['variants']):
   p=a.dst/f'case_{b}_{v}.json';r=json.loads(p.read_text());p.write_text(json.dumps(update_case(r,truth,noise))+'\n')
  print(json.dumps({'done':f'{bi+1}/{len(m["base_ids"])}','base':b,'noise_mse':noise}),flush=True)
if __name__=='__main__':main()
