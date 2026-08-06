from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--n',type=int,default=262144);ap.add_argument('--pairs-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
t=np.load(a.pairs_dir/f'target_{a.seed}_pair_n{a.n}.npz');h=np.load(a.pairs_dir/f'anchor_{a.seed}_pair_n{a.n}.npz')
data={k:np.asarray(t[k]) for k in ['target_final1','target_final2','target_pen1','target_pen2']}|{k:np.asarray(h[k]) for k in ['anchor_mu1','anchor_mu2','anchor_M1','anchor_M2','anchor_raw1','anchor_raw2']}
data|={'qmc_n':np.array(a.n),'target_seed1':t['seed1'],'target_seed2':t['seed2'],'anchor_seed1':h['seed1'],'anchor_seed2':h['seed2'],'fast_reference_version':np.array('paired-fast-v1')}
a.out.parent.mkdir(parents=True,exist_ok=True);np.savez(a.out,**data);print(a.out)
