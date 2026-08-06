from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np,torch
SRC=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE');sys.path.insert(0,str(SRC))
import run_layer31_residual as r
ap=argparse.ArgumentParser();ap.add_argument('--network-seed',type=int,required=True);ap.add_argument('--qmc-seed',type=int,required=True);ap.add_argument('--n',type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--threads',type=int,default=5);a=ap.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
t=time.time();z=r.stream_reference(r.make_weights(a.network_seed),a.n,a.qmc_seed);a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,network_seed=a.network_seed,qmc_seed=a.qmc_seed,n=a.n,**z);print({'out':str(a.out),'runtime':time.time()-t})
