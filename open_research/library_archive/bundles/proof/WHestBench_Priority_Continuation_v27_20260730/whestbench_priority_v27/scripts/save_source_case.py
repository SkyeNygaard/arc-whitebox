import sys, numpy as np, argparse
from pathlib import Path
OGAP=Path('/mnt/data/priority_v26_inputs/ogap/whest_experiments_oracle_gap_20260730');sys.path.insert(0,str(OGAP/'src'));import run_fresh_suite as core
p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('rot',type=int);p.add_argument('out');a=p.parse_args()
W=core.make_weights(a.seed);means,cps,_=core.forward_full(W,core.rotation(a.rot),checkpoint_depths={31});h=next(x[1] for x in cps if x[0]==31);base=h.mean(0,dtype=np.float64);gy=h.reshape(129,512,256).mean(1,dtype=np.float64);yc=gy-gy.mean(0);_,s,vh=np.linalg.svd(yc,full_matrices=False);np.savez_compressed(a.out,U40=vh[:40].T,s=s[:40],base=base);print('done',a.out)
