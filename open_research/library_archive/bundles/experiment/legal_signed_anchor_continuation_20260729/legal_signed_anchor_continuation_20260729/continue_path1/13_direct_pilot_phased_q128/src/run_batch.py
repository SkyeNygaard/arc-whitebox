#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from direct_phase_q128 import run
p=argparse.ArgumentParser();p.add_argument('--networks',type=int,nargs='+',required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);a.outdir.mkdir(parents=True,exist_ok=True);xk,_=fr.make_kerdock()
for n in a.networks:
 z=run(n,xk,a.truth_n,4096);(a.outdir/f'network_{n}.json').write_text(json.dumps(z));print(json.dumps({'network':n,'runtime':z['runtime_seconds'],'first':z['first_mse']/z['baseline_mse'],'p1024':z['methods']['1024']['projection_min'],'p2048mean':z['methods']['2048']['projection_mean']}),flush=True)
