#!/usr/bin/env python3
import argparse,json,os
from pathlib import Path
import torch
import reanchored_pilot_defect as r
p=argparse.ArgumentParser(); p.add_argument('--network',type=int,required=True); p.add_argument('--truth-n',type=int,default=32768); p.add_argument('--chunk',type=int,default=8192); p.add_argument('--threads',type=int,default=6); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
xk,_=r.fr.make_kerdock(); rec=r.run_one(a.network,xk,a.truth_n,a.chunk); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(rec,indent=2)); print(json.dumps({'network':a.network,'runtime':rec['runtime_seconds'],'cos':rec['anchor']['candidate_cosine'],'relerr':rec['anchor']['candidate_relative_error'],'disagreement':rec['anchor']['pilot_disagreement'],'oracle_ratio':rec['oracle_lower_mse']/rec['baseline_mse'],'best_network_ratio':min(rec['candidate_mse_grid'])/rec['baseline_mse']}))
