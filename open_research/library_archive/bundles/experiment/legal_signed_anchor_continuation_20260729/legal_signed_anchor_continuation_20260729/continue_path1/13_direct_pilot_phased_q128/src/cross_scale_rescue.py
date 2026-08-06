#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
sys_path=Path(__file__).resolve().parent
import sys;sys.path.insert(0,str(sys_path))
from base import bootstrap_ratio

def apply_record(r):
 p1=r['methods']['1024']['projection_min']
 p2m=r['methods']['2048']['projection_mean']
 apply=(p1>=0.2 and p2m>=0.0)
 source=r['methods']['1024']['grid']['t0.2_l0.2']
 return (source if apply else r['first_mse']),apply,p1,p2m

def block(rs,seed):
 b=np.array([r['baseline_mse'] for r in rs]);vals=[apply_record(r) for r in rs];c=np.array([x[0] for x in vals]);rat=c/b
 return {'n':len(rs),'candidate_over_base':float(c.sum()/b.sum()),'ci95':bootstrap_ratio(b,c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'applied':int(sum(x[1] for x in vals)),'per_network':rat.tolist(),'details':[{'network':r['network_id'],'apply':x[1],'projection1024_min':x[2],'projection2048_mean':x[3],'ratio':float(y)} for r,x,y in zip(rs,vals,rat)]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--records-dir',type=Path,required=True);p.add_argument('--split-at',type=int,default=18);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rs=sorted([json.loads(x.read_text()) for x in a.records_dir.glob('network_*.json')],key=lambda r:r['network_id']);z={'frozen_rule':{'projection_1024_min_ge':0.2,'projection_2048_mean_ge':0.0,'source_lambda':0.2,'fallback':'exact_first_layer_q128'},'development_ids':[r['network_id'] for r in rs[:a.split_at]],'holdout_ids':[r['network_id'] for r in rs[a.split_at:]],'development':block(rs[:a.split_at],20261400),'holdout':block(rs[a.split_at:],20261401) if len(rs)>a.split_at else {}};a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
if __name__=='__main__':main()
