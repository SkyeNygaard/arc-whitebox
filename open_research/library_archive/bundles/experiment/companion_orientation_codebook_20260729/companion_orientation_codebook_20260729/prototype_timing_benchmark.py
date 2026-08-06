#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import numpy as np
import orientation_codebook_experiment as ex

def bench(fn,reps=5):
 vals=[]
 for _ in range(reps):
  t=time.perf_counter();fn();vals.append(time.perf_counter()-t)
 return {'median_seconds':float(np.median(vals)),'p90_seconds':float(np.quantile(vals,.9)),'all_seconds':vals}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reps',type=int,default=5);a=ap.parse_args()
 z=np.load(a.asset);chirps=z['chirps'].astype(np.float32);main=z['rotation'].astype(np.float32);rots=[ex.haar(s) for s in ex.ORIENTATION_SEEDS[:8]];w=ex.make_weights(730000+3881069650)
 keep=list(range(111))+[128];missing=list(range(111,128))
 def prop(blocks,rot,target,final):
  x=ex.first_blocks(w[0],rot,blocks,chirps);return ex.propagate(x,w,1,target,final)
 # warmups
 prop([0,1],rots[0],ex.TARGET,False);prop(keep,main,ex.TARGET,True)
 main112=bench(lambda:prop(keep,main,ex.TARGET,True),a.reps)
 fixed17=bench(lambda:prop(list(range(17)),rots[2],ex.TARGET,False),a.reps)
 probes8=bench(lambda:[prop([0,1],r,ex.TARGET,False) for r in rots],a.reps)
 remaining15=bench(lambda:prop(list(range(2,17)),rots[0],ex.TARGET,False),a.reps)
 missing17=bench(lambda:prop(missing,main,ex.TARGET,True),a.reps)
 out={'prototype':'dense NumPy/BLAS research timing; not FlopScope residual wall','repetitions':a.reps,'main112_full':main112,'fixed17_to_anchor':fixed17,'eight_two_basis_probes_to_anchor':probes8,'selected_remaining15_to_anchor':remaining15,'missing17_primary_full_fallback':missing17,
 'composed_medians':{'fixed_package':main112['median_seconds']+fixed17['median_seconds'],'k8_selector':main112['median_seconds']+probes8['median_seconds']+remaining15['median_seconds'],'safe_fallback':main112['median_seconds']+probes8['median_seconds']+missing17['median_seconds']},
 'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
