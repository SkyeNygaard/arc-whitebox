#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CANDIDATES=('baseline','two_layer','fixed_three','adaptive_2_6')

def bootstrap_ratio(candidate,baseline,reps=10000,seed=20260729):
    c=np.asarray(candidate,float);b=np.asarray(baseline,float);rng=np.random.default_rng(seed);n=len(c);vals=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,n,n);vals[i]=c[ix].mean()/b[ix].mean()
    return [float(x) for x in np.quantile(vals,[.025,.975])]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--asset',type=Path,default=HERE.parent/'assets/kerdock_mub5_seed3.npz');ap.add_argument('--indices',type=int,nargs='+',default=list(range(100)));ap.add_argument('--outdir',type=Path,default=HERE.parent/'results'/'official_paired');a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    missing=[]
    if not a.data.exists():missing.append(str(a.data))
    if not a.asset.exists():missing.append(str(a.asset))
    try: import flopscope,pyarrow
    except Exception as exc:missing.append(f'python dependency: {exc}')
    if missing: raise SystemExit('EXTERNALLY BLOCKED:\n- '+'\n- '.join(missing))
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    outputs={}
    for candidate in CANDIDATES:
        out=a.outdir/f'{candidate}.json';cmd=[sys.executable,str(HERE/'official_worker.py'),'--candidate',candidate,'--data',str(a.data),'--asset',str(a.asset),'--indices',*[str(x) for x in a.indices],'--out',str(out)]
        if candidate!='baseline':cmd += ['--baseline-means',str((a.outdir/'baseline.json').with_suffix('.means.npz'))]
        subprocess.run(cmd,check=True,env=env);outputs[candidate]=json.loads(out.read_text())
    base={r['index']:r for r in outputs['baseline']['rows']};summary={};csv_rows=[]
    for candidate,payload in outputs.items():
        rows=payload['rows'];adjusted=[r['adjusted_score'] for r in rows];raw=[r['raw_final_mse'] for r in rows]
        timing_keys=sorted({key for row in rows for key in row.get('timings',{})})
        mean_timings={key:float(np.mean([row.get('timings',{}).get(key,0.0) for row in rows])) for key in timing_keys}
        item={'networks':len(rows),'mean_raw_final_mse':float(np.mean(raw)),'mean_adjusted_score':float(np.mean(adjusted)),'mean_tracked_flops':float(np.mean([r['tracked_flops'] for r in rows])),'mean_residual_wall_time_s':float(np.mean([r['residual_wall_time_s'] for r in rows])),'mean_effective_compute':float(np.mean([r['effective_compute'] for r in rows])),'peak_rss_kib':payload['peak_rss_kib'],'process_wall_time_s':payload['process_wall_time_s'],'fallback_count':int(sum(bool(r['fallback']) for r in rows)),'mean_internal_timing_s':mean_timings}
        if candidate!='baseline':
            ratios=np.array([r['adjusted_score']/base[r['index']]['adjusted_score'] for r in rows]);bvals=[base[r['index']]['adjusted_score'] for r in rows]
            item.update({'wins':int(np.sum(ratios<1)),'median_network_ratio':float(np.median(ratios)),'worst_network_ratio':float(np.max(ratios)),'mean_score_ratio':float(np.mean(adjusted)/np.mean(bvals)),'network_bootstrap_interval':bootstrap_ratio(adjusted,bvals)})
        summary[candidate]=item
        for r in rows:
            flat={k:v for k,v in r.items() if k not in ('timings','spec','kink_counts','stable_on_counts','stable_off_counts')};flat['timings_json']=json.dumps(r.get('timings',{}),sort_keys=True);flat['kink_counts_json']=json.dumps(r.get('kink_counts',[]));csv_rows.append(flat)
    (a.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=sorted({k for r in csv_rows for k in r})
    with (a.outdir/'per_network.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csv_rows)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
