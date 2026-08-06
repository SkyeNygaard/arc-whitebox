#!/usr/bin/env python3
"""Run one candidate in one clean official-style subprocess."""
from __future__ import annotations
import argparse, json, os, resource, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from compiler_core import SPECS, BUDGET, LAMBDA_FLOPS_PER_SECOND, adjusted_score, pilot_indices, run_candidate
from fast_matmul_generic import winograd_hybrid_p3_d5_partial_tree
from kerdock_design import first_layer_design, INV_SQRT_2PI


def load_rows(data_dir: Path, indices: list[int]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to load official Mini-100 parquet shards") from exc
    files=sorted(data_dir.glob('*.parquet'))
    if not files: raise FileNotFoundError(f'no parquet shards below {data_dir}')
    wanted=set(indices); rows=[]; offset=0
    for file in files:
        table=pq.read_table(file,columns=['mlp_name','weights','all_layer_means'])
        for local in range(len(table)):
            idx=offset+local
            if idx in wanted:
                rows.append((idx,table['mlp_name'][local].as_py(),np.asarray(table['weights'][local].as_py(),dtype=np.float32),np.asarray(table['all_layer_means'][local].as_py(),dtype=np.float64)))
        offset+=len(table)
    rows.sort(key=lambda x:x[0])
    if len(rows)!=len(wanted): raise IndexError(f'found {len(rows)} of {len(wanted)} requested rows')
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',choices=sorted(SPECS),required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--asset',type=Path,required=True)
    ap.add_argument('--indices',type=int,nargs='+',required=True)
    ap.add_argument('--baseline-means',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    try:
        import flopscope
        import flopscope.numpy as fnp
    except ImportError as exc:
        raise RuntimeError('flopscope==0.9.1 and whestbench challenge runtime are required') from exc

    asset=np.load(args.asset); rotation=np.asarray(asset['rotation'],dtype=np.float32); chirps=np.asarray(asset['chirps'],dtype=np.float32)
    baseline={}
    if args.baseline_means:
        z=np.load(args.baseline_means)
        baseline={int(k.split('_',1)[1]):z[k] for k in z.files}
    rows=[]; saved={}; process_start=time.perf_counter()
    for idx,name,weights,target in load_rows(args.data,args.indices):
        with flopscope.BudgetContext(flop_budget=BUDGET,quiet=True) as ctx:
            tracked_weights=[fnp.asarray(w).astype(fnp.float32) for w in weights]
            first=first_layer_design(tracked_weights[0],fnp.asarray(rotation),fnp.asarray(chirps),fnp)
            matmul=lambda a,b: winograd_hybrid_p3_d5_partial_tree(a,b,fnp)
            mean,diag=run_candidate(first,tracked_weights[1:],SPECS[args.candidate],matmul,fnp,pilot_rows=fnp.asarray(pilot_indices(cols=8)))
            first_mean=fnp.sqrt(fnp.sum(tracked_weights[0]*tracked_weights[0],axis=0))*INV_SQRT_2PI
            prediction=fnp.stack([first_mean]+[fnp.zeros(256) for _ in range(30)]+[mean],axis=0)
        summary=ctx.summary_dict(); pred=np.asarray(prediction)[-1].astype(np.float64)
        raw=float(np.mean((pred-target[-1])**2)); flops=int(summary['flops_used']); residual=float(summary['residual_wall_time_s']); effective=flops+LAMBDA_FLOPS_PER_SECOND*residual
        rec={'index':idx,'name':name,'candidate':args.candidate,'raw_final_mse':raw,'tracked_flops':flops,'residual_wall_time_s':residual,'effective_compute':effective,'adjusted_score':adjusted_score(raw,effective),**diag}
        if idx in baseline:
            delta=pred-baseline[idx]
            rec['approximation_mse_vs_baseline']=float(np.mean(delta*delta));rec['approximation_max_abs_vs_baseline']=float(np.max(np.abs(delta)))
        rows.append(rec);saved[f'network_{idx}']=pred
        print(json.dumps({'completed':idx,'candidate':args.candidate,'raw_mse':raw,'effective_compute':effective}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out.with_suffix('.means.npz'),**saved)
    payload={'candidate':args.candidate,'indices':args.indices,'process_wall_time_s':time.perf_counter()-process_start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rows':rows}
    args.out.write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__': main()
