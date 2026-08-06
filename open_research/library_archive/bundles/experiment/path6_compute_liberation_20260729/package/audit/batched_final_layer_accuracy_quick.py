import json, math, os, time, gc
import numpy as np
from compute_liberation_experiment import N_POINTS, WIDTH, partial_tree_matmul

def final_mean(h,w,chunk):
    total=np.zeros(WIDTH,dtype=np.float64)
    for s in range(0,N_POINTS,chunk):
        z=partial_tree_matmul(h[s:min(s+chunk,N_POINTS)],w)
        total += np.maximum(z,0).astype(np.float64).sum(axis=0)
    return total/N_POINTS
rows=[]
for seed in [17,101]:
    rng=np.random.default_rng(seed)
    h=np.maximum(rng.normal(0.15,0.7,size=(N_POINTS,WIDTH)),0).astype(np.float32)
    w=(rng.standard_normal((WIDTH,WIDTH),dtype=np.float32)/math.sqrt(WIDTH)).astype(np.float32)
    t=time.perf_counter(); ref=final_mean(h,w,N_POINTS); ref_s=time.perf_counter()-t
    for c in [512,2048]:
        gc.collect(); t=time.perf_counter(); got=final_mean(h,w,c); elapsed=time.perf_counter()-t
        d=got-ref
        rows.append({'seed':seed,'chunk_rows':c,'calls':math.ceil(N_POINTS/c),'elapsed_s':elapsed,'ref_elapsed_s':ref_s,
                     'max_abs_mean_diff':float(np.max(np.abs(d))), 'rms_mean_diff':float(np.sqrt(np.mean(d*d))),
                     'checksum_diff':float(got.sum()-ref.sum())})
summary={}
for c in [512,2048]:
    sub=[r for r in rows if r['chunk_rows']==c]
    summary[str(c)]={'max_abs_mean_diff_max':max(r['max_abs_mean_diff'] for r in sub),
                     'rms_mean_diff_max':max(r['rms_mean_diff'] for r in sub),
                     'elapsed_s_median':float(np.median([r['elapsed_s'] for r in sub])),
                     'ref_elapsed_s_median':float(np.median([r['ref_elapsed_s'] for r in sub])),
                     'speedup_vs_ref_median':float(np.median([r['ref_elapsed_s']/r['elapsed_s'] for r in sub]))}
out={'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'rows':rows,'summary':summary}
json.dump(out,open('/mnt/data/paths/06_compute_liberation/batched_final_layer_accuracy_quick_threads1.json','w'),indent=2)
print(json.dumps(summary,indent=2))
