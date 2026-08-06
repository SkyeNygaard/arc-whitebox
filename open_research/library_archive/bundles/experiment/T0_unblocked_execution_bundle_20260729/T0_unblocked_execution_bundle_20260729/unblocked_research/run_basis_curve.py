#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, sys, time, types
from pathlib import Path
import numpy as np

BASIS_COUNTS=(32,64,96,129)
ROWS_PER_BASIS=512
BUDGET=272_000_000_000.0
BASELINE_EFFECTIVE=175_500_000_000.0
SHAPE_FLOPS={32:42332097301.265236,64:84570628129.94832,96:126809158958.63141,129:170382691584.0}
# Preserve the archived baseline residual-wall charge when estimating adjusted economics.
BASELINE_TRACKED=170_906_815_488.0
BASELINE_RESIDUAL_S=(BASELINE_EFFECTIVE-BASELINE_TRACKED)/1e11

class BaseEstimator: pass
class SetupContext:
    def __init__(self,d): self.submission_dir=d
class MLP:
    def __init__(self,w): self.width=256; self.depth=32; self.weights=w

def install_shims():
    fl=types.ModuleType('flopscope'); fl.numpy=np
    sys.modules['flopscope']=fl; sys.modules['flopscope.numpy']=np
    wh=types.ModuleType('whestbench'); wh.BaseEstimator=BaseEstimator; wh.SetupContext=SetupContext
    sys.modules['whestbench']=wh
    dom=types.ModuleType('whestbench.domain'); dom.MLP=MLP; sys.modules['whestbench.domain']=dom

def load_module(pkg:Path, name='a43_estimator'):
    install_shims(); sys.path.insert(0,str(pkg)); sys.modules.pop('fast_matmul',None)
    spec=importlib.util.spec_from_file_location(name,pkg/'estimator.py')
    m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)
    return m

def cumulative_predict(est, module, weights):
    fm=sys.modules['fast_matmul']
    first_weight=weights[0].astype(np.float32)
    prepared=tuple(fm.prepare_right_p3_d5(w.astype(np.float32)) for w in weights[1:])
    first=est._first_layer_design(first_weight)
    total=None; out={}; timings={}; t0=time.perf_counter()
    for block,start in enumerate(range(0,129*512,512), start=1):
        encoded=fm.first_layer_chunk_to_relu_encoding(first[start:start+512],prepared[0])
        for p in prepared[1:-1]: encoded=fm.encoded_chunk_to_relu_encoding(encoded,p)
        chunk=fm.encoded_chunk_to_final_sum(encoded,prepared[-1])
        total=chunk if total is None else total+chunk
        if block in BASIS_COUNTS:
            out[block]=np.asarray(total/(block*512),dtype=np.float64).copy()
            timings[block]=time.perf_counter()-t0
    return out,timings

def package_predict(pkg:Path, weights):
    m=load_module(pkg,'pkg_'+pkg.name)
    e=m.Estimator(); e.setup(SetupContext(pkg))
    return np.asarray(e.predict(MLP(weights),272_000_000_000)[-1],dtype=np.float64)

def grouped_bootstrap(base_ids, base_mse, cand_mse, adjusted_mult, seed=2026072900, reps=20000):
    groups=np.unique(base_ids); rng=np.random.default_rng(seed)
    # aggregate ratio of mean MSE, matching official pooled metric.
    raw=[]; adj=[]
    for _ in range(reps):
        g=rng.choice(groups,size=len(groups),replace=True)
        ix=np.concatenate([np.flatnonzero(base_ids==x) for x in g])
        r=base_mse[ix].mean()/cand_mse[ix].mean()
        raw.append(r); adj.append(r/adjusted_mult)
    q=lambda x:[float(v) for v in np.quantile(x,[.025,.5,.975])]
    return q(raw),q(adj)

def summarize(rows, noise_floor=None):
    base_ids=np.asarray([r['base_network_id'] for r in rows])
    base=np.asarray([r['baseline_mse'] for r in rows],float)
    out={}
    for k in BASIS_COUNTS:
        cand=np.asarray([r[f'mse_{k}'] for r in rows],float)
        ratio_each=cand/base
        # Candidate effective compute: shape-traced tracked cost + same residual wall charge.
        eff=SHAPE_FLOPS[k]+1e11*BASELINE_RESIDUAL_S
        mult=eff/BASELINE_EFFECTIVE
        raw_gain=float(base.mean()/cand.mean())
        score_gain=float(raw_gain/mult)
        raw_ci,adj_ci=grouped_bootstrap(base_ids,base,cand,mult,seed=2026072900+k)
        # Base-network wins: average rotations before comparing.
        group_ratios=[]
        for g in np.unique(base_ids):
            ix=base_ids==g; group_ratios.append(float(cand[ix].mean()/base[ix].mean()))
        rec={
            'basis_count':k,'examples':len(rows),'base_networks':len(np.unique(base_ids)),
            'baseline_raw_mse':float(base.mean()),'candidate_raw_mse':float(cand.mean()),
            'raw_gain_baseline_over_candidate':raw_gain,
            'raw_gain_group_bootstrap_ci95':raw_ci,
            'candidate_over_baseline':float(cand.mean()/base.mean()),
            'wins_base_networks':int(np.sum(np.asarray(group_ratios)<1)),
            'median_base_network_candidate_over_baseline':float(np.median(group_ratios)),
            'p90_base_network_candidate_over_baseline':float(np.quantile(group_ratios,.9)),
            'worst_base_network_candidate_over_baseline':float(np.max(group_ratios)),
            'tracked_flops_shape_scaled':SHAPE_FLOPS[k],
            'assumed_residual_wall_s':BASELINE_RESIDUAL_S,
            'effective_compute':eff,'effective_compute_ratio_vs_129_archived_baseline':mult,
            'adjusted_gain_baseline_over_candidate':score_gain,
            'adjusted_gain_group_bootstrap_ci95':adj_ci,
            'adjusted_candidate_over_baseline':float(mult/raw_gain),
        }
        if noise_floor is not None:
            b=max(base.mean()-noise_floor,1e-30); c=max(cand.mean()-noise_floor,1e-30)
            rec['noise_floor_mean']=noise_floor
            rec['noise_corrected_raw_gain']=float(b/c)
            rec['noise_corrected_adjusted_gain']=float((b/c)/mult)
        out[str(k)]=rec
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',required=True); ap.add_argument('--package-root',required=True)
    ap.add_argument('--outdir',required=True); ap.add_argument('--split',default='test',choices=['test','validation','all'])
    ap.add_argument('--limit',type=int); ap.add_argument('--resume',action='store_true'); ap.add_argument('--validate-only',action='store_true')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    pkgroot=Path(args.package_root); pkg=pkgroot/'A43'
    labels=np.load(args.labels,allow_pickle=False)
    weights=labels['weights']; errors=labels['baseline_error'].astype(np.float64); base_ids=labels['base_network_id']; rotations=labels['rotation_id']
    # Canonical split assignment from frozen archive: first 34 train, next 8 calibration, next 8 validation, final 32 test.
    if args.split=='test': indices=np.arange(50,82)
    elif args.split=='validation': indices=np.arange(42,50)
    else: indices=np.arange(82)
    if args.limit: indices=indices[:args.limit]
    m=load_module(pkg); e=m.Estimator(); e.setup(SetupContext(pkg))
    # One-row exact implementation validation.
    vi=int(indices[0]); w=[x.astype(np.float32) for x in weights[vi]]
    cum,timing=cumulative_predict(e,m,w)
    validation={}
    for k in BASIS_COUNTS:
        direct=package_predict(pkgroot/f'A43_basis{k:03d}',w)
        diff=cum[k]-direct
        validation[str(k)]={'max_abs':float(np.max(np.abs(diff))),'rms':float(np.sqrt(np.mean(diff*diff))),
                            'exact_equal':bool(np.array_equal(cum[k],direct)),
                            'cum_sha256':hashlib.sha256(np.ascontiguousarray(cum[k]).view(np.uint8)).hexdigest(),
                            'direct_sha256':hashlib.sha256(np.ascontiguousarray(direct).view(np.uint8)).hexdigest()}
    (outdir/'IMPLEMENTATION_VALIDATION.json').write_text(json.dumps(validation,indent=2))
    print('VALIDATION',json.dumps(validation),flush=True)
    if args.validate_only:return
    rows_path=outdir/f'ROWS_{args.split}.jsonl'; done={}
    if args.resume and rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r=json.loads(line); done[int(r['index'])]=r
    for n,idx in enumerate(indices,1):
        idx=int(idx)
        if idx in done: print(f'SKIP {idx}',flush=True); continue
        w=[x.astype(np.float32) for x in weights[idx]]
        t=time.perf_counter(); preds,times=cumulative_predict(e,m,w); elapsed=time.perf_counter()-t
        e129=errors[idx]
        rec={'index':idx,'base_network_id':int(base_ids[idx]),'rotation_id':int(rotations[idx]),
             'baseline_mse':float(np.mean(e129*e129)),'elapsed_s':elapsed}
        for k in BASIS_COUNTS:
            delta=preds[k]-preds[129]
            ek=e129+delta
            rec[f'mse_{k}']=float(np.mean(ek*ek)); rec[f'delta_rms_{k}']=float(np.sqrt(np.mean(delta*delta)))
            rec[f'cumulative_s_{k}']=float(times[k])
        with rows_path.open('a') as f:f.write(json.dumps(rec)+'\n')
        print(f"DONE {n}/{len(indices)} idx={idx} base={rec['base_network_id']} elapsed={elapsed:.3f} ratios="+
              ','.join(f"{k}:{rec[f'mse_{k}']/rec['baseline_mse']:.4f}" for k in BASIS_COUNTS),flush=True)
    rows=[json.loads(x) for x in rows_path.read_text().splitlines()]
    summary=summarize(rows,noise_floor=2.1885e-8 if args.split=='test' else None)
    payload={'status':'completed','evidence_class':'full-width synthetic paired replay; frozen held-out test block already exposed by Prompt7',
             'split':args.split,'indices':[int(x) for x in indices],
             'weight_storage':'float16 frozen weights converted to float32 for propagation',
             'baseline_error_contract':'stored direct 256-vector; candidate_error = baseline_error + prediction_k - prediction_129',
             'implementation_validation':validation,'cost_assumptions':{'budget':BUDGET,'baseline_effective':BASELINE_EFFECTIVE,
             'baseline_tracked':BASELINE_TRACKED,'baseline_residual_s':BASELINE_RESIDUAL_S,'shape_flops':SHAPE_FLOPS},
             'arms':summary}
    (outdir/f'RESULTS_{args.split}.json').write_text(json.dumps(payload,indent=2))
    # CSV
    keys=sorted({k for r in rows for k in r})
    with (outdir/f'ROWS_{args.split}.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
