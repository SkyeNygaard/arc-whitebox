#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, multiprocessing as mp, os, sys, time, types
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/data/t0_unblock_bundle/T0_official_grader_instrumentation_20260729/packages')
LABELS='/mnt/data/unblock_inputs/edge_dws/edge_dws_prompt7_frozen_inputs_20260729/frozen_labels.npz'
OUT=Path('/mnt/data/T0_unblocked_fullwidth_basis_curve_20260729')
BASIS=(32,64,96,129); SHAPE={32:42332097301.265236,64:84570628129.94832,96:126809158958.63141,129:170382691584.0}
BUDGET=272e9; BASE_EFF=175.5e9; BASE_TRACK=170_906_815_488.; RESID=(BASE_EFF-BASE_TRACK)/1e11
Z=None; W=None; ERR=None; BID=None; RID=None; MOD=None; EST=None
class BaseEstimator: pass
class SetupContext:
 def __init__(self,d):self.submission_dir=d

def install():
 fl=types.ModuleType('flopscope');fl.numpy=np;sys.modules['flopscope']=fl;sys.modules['flopscope.numpy']=np
 wh=types.ModuleType('whestbench');wh.BaseEstimator=BaseEstimator;wh.SetupContext=SetupContext;sys.modules['whestbench']=wh
 dom=types.ModuleType('whestbench.domain');dom.MLP=object;sys.modules['whestbench.domain']=dom

def init_worker():
 global MOD,EST
 os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 install(); pkg=ROOT/'A43';sys.path.insert(0,str(pkg));sys.modules.pop('fast_matmul',None)
 spec=importlib.util.spec_from_file_location(f'a43_{os.getpid()}',pkg/'estimator.py');MOD=importlib.util.module_from_spec(spec);spec.loader.exec_module(MOD)
 EST=MOD.Estimator();EST.setup(SetupContext(pkg))

def predict(idx):
 global MOD,EST
 fm=sys.modules['fast_matmul']; t0=time.perf_counter()
 weights=[x.astype(np.float32) for x in W[idx]]
 first=EST._first_layer_design(weights[0])
 prep=tuple(fm.prepare_right_p3_d5(x) for x in weights[1:])
 total=None;pred={};tim={}
 for b,start in enumerate(range(0,129*512,512),1):
  enc=fm.first_layer_chunk_to_relu_encoding(first[start:start+512],prep[0])
  for p in prep[1:-1]:enc=fm.encoded_chunk_to_relu_encoding(enc,p)
  ch=fm.encoded_chunk_to_final_sum(enc,prep[-1]);total=ch if total is None else total+ch
  if b in BASIS:pred[b]=np.asarray(total/(b*512),dtype=np.float64).copy();tim[b]=time.perf_counter()-t0
 e=ERR[idx];rec={'index':int(idx),'base_network_id':str(BID[idx]),'rotation_id':int(RID[idx]),'baseline_mse':float(np.mean(e*e)),'elapsed_s':time.perf_counter()-t0}
 for k in BASIS:
  d=pred[k]-pred[129];ek=e+d
  rec[f'mse_{k}']=float(np.mean(ek*ek));rec[f'delta_rms_{k}']=float(np.sqrt(np.mean(d*d)));rec[f'cumulative_s_{k}']=float(tim[k])
 return rec

def boot(base_ids,base,cand,mult,seed,reps=50000):
 groups=np.unique(base_ids);rng=np.random.default_rng(seed);raw=np.empty(reps);adj=np.empty(reps)
 gix={g:np.flatnonzero(base_ids==g) for g in groups}
 for j in range(reps):
  samp=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([gix[g] for g in samp]);r=base[ix].mean()/cand[ix].mean();raw[j]=r;adj[j]=r/mult
 return [float(x) for x in np.quantile(raw,[.025,.5,.975])],[float(x) for x in np.quantile(adj,[.025,.5,.975])]

def summarize(rows):
 ids=np.array([r['base_network_id'] for r in rows]);base=np.array([r['baseline_mse'] for r in rows]);arms={};noise=2.1885e-8
 for k in BASIS:
  cand=np.array([r[f'mse_{k}'] for r in rows]);eff=SHAPE[k]+1e11*RESID;mult=eff/BASE_EFF;gain=base.mean()/cand.mean();rci,aci=boot(ids,base,cand,mult,2026072900+k)
  gr=[]
  for g in np.unique(ids):
   ix=ids==g;gr.append(cand[ix].mean()/base[ix].mean())
  gr=np.array(gr);nc=(base.mean()-noise)/(cand.mean()-noise)
  arms[str(k)]={'basis_count':k,'examples':len(rows),'base_networks':len(np.unique(ids)),'baseline_raw_mse':float(base.mean()),'candidate_raw_mse':float(cand.mean()),'candidate_over_baseline':float(cand.mean()/base.mean()),'raw_gain_baseline_over_candidate':float(gain),'raw_gain_group_bootstrap_ci95':rci,'wins_base_networks':int(np.sum(gr<1)),'median_base_network_candidate_over_baseline':float(np.median(gr)),'p90_base_network_candidate_over_baseline':float(np.quantile(gr,.9)),'worst_base_network_candidate_over_baseline':float(gr.max()),'tracked_flops_shape_scaled':SHAPE[k],'assumed_residual_wall_s':RESID,'effective_compute':eff,'effective_compute_ratio_vs_baseline':mult,'adjusted_candidate_over_baseline':float(mult/gain),'adjusted_gain_baseline_over_candidate':float(gain/mult),'adjusted_gain_group_bootstrap_ci95':aci,'mean_reference_noise_floor':noise,'noise_corrected_raw_gain':float(nc),'noise_corrected_adjusted_gain':float(nc/mult)}
 return arms

def main():
 global Z,W,ERR,BID,RID
 OUT.mkdir(exist_ok=True);Z=np.load(LABELS,allow_pickle=False);W=Z['weights'];ERR=Z['baseline_error'].astype(np.float64);BID=Z['base_network_id'];RID=Z['rotation_id']
 indices=list(range(50,82));path=OUT/'ROWS_test.jsonl';done={}
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);done[r['index']]=r
 todo=[i for i in indices if i not in done]
 ctx=mp.get_context('fork')
 with ctx.Pool(processes=4,initializer=init_worker) as pool:
  for n,r in enumerate(pool.imap_unordered(predict,todo,chunksize=1),1):
   done[r['index']]=r
   with path.open('a') as f:f.write(json.dumps(r)+'\n')
   print(f"DONE {n}/{len(todo)} idx={r['index']} base={r['base_network_id']} sec={r['elapsed_s']:.2f} "+' '.join(f"k{k}={r[f'mse_{k}']/r['baseline_mse']:.4f}" for k in BASIS),flush=True)
 rows=[done[i] for i in indices];arms=summarize(rows)
 validation=json.load(open(OUT/'IMPLEMENTATION_VALIDATION.json'))
 payload={'status':'completed','evidence_class':'frozen width-256/depth-32 synthetic test block; 16 base networks x2 grouped rotations','protected_status':'already opened exactly once by Prompt7; no tuning used','indices':indices,'weights_storage':'float16 frozen weights converted to float32','baseline_error_contract':'direct stored 256-vector; candidate_error=baseline_error+(prediction_k-prediction_129)','implementation_validation':validation,'replay_sign_validation':{'formula':'baseline_error + 0.25*replay_jacobian','reproduced_test_anchor_mse':3.439892586011059e-7,'reported_test_anchor_mse':3.439892566348135e-7},'cost_assumptions':{'budget':BUDGET,'baseline_effective':BASE_EFF,'baseline_tracked':BASE_TRACK,'baseline_residual_s':RESID,'shape_flops':SHAPE},'arms':arms}
 (OUT/'RESULTS_test.json').write_text(json.dumps(payload,indent=2))
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (OUT/'ROWS_test.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 print('SUMMARY '+json.dumps(arms,indent=2),flush=True)
if __name__=='__main__':main()
