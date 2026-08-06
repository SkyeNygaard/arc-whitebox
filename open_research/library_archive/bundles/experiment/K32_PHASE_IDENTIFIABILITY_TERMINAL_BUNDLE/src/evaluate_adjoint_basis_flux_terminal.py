#!/usr/bin/env python3
from pathlib import Path
import glob,json,csv,hashlib
import numpy as np
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1];BP=ROOT/'basis_phase'
M=np.load(BP/'ADJOINT_BASIS_FLUX_FROZEN_MODEL.npz');template=M['template'];coef=float(M['linear_coef']);inter=float(M['intercept']);shrink=float(M['shrink'])
rows=[]
for p in sorted(glob.glob(str(BP/'terminal_data/test_network_*.npz'))):
 with np.load(p) as z:
  nid=int(z['network_id']);j=0;beta=z['beta_bar'][j].astype(float);corr=template@beta;truth1=z['truth_half1'][j].astype(float);truth2=z['truth_half2'][j].astype(float);truth=.5*(truth1+truth2);sample=z['sample_prediction'][j].astype(float);base=z['baseline_prediction'][j].astype(float);oracle=z['oracle_prediction'][j].astype(float)
 with np.load(BP/f'terminal_basis/{nid}_0.npz') as b:dx=b['dx'].astype(float);yb=b['yb'].astype(float)
 yc=yb-yb.mean(0);zz=dx@template;ww=yc@(corr/max(np.linalg.norm(corr),1e-30));zc=zz-zz.mean();wc=ww-ww.mean();stat=float(np.mean(zc*wc));pred_scale=shrink*(coef*stat+inter);pred=sample+pred_scale*corr
 def mse(x):return float(np.mean((x-truth)**2))
 def cross(x):return float(np.mean((x-truth1)*(x-truth2)))
 den=float(corr@corr);oracle_scale=float(corr@(truth-sample)/max(den,1e-30))
 bm=mse(base);cm=mse(pred);sm=mse(sample);om=mse(oracle);bc=cross(base);cc=cross(pred);sc=cross(sample);oc=cross(oracle)
 rows.append({'network_id':nid,'statistic':stat,'predicted_scale':pred_scale,'oracle_scale':oracle_scale,'base_mse':bm,'candidate_mse':cm,'sample_mse':sm,'oracle_mse':om,'candidate_ratio':cm/bm,'sample_ratio':sm/bm,'oracle_ratio':om/bm,'base_cross_mse':bc,'candidate_cross_mse':cc,'candidate_cross_ratio':cc/bc if bc>0 else None})
base=np.array([r['base_mse'] for r in rows]);cand=np.array([r['candidate_mse'] for r in rows]);sample=np.array([r['sample_mse'] for r in rows]);oracle=np.array([r['oracle_mse'] for r in rows]);rr=cand/base
bc=np.array([r['base_cross_mse'] for r in rows]);cc=np.array([r['candidate_cross_mse'] for r in rows])
rng=np.random.default_rng(20260729);idx=rng.integers(0,len(rows),size=(50000,len(rows)));bs=cand[idx].sum(1)/base[idx].sum(1);cost_factor=1+(2.248e9+38184)/175.52e9;abs_=bs*cost_factor
preds=np.array([r['predicted_scale'] for r in rows]);truths=np.array([r['oracle_scale'] for r in rows])
summary={'terminal_state':'PASS' if (np.quantile(abs_,.975)<1 and rr.max()<=1.10 and pearsonr(preds,truths).statistic>0) else 'FAIL','n':len(rows),'raw_ratio':float(cand.sum()/base.sum()),'raw_bootstrap95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'adjusted_cost_factor':float(cost_factor),'adjusted_ratio':float(cand.sum()/base.sum()*cost_factor),'adjusted_bootstrap95':[float(np.quantile(abs_,.025)),float(np.quantile(abs_,.975))],'wins':int(np.sum(rr<1)),'median_ratio':float(np.median(rr)),'worst_ratio':float(rr.max()),'p90_ratio':float(np.quantile(rr,.9)),'scale_pearson':float(pearsonr(preds,truths).statistic),'scale_sign_accuracy':float(np.mean((preds>0)==(truths>0))),'predicted_positive_fraction':float(np.mean(preds>0)),'sample_anchor_ratio':float(sample.sum()/base.sum()),'oracle_k32_ratio':float(oracle.sum()/base.sum()),'cross_mse_ratio':float(cc.sum()/bc.sum()),'model_sha256':hashlib.sha256((BP/'ADJOINT_BASIS_FLUX_FROZEN_MODEL.npz').read_bytes()).hexdigest(),'terminal_ids':[r['network_id'] for r in rows]}
(BP/'ADJOINT_BASIS_FLUX_TERMINAL_RESULTS.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
with (BP/'ADJOINT_BASIS_FLUX_TERMINAL_ROWS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
