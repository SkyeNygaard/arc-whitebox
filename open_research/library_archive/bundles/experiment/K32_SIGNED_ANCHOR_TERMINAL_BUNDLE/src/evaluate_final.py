#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import train_models as tm


def load_models(exp:Path, device):
    models=[]
    for p in sorted((exp/'models').glob('model_*.pt')):
        z=torch.load(p,map_location='cpu',weights_only=False)
        m=tm.ProbeSetModel(z['token_dim'],z['global_dim'],z['hidden']); m.load_state_dict(z['state_dict']); m.to(device); m.eval(); models.append(m)
    return models

def candidate_metrics(pred,data):
    y1=data['truth_half1']; y2=data['truth_half2']; truth=.5*(y1+y2)
    mse=np.mean((pred-truth)**2,axis=1); umse=np.mean((pred-y1)*(pred-y2),axis=1)
    base=np.asarray(data['base_mse']).reshape(-1); baseu=np.mean((data['baseline_prediction']-y1)*(data['baseline_prediction']-y2),axis=1)
    ratio=float(mse.sum()/base.sum()); ur=float(umse.sum()/baseu.sum())
    rr=mse/np.maximum(base,1e-300)
    return {'aggregate_ratio':ratio,'aggregate_unbiased_ratio':ur,'wins':int(np.sum(rr<1)),'examples':len(rr),'median':float(np.median(rr)),'p90':float(np.quantile(rr,.9)),'worst':float(rr.max()),'ratios':rr,'mse':mse,'unbiased_mse':umse}

def correction_diagnostics(pred_delta,data):
    true=data['target_delta']; x=pred_delta.reshape(-1); y=true.reshape(-1)
    corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.0
    sign=float(np.mean(np.sign(x)==np.sign(y)))
    pc=np.einsum('np,npd->nd',pred_delta,data['beta_bar']); tc=data['target_correction']
    den=np.maximum(np.linalg.norm(pc,axis=1)*np.linalg.norm(tc,axis=1),1e-30); cos=np.sum(pc*tc,axis=1)/den
    return {'anchor_pearson':corr,'anchor_sign_accuracy':sign,'mean_final_correction_cosine':float(np.mean(cos)),'median_final_correction_cosine':float(np.median(cos)),'negative_correction_cosines':int(np.sum(cos<0)),'worst_correction_cosine':float(np.min(cos))}

def rotation_table(metrics,data):
    out={}
    rr=metrics['ratios']
    for rot in sorted(set(data['rotation_seed'].tolist())):
        ix=np.flatnonzero(data['rotation_seed']==rot)
        out[str(rot)]={'n':len(ix),'aggregate_ratio':float(metrics['mse'][ix].sum()/np.asarray(data['base_mse']).reshape(-1)[ix].sum()),'wins':int(np.sum(rr[ix]<1)),'median':float(np.median(rr[ix])),'worst':float(rr[ix].max())}
    return out

def base_table(metrics,data):
    out=[]; base=np.asarray(data['base_mse']).reshape(-1)
    for nid in sorted(set(data['network_id'].tolist())):
        ix=np.flatnonzero(data['network_id']==nid); ratio=float(metrics['mse'][ix].sum()/base[ix].sum())
        out.append({'network_id':int(nid),'ratio':ratio,'wins':int(np.sum(metrics['ratios'][ix]<1)),'rotations':len(ix),'worst_rotation':float(metrics['ratios'][ix].max())})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); args=ap.parse_args(); exp=args.experiment
    cfg=json.loads((exp/'final_test_config.json').read_text()); tv=json.loads((exp/'results/training_validation.json').read_text())
    data=tm.load_split(exp/'final_test_data','test')
    with np.load(exp/'models/normalization.npz') as z: norm={k:z[k].copy() for k in z.files}
    g=((data['global_features']-norm['global_mean'])/norm['global_std']).astype(np.float32)
    t=((data['token_features']-norm['token_mean'])/norm['token_std']).astype(np.float32)
    std=(g,t,(data['target_delta']/float(norm['target_scale'])).astype(np.float32))
    device=torch.device('cpu'); torch.set_num_threads(min(5,os.cpu_count() or 1)); torch.set_num_interop_threads(1)
    models=load_models(exp,device); ens=tm.ensemble_predict(models,std,norm,data,device)
    pred_gated=tm.predict_candidate(ens['gated_delta'],data); pred_ungated=tm.predict_candidate(ens['mean_delta'],data)
    gated=candidate_metrics(pred_gated,data); ungated=candidate_metrics(pred_ungated,data)
    sample=candidate_metrics(data['sample_prediction'],data); oracle=candidate_metrics(data['oracle_prediction'],data); baseline=candidate_metrics(data['baseline_prediction'],data)
    members=[]
    for i,d in enumerate(ens['member_delta']):
        mm=candidate_metrics(tm.predict_candidate(d,data),data); members.append({'member':i,**{k:mm[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','median','worst']}})
    # Conservative arithmetic charge: direct radial control cost from the frozen M111 accounting, plus exact model inference count.
    baseline_flops=175.62e9; direct_control_flops=2.238e9
    token_dim=t.shape[-1]; global_dim=g.shape[-1]; h=cfg['model']['token_hidden']; p=t.shape[1]
    one_model_flops=2.0*(p*(token_dim*h+h*h+(3*h)*h+h*(h//2)+(h//2))+global_dim*h+h*h)
    model_flops=one_model_flops*len(models); added_flops=direct_control_flops+model_flops; multiplier=(baseline_flops+added_flops)/baseline_flops
    raw_ci=tm.bootstrap_base_ratio(pred_gated,data,reps=10000,seed=20260729); adjusted_ci=[x*multiplier for x in raw_ci]
    adjusted_ratio=gated['aggregate_ratio']*multiplier
    gate={
      'raw_candidate_over_baseline_le_0_595':gated['aggregate_ratio']<=.595,
      'preferred_le_0_537':gated['aggregate_ratio']<=.537,
      'adjusted_bootstrap_upper_below_1':adjusted_ci[1]<1,
      'worst_le_1_15':gated['worst']<=1.15,
      'screen_worst_le_1_25':gated['worst']<=1.25,
      'added_flops_below_14B':added_flops<14e9,
    }; gate['overall_pass']=all([gate['raw_candidate_over_baseline_le_0_595'],gate['adjusted_bootstrap_upper_below_1'],gate['worst_le_1_15'],gate['added_flops_below_14B']])
    result={
      'terminal_state':'PASS' if gate['overall_pass'] else 'FAIL',
      'scope':'Frozen three-member probe-set equivariant learner for K32 lower-order direct-output anchors; no conclusion about all learned or analytic anchor estimators.',
      'final_test_freeze_sha256':cfg['freeze_sha256'],'model_hashes':tv['model_hashes'],'base_networks':len(set(data['network_id'].tolist())),'examples':len(data['network_id']),'rotations':sorted(set(data['rotation_seed'].tolist())),
      'candidate':{**{k:gated[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','examples','median','p90','worst']},'raw_bootstrap_95':raw_ci,'compute_multiplier':multiplier,'added_flops_estimate':added_flops,'adjusted_ratio':adjusted_ratio,'adjusted_bootstrap_95':adjusted_ci,'mean_scale':float(ens['scale'].mean()),'median_scale':float(np.median(ens['scale'])),'abstentions':int(np.sum(ens['agreement']==0)),'min_member_cosine':float(np.min(ens['min_member_cosine']))},
      'diagnostics':{
        'oracle':{k:oracle[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','median','worst']},
        'sample_anchor_null':{k:sample[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','median','worst']},
        'baseline_check':{k:baseline[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','median','worst']},
        'ungated_ensemble':{k:ungated[k] for k in ['aggregate_ratio','aggregate_unbiased_ratio','wins','median','worst']},
        'members':members,
        'gated_anchor':correction_diagnostics(ens['gated_delta'],data),
        'ungated_anchor':correction_diagnostics(ens['mean_delta'],data),
      },
      'by_rotation':rotation_table(gated,data),'by_base_network':base_table(gated,data),'gate':gate,
      'integrity':{'excluded_exposed_test_ids':list(range(3080,3096)),'terminal_test_ids':sorted(set(data['network_id'].tolist())),'no_terminal_test_tuning':True,'normalization_sha256':cfg['final_test_freeze']['normalization_sha256'],'training_validation_sha256':cfg['final_test_freeze']['training_validation_sha256']},
    }
    (exp/'results/final_test_results.json').write_text(json.dumps(result,indent=2))
    rows=[]
    for i in range(len(data['network_id'])):
        rows.append({'network_id':int(data['network_id'][i]),'rotation_seed':int(data['rotation_seed'][i]),'candidate_ratio':float(gated['ratios'][i]),'oracle_ratio':float(data['oracle_ratio'][i]),'scale':float(ens['scale'][i]),'agreement':int(ens['agreement'][i]),'min_member_cosine':float(ens['min_member_cosine'][i])})
    import csv
    with (exp/'results/final_test_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
