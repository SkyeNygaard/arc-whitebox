from __future__ import annotations
import csv, gc, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import torch

from src.baselines import fit_anchor_shrink, ridge_fit, ridge_predict
from src.contracts import load_bundle, sha256_file
from src.cost import estimate_dws_flops, effective_compute_b
from src.edge_dws_orig import EdgeStateDWS, permute_hidden_layers
from src.metrics import evaluate, replay_error, mse_rows, aggregate_by_base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'run_001'
DATA=ROOT/'inputs'/'frozen_labels.npz'
MAN=ROOT/'inputs'/'frozen_label_manifest.json'
SPL=ROOT/'inputs'/'canonical_split_registry.json'
CFG=ROOT/'frozen_config.json'
STATE=OUT/'train_state.pt'
FRESH=Path('/mnt/data/edge_dws_run/prompt7_fresh64/bases')

def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def predict(model,a,idx):
    model.eval(); corr=[]; scale=[]; conf=[]; times=[]
    with torch.no_grad():
        for i in idx:
            w=torch.from_numpy(a['weights'][i:i+1].astype(np.float32,copy=False))
            no=torch.from_numpy(a['node_observables'][i:i+1].astype(np.float32,copy=False))
            lo=torch.from_numpy(a['layer_observables'][i:i+1].astype(np.float32,copy=False))
            t=time.perf_counter(); o=model(w,no,lo); times.append(time.perf_counter()-t)
            corr.append(o.correction.numpy()[0]);scale.append(float(o.scale[0]));conf.append(float(o.confidence[0]))
            del w,no,lo,o; gc.collect()
    return np.asarray(corr),np.asarray(scale),np.asarray(conf),np.asarray(times)

def bootstrap(base,cand,cost_mult,reps=100000,seed=2026072919):
    rng=np.random.default_rng(seed); n=len(base); raw=np.empty(reps); at=0
    while at<reps:
        k=min(10000,reps-at); ix=rng.integers(0,n,size=(k,n)); raw[at:at+k]=base[ix].mean(1)/np.maximum(cand[ix].mean(1),1e-300);at+=k
    adj=raw/cost_mult
    return {'raw_gain_ci95':[float(x) for x in np.quantile(raw,[.025,.5,.975])],
            'adjusted_gain_ci95':[float(x) for x in np.quantile(adj,[.025,.5,.975])],
            'probability_adjusted_gain_gt_1':float(np.mean(adj>1))}

def noise_corrected_test(a,idx,coeff,candidate_compute,baseline_compute):
    err=replay_error(a['baseline_error'][idx],a['replay_jacobian'][idx],coeff)
    base=mse_rows(a['baseline_error'][idx]); cand=mse_rows(err); ids=np.asarray(a['base_network_id'][idx])
    u,bb=aggregate_by_base(base,ids);_,cc=aggregate_by_base(cand,ids)
    floors=[]
    for x in u:
        seed=int(str(x).split('-')[-1]); d=json.loads((FRESH/f'base_{seed}.json').read_text()); floors.append(float(d['reference_noise_floor']))
    floors=np.asarray(floors); b=np.maximum(bb-floors,1e-300);c=np.maximum(cc-floors,1e-300)
    mult=candidate_compute/baseline_compute; boot=bootstrap(b,c,mult)
    return {'base_networks':len(u),'baseline_mean_noise_corrected_mse':float(b.mean()),'candidate_mean_noise_corrected_mse':float(c.mean()),
            'raw_gain_noise_corrected':float(b.mean()/c.mean()),'adjusted_gain_noise_corrected':float(b.mean()/c.mean()/mult),
            'wins_noise_corrected':int(np.sum(c<b)),'median_candidate_over_baseline_noise_corrected':float(np.median(c/b)),
            'worst_candidate_over_baseline_noise_corrected':float(np.max(c/b)),**boot,
            'mean_reference_noise_floor':float(floors.mean()),'mean_reference_noise_fraction_of_observed_baseline':float(np.mean(floors/bb))}

def full_equivariance(model,a,i,seed=9917):
    rng=np.random.default_rng(seed); width=256; depth=32
    perms=[torch.arange(width)]
    for _ in range(1,depth): perms.append(torch.from_numpy(rng.permutation(width).astype(np.int64)))
    perms.append(torch.arange(width))
    w=torch.from_numpy(a['weights'][i:i+1].astype(np.float32,copy=False)); no=torch.from_numpy(a['node_observables'][i:i+1].astype(np.float32,copy=False));lo=torch.from_numpy(a['layer_observables'][i:i+1].astype(np.float32,copy=False))
    wp,nop=permute_hidden_layers(w,perms,no)
    model.eval()
    with torch.no_grad():
        x=model(w,no,lo); y=model(wp,nop,lo)
    return {'max_correction_abs_diff':float((x.correction-y.correction).abs().max()),'max_direction_abs_diff':float((x.direction-y.direction).abs().max()),
            'scale_abs_diff':float((x.scale-y.scale).abs().max()),'confidence_abs_diff':float((x.confidence-y.confidence).abs().max()),
            'pass_at_1e_5':bool((x.correction-y.correction).abs().max()<1e-5 and (x.scale-y.scale).abs().max()<1e-5 and (x.confidence-y.confidence).abs().max()<1e-5)}

def fast_invariant_features(weights, layer_observables=None):
    n,depth,width,_=weights.shape
    out=np.empty((n,depth*(17 if layer_observables is not None else 13)),dtype=np.float64)
    for i in range(n):
        w=weights[i].astype(np.float32,copy=False)
        mu=w.mean(axis=(1,2)); sd=w.std(axis=(1,2)); ab=np.abs(w).mean(axis=(1,2))
        z=(w-mu[:,None,None])/np.maximum(sd[:,None,None],1e-12)
        skew=(z*z*z).mean(axis=(1,2)); kurt=(z*z*z*z).mean(axis=(1,2))-3.0
        rn=np.sqrt((w*w).sum(axis=2)); cn=np.sqrt((w*w).sum(axis=1))
        parts=[mu,sd,ab,skew,kurt,rn.mean(1),rn.std(1),rn.min(1),rn.max(1),cn.mean(1),cn.std(1),cn.min(1),cn.max(1)]
        mat=np.stack(parts,axis=1)
        if layer_observables is not None:
            o=layer_observables[i].astype(np.float64,copy=False)
            mat=np.concatenate([mat,np.stack([o.mean(1),o.std(1),o.min(1),o.max(1)],axis=1)],axis=1)
        out[i]=mat.reshape(-1)
    return out

def main():
    print('stage:load',flush=True); cfg=json.loads(CFG.read_text()); b=load_bundle(DATA,MAN,SPL);a=b.arrays
    st=torch.load(STATE,map_location='cpu',weights_only=False)
    if st['epoch']<cfg['training']['epochs']: raise RuntimeError('training incomplete')
    m=EdgeStateDWS(depth=32,label_dim=1,node_obs_dim=a['node_observables'].shape[-1],layer_obs_dim=a['layer_observables'].shape[-1],edge_channels=8,node_channels=8,token_channels=48,passes=1,transformer_heads=4)
    m.load_state_dict(st['best_state']);m.eval()
    train,cal,val,test=(b.splits[k] for k in ('train','calibration','validation','test'))
    print('stage:predict_cal',flush=True); pred_cal,scale_cal,conf_cal,_=predict(m,a,cal); print('stage:predict_cal_done',flush=True)
    grid=np.linspace(0,1.5,301); losses=[]
    for alpha in grid:
        coeff=a['anchor_coeffs'][cal]+alpha*pred_cal; e=replay_error(a['baseline_error'][cal],a['replay_jacobian'][cal],coeff);losses.append(float(np.mean(e*e)))
    best_alpha=float(grid[int(np.argmin(losses))])
    alpha_anchor=fit_anchor_shrink(a['anchor_coeffs'][cal],a['baseline_error'][cal],a['replay_jacobian'][cal])
    print('stage:invariant_features',flush=True); x_all=fast_invariant_features(a['weights'],a.get('layer_observables')); print('stage:invariant_features_done',flush=True);y=a['target_coeffs']-a['anchor_coeffs'];rmods=[]
    for r in cfg['ridge_grid']:
        rm=ridge_fit(x_all[train],y[train],r); pc=ridge_predict(rm,x_all[cal]);e=replay_error(a['baseline_error'][cal],a['replay_jacobian'][cal],a['anchor_coeffs'][cal]+pc);rmods.append((float(np.mean(e*e)),float(r),rm))
    ridge_loss,ridge_selected,ridge_model=min(rmods,key=lambda x:x[0])
    flops=estimate_dws_flops(32,256,8,8,48,1,1);base_compute=float(cfg['cost']['baseline_effective_compute_B']);cand_compute=effective_compute_b(base_compute,float(cfg['cost']['anchor_extra_compute_B']),flops,float(cfg['cost'].get('replay_extra_compute_B',0)))
    report={'status':'completed','decision':None,'seed':cfg['seed'],'device':'cpu','dataset_sha256':sha256_file(DATA),'split_registry_sha256':sha256_file(SPL),
            'split_examples':{k:int(len(v)) for k,v in b.splits.items()},'split_base_networks':{k:int(len(set(map(str,a['base_network_id'][v])))) for k,v in b.splits.items()},
            'params':int(sum(p.numel() for p in m.parameters())),'inference_flops':int(flops),'inference_effective_compute_B':flops/1e9,'candidate_effective_compute_B':cand_compute,
            'candidate_compute_multiplier':cand_compute/base_compute,'best_epoch':int(np.argmin([h['calibration_raw_mse'] for h in st['history']])+1),
            'best_calibration_raw_mse':float(st['best']),'calibrated_model_residual_alpha':best_alpha,'constant_anchor_shrink_alpha':float(alpha_anchor),'ridge_selected':ridge_selected,
            'training_history':st['history'],'splits':{}}
    predictions={}
    print('stage:evaluate_splits',flush=True)
    for name,idx in [('validation',val),('test',test)]:
        print('stage:predict_'+name,flush=True)
        pc,sc,cf,times=predict(m,a,idx); predictions[name]=(pc,sc,cf)
        coefs={'anchor_only':a['anchor_coeffs'][idx], 'constant_shrinkage':alpha_anchor*a['anchor_coeffs'][idx],
               'invariant_ridge':a['anchor_coeffs'][idx]+ridge_predict(ridge_model,x_all[idx]),
               'edge_dws_uncalibrated':a['anchor_coeffs'][idx]+pc,
               'edge_dws':a['anchor_coeffs'][idx]+best_alpha*pc}
        rr={}
        for nm,c in coefs.items():
            comp=base_compute+cfg['cost']['anchor_extra_compute_B'] if nm not in ('edge_dws','edge_dws_uncalibrated') else cand_compute
            conf=cf if nm.startswith('edge_dws') else np.ones(len(idx));rr[nm]=evaluate(a,idx,c,conf,base_compute,comp)
        rr['prediction_scale_mean']=float(sc.mean());rr['prediction_scale_std']=float(sc.std());rr['prediction_confidence_mean']=float(cf.mean());rr['prediction_confidence_std']=float(cf.std())
        rr['inference_wall_seconds_total']=float(times.sum());rr['inference_wall_seconds_mean']=float(times.mean());rr['inference_wall_seconds_median']=float(np.median(times));rr['inference_wall_seconds_p95']=float(np.quantile(times,.95))
        if name=='test': rr['edge_dws_noise_corrected']=noise_corrected_test(a,idx,coefs['edge_dws'],cand_compute,base_compute)
        report['splits'][name]=rr
    print('stage:equivariance',flush=True); report['equivariance_full_width']=full_equivariance(m,a,int(val[0])); print('stage:equivariance_done',flush=True)
    edge=report['splits']['test']['edge_dws']; nc=report['splits']['test']['edge_dws_noise_corrected']
    report['gate_observed']={'raw_gain_ge_1_15':edge['raw_gain_baseline_over_candidate']>=1.15,'adjusted_ci_excludes_no_gain':edge['adjusted_gain_group_bootstrap_ci95'][0]>1,
                             'worst_candidate_over_baseline_le_1_10':edge['worst_candidate_over_baseline']<=1.10,'inference_cost_repaid':edge['adjusted_gain_baseline_over_candidate']>1}
    report['gate_observed']['pass']=all(report['gate_observed'].values())
    report['gate_noise_corrected']={'raw_gain_ge_1_15':nc['raw_gain_noise_corrected']>=1.15,'adjusted_ci_excludes_no_gain':nc['adjusted_gain_ci95'][0]>1,
                                    'worst_observed_candidate_over_baseline_le_1_10':edge['worst_candidate_over_baseline']<=1.10,'inference_cost_repaid':nc['adjusted_gain_noise_corrected']>1}
    report['gate_noise_corrected']['pass']=all(report['gate_noise_corrected'].values())
    report['decision']='PASS' if report['gate_noise_corrected']['pass'] else 'PAUSE_TESTED_MODEL_CLASS'
    print('stage:save',flush=True); OUT.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':st['best_state'],'config':cfg,'alpha':best_alpha,'best_epoch':report['best_epoch']},OUT/'model.pt')
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    # Base-level CSV for the untouched test.
    idx=test;pc,sc,cf=predictions['test']; coeff=a['anchor_coeffs'][idx]+best_alpha*pc; err=replay_error(a['baseline_error'][idx],a['replay_jacobian'][idx],coeff);base=mse_rows(a['baseline_error'][idx]);cand=mse_rows(err)
    ids=np.asarray(a['base_network_id'][idx]); rows=[]
    for uid in sorted(set(map(str,ids))):
        mask=np.asarray([str(x)==uid for x in ids]); rows.append({'base_network_id':uid,'examples':int(mask.sum()),'baseline_mse':float(base[mask].mean()),'candidate_mse':float(cand[mask].mean()),'candidate_over_baseline':float(cand[mask].mean()/base[mask].mean()),'mean_predicted_residual':float((best_alpha*pc[mask]).mean()),'mean_true_residual':float((a['target_coeffs'][idx][mask]-a['anchor_coeffs'][idx][mask]).mean()),'mean_confidence':float(cf[mask].mean())})
    with (OUT/'per_network.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'decision':report['decision'],'best_epoch':report['best_epoch'],'alpha':best_alpha,'test_observed':edge,'test_noise_corrected':nc,'gate':report['gate_noise_corrected']},indent=2))
if __name__=='__main__':main()
