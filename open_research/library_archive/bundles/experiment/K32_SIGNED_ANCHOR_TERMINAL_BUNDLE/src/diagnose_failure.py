#!/usr/bin/env python3
from pathlib import Path
import json, sys, os
import numpy as np, torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import train_models as tm;import evaluate_final as ef
exp=HERE.parent;data=tm.load_split(exp/'final_test_data','test')
with np.load(exp/'models/normalization.npz') as z:norm={k:z[k].copy() for k in z.files}
g=((data['global_features']-norm['global_mean'])/norm['global_std']).astype(np.float32);t=((data['token_features']-norm['token_mean'])/norm['token_std']).astype(np.float32);std=(g,t,(data['target_delta']/float(norm['target_scale'])).astype(np.float32))
torch.set_num_threads(5);torch.set_num_interop_threads(1);models=ef.load_models(exp,torch.device('cpu'));ens=tm.ensemble_predict(models,std,norm,data,torch.device('cpu'))
truth=.5*(data['truth_half1']+data['truth_half2']);sample=data['sample_prediction'];base=np.asarray(data['base_mse']).reshape(-1)

def metrics(pred):
 mse=np.mean((pred-truth)**2,axis=1);rr=mse/base;return {'aggregate_ratio':float(mse.sum()/base.sum()),'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max())}

def oracle_scale(corr):
 e=truth-sample;den=np.sum(corr*corr,axis=1);s=np.sum(corr*e,axis=1)/np.maximum(den,1e-30);pred=sample+s[:,None]*corr;return metrics(pred),s
cm=ens['mean_correction']; m1,s1=oracle_scale(cm)
# Oracle token signs but predicted magnitudes.
dsign=np.abs(ens['mean_delta'])*np.sign(data['target_delta']);m2=metrics(tm.predict_candidate(dsign,data))
# Exact anchor direction, model-predicted L2 magnitude.
true=data['target_delta'];pn=np.linalg.norm(ens['mean_delta'],axis=1);tn=np.linalg.norm(true,axis=1);dmag=true*(pn/np.maximum(tn,1e-30))[:,None];m3=metrics(tm.predict_candidate(dmag,data))
# Fixed mean-by-rank template, optimal scalar diagnostic.
train=tm.load_split(exp/'data','train');template=train['target_delta'].mean(axis=0);tcorr=np.einsum('p,npd->nd',template,data['beta_bar']);m4,s4=oracle_scale(tcorr)
# Median fixed scalar on training optimal scales, evaluated here without test tuning.
ttruth=.5*(train['truth_half1']+train['truth_half2']);te=ttruth-train['sample_prediction'];td=np.einsum('p,npd->nd',template,train['beta_bar']);ts=np.sum(td*te,axis=1)/np.maximum(np.sum(td*td,axis=1),1e-30);fixed=float(np.median(ts));m5=metrics(sample+fixed*tcorr)
out={'diagnosis':'sign_and_direction_failure','predicted_direction_with_oracle_scale':m1,'oracle_scale_distribution':{'median':float(np.median(s1)),'mean':float(np.mean(s1)),'positive_fraction':float(np.mean(s1>0))},'predicted_magnitude_with_oracle_token_signs':m2,'exact_direction_with_predicted_norm':m3,'fixed_rank_template_with_oracle_scale':m4,'template_oracle_scale_distribution':{'median':float(np.median(s4)),'mean':float(np.mean(s4)),'positive_fraction':float(np.mean(s4>0))},'fixed_rank_template_with_train_median_scale':{'scale':fixed,**m5}}
(exp/'results/diagnostic_ablation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
