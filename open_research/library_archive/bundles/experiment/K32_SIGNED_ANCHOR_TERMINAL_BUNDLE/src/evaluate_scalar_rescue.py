#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import train_models as tm;import train_scalar_rescue as sr

def load_models(exp,device):
 out=[]
 for p in sorted((exp/'rescue/models').glob('scalar_model_*.pt')):
  z=torch.load(p,map_location='cpu',weights_only=False);m=sr.ScalarMLP(z['input_dim']);m.load_state_dict(z['state_dict']);m.to(device);m.eval();out.append(m)
 return out

def metrics(scale,d,data):
 pred=data['sample_prediction']+scale[:,None]*d;y1=data['truth_half1'];y2=data['truth_half2'];truth=.5*(y1+y2);mse=np.mean((pred-truth)**2,1);um=np.mean((pred-y1)*(pred-y2),1);base=np.asarray(data['base_mse']).reshape(-1);baseu=np.mean((data['baseline_prediction']-y1)*(data['baseline_prediction']-y2),1);rr=mse/base
 return {'pred':pred,'mse':mse,'umse':um,'ratios':rr,'aggregate_ratio':float(mse.sum()/base.sum()),'unbiased_ratio':float(um.sum()/baseu.sum()),'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'p90':float(np.quantile(rr,.9)),'worst':float(rr.max())}

def bootstrap(m,data,reps=10000,seed=20260729):
 rng=np.random.default_rng(seed);ids=np.unique(data['network_id']);groups={i:np.flatnonzero(data['network_id']==i) for i in ids};base=np.asarray(data['base_mse']).reshape(-1);vals=[]
 for _ in range(reps):
  ch=rng.choice(ids,len(ids),replace=True);num=den=0
  for i in ch:ix=groups[int(i)];num+=float(m['mse'][ix].sum());den+=float(base[ix].sum())
  vals.append(num/den)
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args();exp=a.experiment;cfg=json.loads((exp/'rescue/final_test_config.json').read_text());rc=json.loads((exp/'rescue/rescue_config.json').read_text());tv=json.loads((exp/'rescue/results/training_validation.json').read_text());data=tm.load_split(exp/'rescue/final_test_data','test')
 with np.load(exp/'rescue/models/scalar_normalization_and_template.npz') as z:mu=z['mean'];sd=z['std'];template=z['template']
 x,d=sr.features(data,template);x=((x-mu)/sd).astype(np.float32);device=torch.device('cpu');torch.set_num_threads(5);torch.set_num_interop_threads(1);models=load_models(exp,device);xx=torch.from_numpy(x).to(device);sc=[]
 with torch.no_grad():
  for model in models:sc.append(model(xx).cpu().numpy())
 sc=np.stack(sc);mean=sc.mean(0);std=sc.std(0);agree=np.all(sc>0,0)|np.all(sc<0,0);shrink=np.abs(mean)/(np.abs(mean)+std+1e-12);scale=mean*shrink*agree
 cand=metrics(scale,d,data);ung=metrics(mean,d,data);zero=metrics(np.zeros(len(mean)),d,data)
 truth=.5*(data['truth_half1']+data['truth_half2']);e=truth-data['sample_prediction'];os=np.sum(d*e,1)/np.maximum(np.sum(d*d,1),1e-30);oracle=metrics(os,d,data)
 raw_ci=bootstrap(cand,data);baseline_flops=175.62e9;direct=2.238e9;dim=x.shape[1];one=2.0*(dim*256+256*128+128*64+64);added=direct+one*len(models);mult=(baseline_flops+added)/baseline_flops;adj=cand['aggregate_ratio']*mult;aci=[v*mult for v in raw_ci]
 gate={'raw_ratio_le_0_595':cand['aggregate_ratio']<=.595,'preferred_le_0_537':cand['aggregate_ratio']<=.537,'adjusted_upper_ci_below_1':aci[1]<1,'worst_le_1_15':cand['worst']<=1.15,'added_flops_below_14B':added<14e9};gate['overall_pass']=all(gate.values())
 byrot={}
 base=np.asarray(data['base_mse']).reshape(-1)
 for rot in sorted(set(data['rotation_seed'].tolist())):
  ix=np.flatnonzero(data['rotation_seed']==rot);byrot[str(rot)]={'aggregate_ratio':float(cand['mse'][ix].sum()/base[ix].sum()),'wins':int(np.sum(cand['ratios'][ix]<1)),'median':float(np.median(cand['ratios'][ix])),'worst':float(cand['ratios'][ix].max())}
 res={'terminal_state':'PASS' if gate['overall_pass'] else 'FAIL','scope':'Single mechanistic rescue: invariant scalar sign/scale learner on a frozen mean-by-rank K32 lower-order anchor template.','freeze_sha256':cfg['freeze_sha256'],'rescue_config_sha256':rc['freeze_sha256'],'model_hashes':tv['model_hashes'],'base_networks':len(set(data['network_id'].tolist())),'examples':len(mean),'rotations':sorted(set(data['rotation_seed'].tolist())),'candidate':{'aggregate_ratio':cand['aggregate_ratio'],'unbiased_ratio':cand['unbiased_ratio'],'wins':cand['wins'],'median':cand['median'],'p90':cand['p90'],'worst':cand['worst'],'bootstrap_95':raw_ci,'adjusted_ratio':adj,'adjusted_bootstrap_95':aci,'compute_multiplier':mult,'added_flops_estimate':added,'mean_scale':float(scale.mean()),'mean_abs_scale':float(np.mean(np.abs(scale))),'abstentions':int(np.sum(~agree))},'diagnostics':{'oracle_scale_template':{k:oracle[k] for k in ['aggregate_ratio','unbiased_ratio','wins','median','worst']},'ungated':{k:ung[k] for k in ['aggregate_ratio','unbiased_ratio','wins','median','worst']},'sample_null':{k:zero[k] for k in ['aggregate_ratio','unbiased_ratio','wins','median','worst']},'scale_target':{'pearson':float(np.corrcoef(mean,os)[0,1]),'sign_accuracy':float(np.mean(np.sign(mean)==np.sign(os))),'oracle_scale_positive_fraction':float(np.mean(os>0)),'predicted_scale_positive_fraction':float(np.mean(mean>0))}},'by_rotation':byrot,'gate':gate,'integrity':{'terminal_ids':sorted(set(data['network_id'].tolist())),'excluded_primary_terminal_ids':list(range(3200,3216)),'no_terminal_test_tuning':True,'normalization_template_sha256':tv['normalization_template_sha256']}}
 (exp/'rescue/results/final_test_results.json').write_text(json.dumps(res,indent=2));rows=[]
 for i in range(len(mean)):rows.append({'network_id':int(data['network_id'][i]),'rotation_seed':int(data['rotation_seed'][i]),'candidate_ratio':float(cand['ratios'][i]),'oracle_template_ratio':float(oracle['ratios'][i]),'predicted_scale':float(mean[i]),'applied_scale':float(scale[i]),'oracle_scale':float(os[i]),'agreement':int(agree[i])})
 with (exp/'rescue/results/final_test_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
