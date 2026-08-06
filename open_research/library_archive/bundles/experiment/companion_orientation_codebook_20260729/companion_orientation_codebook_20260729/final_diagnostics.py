#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
EPS=1e-30
SUBSETS={1:[2],2:[2,7],4:[2,7,1,4],8:[2,7,1,4,3,5,0,6]}
COST={1:(112*32+17*30)/(129*32),2:(112*32+19*30)/(129*32),4:(112*32+23*30)/(129*32),8:(112*32+31*30)/(129*32)}

def load(raw):return [json.loads(p.read_text()) for p in sorted(raw.glob('case_*.json'))]
def pooled(cs,ms):return float(sum(ms)/sum(c['baseline_mse_nc'] for c in cs))
def metrics(cs,ids,cost=1):
 ms=[c['orientations'][i]['mse_nc'] for c,i in zip(cs,ids)];r=np.array([c['orientations'][i]['ratio_nc'] for c,i in zip(cs,ids)])
 return {'raw':pooled(cs,ms),'adjusted':cost*pooled(cs,ms),'wins':int(np.sum(r<1)),'worst':float(r.max()),'median':float(np.median(r)),'p90':float(np.quantile(r,.9))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cs=load(a.raw)
 out={'oracle_frontier':{},'direction_amplitude':{},'selector_information':{},'offset_geometry':{},'rotation_stability':{}}
 fixed=[2]*len(cs);fixed_ms=[c['orientations'][2]['mse_nc'] for c in cs]
 fullids=[]
 for k,sub in SUBSETS.items():
  ids=[min(sub,key=lambda i:c['orientations'][i]['mse_nc']) for c in cs];out['oracle_frontier'][str(k)]={'subset':sub,**metrics(cs,ids,COST[k])}
  if k==8:fullids=ids
 fixed_or_scale=[];best_or_scale=[];c2_or_scale=[]
 c2ids=[int(np.argmax([o['c2_norm'] for o in c['orientations']])) for c in cs];p2ids=[int(np.argmax([o['p2_norm'] for o in c['orientations']])) for c in cs]
 for c,c2i in zip(cs,c2ids):
  n=c['truth_noise_mse'];fixed_or_scale.append(max(c['orientations'][2]['oracle_scale_mse']-n,1e-20));best_or_scale.append(min(max(o['oracle_scale_mse']-n,1e-20) for o in c['orientations']));c2_or_scale.append(max(c['orientations'][c2i]['oracle_scale_mse']-n,1e-20))
 out['direction_amplitude']={'fixed_r3_fixed_amplitude':pooled(cs,fixed_ms),'best_orientation_fixed_amplitude':out['oracle_frontier']['8']['raw'],
 'fixed_r3_oracle_scale':pooled(cs,fixed_or_scale),'best_orientation_oracle_scale':pooled(cs,best_or_scale),'c2_selector_oracle_scale':pooled(cs,c2_or_scale)}
 # Shared mean and geometry.
 shared=[];pair=[];pca=[];mean_norm=[]
 for c in cs:
  C=np.array([o['correction'] for o in c['orientations']]);cm=C.mean(0);y=np.asarray(c['y0'])+cm;t=np.asarray(c['truth']);shared.append(max(float(np.mean((y-t)**2))-c['truth_noise_mse'],1e-20));N=np.linalg.norm(C,axis=1)
  for i in range(8):
   for j in range(i):pair.append(float(C[i]@C[j]/(N[i]*N[j]+EPS)))
  s=np.linalg.svd(C-C.mean(0),compute_uv=False);pca.append(float(s[0]**2/np.sum(s*s)));mean_norm.append(float(np.linalg.norm(cm)/np.mean(N)))
 out['offset_geometry']={'shared_orientation_mean_raw':pooled(cs,shared),'pairwise_cosine_mean':float(np.mean(pair)),'pairwise_cosine_p10':float(np.quantile(pair,.1)),
 'median_centered_first_pc_fraction':float(np.median(pca)),'median_mean_norm_fraction':float(np.median(mean_norm))}
 # Probe information and oracle probe-direction ceiling.
 p2c=[];normrat=[];truth_p2_ids=[];reward_rho=[]
 for c in cs:
  err=np.asarray(c['truth'])-np.asarray(c['y0']);vals=[];rew=[]
  for o in c['orientations']:
   p=np.asarray(o['p2']);q=np.asarray(o['correction']);p2c.append(float(p@q/(np.linalg.norm(p)*np.linalg.norm(q)+EPS)));normrat.append(float(np.linalg.norm(p)/(np.linalg.norm(q)+EPS)));vals.append(float(p@err/(np.linalg.norm(p)*np.linalg.norm(err)+EPS)));rew.append(-o['ratio_nc'])
  truth_p2_ids.append(int(np.argmax(vals)));reward_rho.append(float(np.corrcoef(np.argsort(np.argsort(vals)),np.argsort(np.argsort(rew)))[0,1]))
 out['selector_information']={'c2_norm':metrics(cs,c2ids,COST[8]),'p2_norm':metrics(cs,p2ids,COST[8]),'c2_identity_accuracy':float(np.mean(np.array(c2ids)==np.array(fullids))),
 'p2_identity_accuracy':float(np.mean(np.array(p2ids)==np.array(fullids))),'oracle_true_error_p2_direction':metrics(cs,truth_p2_ids,COST[8]),
 'mean_p2_c17_cosine':float(np.mean(p2c)),'median_p2_to_c17_norm_ratio':float(np.median(normrat)),'mean_within_case_p2_reward_rank_correlation':float(np.mean(reward_rho))}
 # Stability across three rotations per base.
 for name,ids in [('oracle',fullids),('c2',c2ids),('p2',p2ids)]:
  by={}
  for c,i in zip(cs,ids):by.setdefault(c['base_id'],[]).append(i)
  ag=[];modal=[]
  for v in by.values():ag += [v[i]==v[j] for i in range(len(v)) for j in range(i)];modal.append(max(v.count(x) for x in set(v))/len(v))
  out['rotation_stability'][name]={'pairwise_identity_agreement':float(np.mean(ag)),'mean_modal_fraction':float(np.mean(modal))}
 out['no_headroom_cases_oracle8_ratio_ge_1']=int(sum(min(o['ratio_nc'] for o in c['orientations'])>=1 for c in cs))
 # Test ceiling fraction for k2/k4 relative fixed->k8.
 f=sum(fixed_ms);b8=sum(min(o['mse_nc'] for o in c['orientations']) for c in cs)
 out['ceiling_fraction']={str(k):float((f-sum(min(c['orientations'][i]['mse_nc'] for i in sub) for c in cs))/max(f-b8,EPS)) for k,sub in SUBSETS.items()}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
