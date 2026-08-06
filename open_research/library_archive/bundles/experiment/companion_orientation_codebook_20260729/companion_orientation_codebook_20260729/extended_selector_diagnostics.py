#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
EPS=1e-30
FIXED=2
SCORES={
 'p2_consensus':lambda o:o['p2_consensus_cos'],
 'p2_common':lambda o:o['p2_common_cos'],
 'first_layer':lambda o:-o['first_layer_relerr'],
 'p1_p2':lambda o:o['p1_p2_cos'],
 'c1_c2':lambda o:o['c1_c2_cos'],
 'c2_p2':lambda o:o['c2_p2_cos'],
 'nested_rel':lambda o:-o['nested_rel'],
 'p2_norm_high':lambda o:math.log1p(o['p2_norm']),
 'p2_norm_low':lambda o:-math.log1p(o['p2_norm']),
 'c2_norm_high':lambda o:math.log1p(o['c2_norm']),
 'c2_norm_low':lambda o:-math.log1p(o['c2_norm']),
 'phase_combo':lambda o:o['p2_consensus_cos']+o['p1_p2_cos']+o['c1_c2_cos']+.5*o['c2_p2_cos']-o['nested_rel'],
 'c17_p2_self_consistency':lambda o:float(np.asarray(o['correction'])@np.asarray(o['p2'])/(np.linalg.norm(o['correction'])*np.linalg.norm(o['p2'])+EPS)),
}

def load(raw):
 m=json.loads((raw/'freeze_manifest.json').read_text()); split={int(b):s for s,bs in m['split_by_base'].items() for b in bs};cs=[]
 for p in sorted(raw.glob('case_*.json')):
  c=json.loads(p.read_text());c['split']=split[int(c['base_id'])];cs.append(c)
 return m,cs

def greedy(dev,n):
 sub=[FIXED];out={1:sub.copy()}
 for k in range(2,n+1):
  j=min((sum(min(c['orientations'][i]['mse_nc'] for i in sub+[q]) for c in dev),q) for q in range(n) if q not in sub)[1]
  sub.append(j)
  if k in (2,4,8,n):out[k]=sub.copy()
 return out

def rec_metrics(cs,ids):
 ratios=[];m=[];b=[];hit=[];regret=[]
 for c,oi in zip(cs,ids):
  o=c['orientations'][oi];m.append(o['mse_nc']);b.append(c['baseline_mse_nc']);ratios.append(o['ratio_nc'])
  best=min(range(len(c['orientations'])),key=lambda j:c['orientations'][j]['mse_nc']);hit.append(oi==best)
  regret.append((o['mse_nc']-c['orientations'][best]['mse_nc'])/c['baseline_mse_nc'])
 r=np.array(ratios)
 return {'pooled_ratio':float(sum(m)/sum(b)),'wins':int(np.sum(r<1)),'median':float(np.median(r)),'p90':float(np.quantile(r,.9)),'worst':float(r.max()),
         'oracle_identity_accuracy_full8':float(np.mean(hit)),'mean_regret_to_oracle8':float(np.mean(regret))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();m,cs=load(a.raw);dev=[c for c in cs if c['split']=='development'];subs=greedy(dev,len(m['orientation_seeds']))
 rows=[];extra={}
 for k,sub in subs.items():
  for sp in ('development','calibration','test'):
   ss=[c for c in cs if c['split']==sp]
   for name,fn in SCORES.items():
    ids=[];rhos=[]
    for c in ss:
     scores=np.array([fn(c['orientations'][i]) for i in sub]);reward=np.array([-c['orientations'][i]['mse_nc']/c['baseline_mse_nc'] for i in sub]);ids.append(sub[int(np.argmax(scores))])
     if len(sub)>1:rhos.append(float(spearmanr(scores,reward).statistic))
    rows.append({'k':k,'subset':sub,'split':sp,'selector':name,'mean_within_case_reward_spearman':float(np.nanmean(rhos)) if rhos else 1.0,**rec_metrics(ss,ids)})
   # Research-only: which p2 points toward true repair; tests whether paired direction contains orientation info.
   ids=[];rhos=[];scale_rel=[];p2cos=[]
   for c in ss:
    err=np.asarray(c['truth'])-np.asarray(c['y0']);vals=[];reward=[]
    for i in sub:
     o=c['orientations'][i];p=np.asarray(o['p2']);corr=np.asarray(o['correction'])
     vals.append(float(p@err/(np.linalg.norm(p)*np.linalg.norm(err)+EPS)));reward.append(-o['mse_nc']/c['baseline_mse_nc'])
     p2cos.append(float(p@corr/(np.linalg.norm(p)*np.linalg.norm(corr)+EPS)))
     scale_rel.append(float(np.linalg.norm(p)/(np.linalg.norm(corr)+EPS)))
    ids.append(sub[int(np.argmax(vals))])
    if len(sub)>1:rhos.append(float(spearmanr(vals,reward).statistic))
   rows.append({'k':k,'subset':sub,'split':sp,'selector':'oracle_true_error_p2_direction','mean_within_case_reward_spearman':float(np.nanmean(rhos)) if rhos else 1.0,**rec_metrics(ss,ids)})
   extra[f'k{k}_{sp}']={'mean_p2_c17_cosine':float(np.mean(p2cos)),'median_p2_to_c17_norm_ratio':float(np.median(scale_rel))}
 # Ridge selected raw from existing summary, reconstruct models.
 summary=json.loads((a.out.parent/'analysis'/'RESULTS_SUMMARY.json').read_text()) if (a.out.parent/'analysis'/'RESULTS_SUMMARY.json').exists() else None
 a.out.parent.mkdir(parents=True,exist_ok=True)
 keys=[]
 for r in rows:
  for q in r:
   if q not in keys:keys.append(q)
 with a.out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({q:json.dumps(v) if isinstance(v,(list,dict)) else v for q,v in r.items()})
 (a.out.parent/'EXTENDED_SELECTOR_DIAGNOSTICS.json').write_text(json.dumps({'rows':rows,'probe_diagnostics':extra},indent=2)+'\n')
 print(json.dumps({'test':[r for r in rows if r['split']=='test'],'probe':{k:v for k,v in extra.items() if k.endswith('_test')}},indent=2))
if __name__=='__main__':main()
