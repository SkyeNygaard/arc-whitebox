#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
D=256;DEPTH=32;TARGET=30;MAIN=112;FULL=129;FIXED=2;EPS=1e-30;BASE_EFF=175.62e9
COST_FIXED=(MAIN*DEPTH+17*TARGET)/(FULL*DEPTH)
COST_K8=(MAIN*DEPTH+(15+16)*TARGET)/(FULL*DEPTH)
COST_SAFE_FALLBACK=(MAIN*DEPTH+16*TARGET+17*DEPTH)/(FULL*DEPTH)

def load(raw):
 m=json.loads((raw/'freeze_manifest.json').read_text());cs=[]
 for p in sorted(raw.glob('case_*.json')):cs.append(json.loads(p.read_text()))
 exp=len(m['base_ids'])*m['variants']
 if len(cs)!=exp:raise RuntimeError((len(cs),exp))
 return m,cs

def record(c,oi=None,full=False,cost=1):
 y0=np.asarray(c['y0']);truth=np.asarray(c['truth']);noise=c['truth_noise_mse']
 if full:y=np.asarray(c['base_output']);corr=y-y0
 elif oi is None:y=y0;corr=np.zeros_like(y0)
 else:corr=np.asarray(c['orientations'][oi]['correction']);y=y0+corr
 mse=float(np.mean((y-truth)**2));mn=max(mse-noise,1e-20);err=truth-y0
 return {'mse':mse,'mse_nc':mn,'ratio':mse/c['baseline_mse'],'ratio_nc':mn/c['baseline_mse_nc'],'cost':cost,'oi':oi,
 'ip':float(err@corr),'norm':float(np.linalg.norm(corr)),'cos':float(err@corr/(np.linalg.norm(err)*np.linalg.norm(corr)+EPS))}

def summarize(name,recs,cs,nboot=10000):
 bm=sum(c['baseline_mse_nc'] for c in cs);cm=sum(r['mse_nc'] for r in recs);adj=sum(r['mse_nc']*r['cost'] for r in recs)/bm
 rat=np.array([r['ratio_nc'] for r in recs]);obs=np.array([r['ratio'] for r in recs]);bases=sorted(set(c['base_id'] for c in cs));by={b:[] for b in bases}
 for r,c in zip(recs,cs):by[c['base_id']].append((r,c))
 rng=np.random.default_rng(2026072917);boot=[]
 for _ in range(nboot):
  sample=rng.choice(bases,len(bases),replace=True);x=y=0
  for b in sample:
   for r,c in by[int(b)]:x+=r['mse_nc']*r['cost'];y+=c['baseline_mse_nc']
  boot.append(x/y)
 return {'name':name,'n_cases':len(cs),'n_bases':len(bases),'pooled_raw_ratio_nc':cm/bm,'pooled_raw_ratio_observed':sum(r['mse'] for r in recs)/sum(c['baseline_mse'] for c in cs),
 'adjusted_ratio_projected':adj,'mean_cost_ratio_mse_weighted':sum(r['mse_nc']*r['cost'] for r in recs)/max(cm,EPS),
 'projected_effective_compute_at_mean_cost':BASE_EFF*np.mean([r['cost'] for r in recs]),'wins':int(np.sum(rat<1)),'win_rate':float(np.mean(rat<1)),
 'median':float(np.median(rat)),'p90':float(np.quantile(rat,.9)),'worst':float(rat.max()),'observed_worst':float(obs.max()),
 'mean_error_correction_ip':float(np.mean([r['ip'] for r in recs])),'mean_correction_norm':float(np.mean([r['norm'] for r in recs])),
 'mean_correction_cosine':float(np.mean([r['cos'] for r in recs])),'grouped_adjusted_ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

def select(c,kind):
 O=c['orientations']
 if kind=='fixed':return FIXED
 if kind=='c2':return int(np.argmax([o['c2_norm'] for o in O]))
 if kind=='p2':return int(np.argmax([o['p2_norm'] for o in O]))
 if kind=='oracle':return int(np.argmin([o['mse_nc'] for o in O]))
 raise KeyError(kind)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True);m,cs=load(a.raw)
 arms={}
 for name,kind,cost in [('fixed_r3','fixed',COST_FIXED),('primary_c2_norm','c2',COST_K8),('comparator_p2_norm','p2',COST_K8),('oracle_best8','oracle',COST_K8)]:
  rec=[record(c,select(c,kind),cost=cost) for c in cs];arms[name]={'summary':summarize(name,rec,cs),'selected_orientations':[r['oi'] for r in rec]}
 # Frozen safety rule and full-baseline fallback.
 rec=[];flags=[]
 for c in cs:
  j=select(c,'c2');o=c['orientations'][j];c2=np.asarray(o['c2']);c17=np.asarray(o['correction']);cc=float(c2@c17/(np.linalg.norm(c2)*np.linalg.norm(c17)+EPS))
  use=(o['p1_p2_cos']>=.90 and o['nested_rel']<=.70 and cc>=.30);flags.append(use)
  rec.append(record(c,j,cost=COST_K8) if use else record(c,full=True,cost=COST_SAFE_FALLBACK))
 arms['safe_full_baseline']={'summary':summarize('safe_full_baseline',rec,cs),'selected_orientations':[r['oi'] for r in rec],'application_flags':flags,'coverage':float(np.mean(flags))}
 # Base-network rotation stability.
 diag={}
 for kind in ('c2','p2','oracle'):
  by={}
  for c in cs:by.setdefault(c['base_id'],[]).append(select(c,kind))
  pairs=[];modal=[]
  for ids in by.values():
   pairs += [ids[i]==ids[j] for i in range(len(ids)) for j in range(i)];modal.append(max(ids.count(x) for x in set(ids))/len(ids))
  diag[kind]={'pairwise_identity_agreement':float(np.mean(pairs)),'mean_modal_fraction':float(np.mean(modal))}
 # Pairwise geometry, direction/amplitude.
 pair=[];pca=[]
 for c in cs:
  C=np.array([o['correction'] for o in c['orientations']]);N=np.linalg.norm(C,axis=1)
  for i in range(8):
   for j in range(i):pair.append(float(C[i]@C[j]/(N[i]*N[j]+EPS)))
  s=np.linalg.svd(C-C.mean(0),compute_uv=False);pca.append(float(s[0]**2/np.sum(s*s)))
 diag['geometry']={'pairwise_cosine_mean':float(np.mean(pair)),'pairwise_cosine_p10':float(np.quantile(pair,.1)),'median_centered_first_pc_fraction':float(np.median(pca))}
 # Flat rows.
 rows=[]
 for c in cs:
  row={'case_id':c['case_id'],'base_id':c['base_id'],'variant':c['variant'],'baseline_mse_nc':c['baseline_mse_nc'],'noise_fraction':c['truth_noise_mse']/c['baseline_mse']}
  for kind in ('fixed','c2','p2','oracle'):
   j=select(c,kind);row[f'{kind}_orientation']=j;row[f'{kind}_ratio_nc']=c['orientations'][j]['ratio_nc']
  rows.append(row)
 with (a.outdir/'VALIDATION_ROWS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'manifest':m,'cost_model':{'fixed':COST_FIXED,'k8':COST_K8,'safe_fallback':COST_SAFE_FALLBACK},'arms':arms,'diagnostics':diag,
 'reference_noise_fraction':{'mean':float(np.mean([c['truth_noise_mse']/c['baseline_mse'] for c in cs])),'p90':float(np.quantile([c['truth_noise_mse']/c['baseline_mse'] for c in cs],.9)),'worst':float(np.max([c['truth_noise_mse']/c['baseline_mse'] for c in cs]))},
 'prototype_runtime':{'mean_case_seconds':float(np.mean([c['seconds'] for c in cs])),'p90_case_seconds':float(np.quantile([c['seconds'] for c in cs],.9)),'peak_rss_kb':int(max(c['peak_rss_kb'] for c in cs))}}
 (a.outdir/'VALIDATION_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
