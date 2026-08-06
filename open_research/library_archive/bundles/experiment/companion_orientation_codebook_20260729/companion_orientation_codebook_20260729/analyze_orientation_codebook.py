#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from typing import Any
import numpy as np

D=256; DEPTH=32; TARGET_LAYERS=30; MAIN_BASES=112; FULL_BASES=129; AMP=.20
BASELINE_EFFECTIVE=175.62e9
FIXED_R3=2
EPS=1e-30
FEATURES=['p1_p2_cos','c1_c2_cos','c2_p2_cos','p2_norm','c2_norm','nested_rel','first_layer_relerr','p2_common_cos','p2_consensus_cos']


def load_cases(raw:Path):
    manifest=json.loads((raw/'freeze_manifest.json').read_text())
    cases=[]
    for p in sorted(raw.glob('case_*.json')):
        r=json.loads(p.read_text()); r['_path']=str(p); cases.append(r)
    expected=len(manifest['base_ids'])*manifest['variants']
    if len(cases)!=expected: raise RuntimeError(f'expected {expected} cases, found {len(cases)}')
    split_by={int(b):s for s,bs in manifest['split_by_base'].items() for b in bs}
    for r in cases:r['split']=split_by[int(r['base_id'])]
    return manifest,cases

def mse_nc(y,truth,noise): return max(float(np.mean((y-truth)**2))-noise,1e-20)
def ratio_for(c,oi): return c['orientations'][oi]['mse_nc']/c['baseline_mse_nc']
def observed_ratio_for(c,oi): return c['orientations'][oi]['mse']/c['baseline_mse']

def policy_record(c,oi:int|None):
    y0=np.asarray(c['y0']); truth=np.asarray(c['truth']); noise=c['truth_noise_mse']; err=truth-y0
    if oi is None:
        corr=np.zeros_like(y0); y=y0
    else:
        corr=np.asarray(c['orientations'][oi]['correction']); y=y0+corr
    mse=float(np.mean((y-truth)**2)); mn=max(mse-noise,1e-20)
    return {'mse':mse,'mse_nc':mn,'ratio':mse/c['baseline_mse'],'ratio_nc':mn/c['baseline_mse_nc'],
            'ip':float(err@corr),'norm':float(np.linalg.norm(corr)),'cos':float(err@corr/(np.linalg.norm(err)*np.linalg.norm(corr)+EPS)),
            'orientation':oi,'correction':corr}

def grouped_bootstrap(records,cases,n=10000,seed=2026072906,cost_ratio=1.0):
    bases=sorted(set(int(c['base_id']) for c in cases)); by={b:[] for b in bases}
    for r,c in zip(records,cases):by[int(c['base_id'])].append((r,c))
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        sample=rng.choice(bases,size=len(bases),replace=True); cm=0.;bm=0.
        for b in sample:
            for r,c in by[int(b)]: cm+=r['mse_nc']; bm+=c['baseline_mse_nc']
        vals.append(cost_ratio*cm/max(bm,EPS))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def summarize(name,records,cases,cost_ratio=1.0,bootstrap=True):
    cm=sum(r['mse_nc'] for r in records); bm=sum(c['baseline_mse_nc'] for c in cases)
    ratios=np.array([r['ratio_nc'] for r in records]); obs=np.array([r['ratio'] for r in records])
    return {'name':name,'n_cases':len(records),'n_bases':len(set(c['base_id'] for c in cases)),
            'pooled_raw_ratio_nc':float(cm/bm),'pooled_raw_ratio_observed':float(sum(r['mse'] for r in records)/sum(c['baseline_mse'] for c in cases)),
            'adjusted_ratio_projected':float(cost_ratio*cm/bm),'cost_ratio_projected':float(cost_ratio),
            'wins':int(np.sum(ratios<1)),'win_rate':float(np.mean(ratios<1)),'median':float(np.median(ratios)),
            'p90':float(np.quantile(ratios,.9)),'worst':float(ratios.max()),'observed_worst':float(obs.max()),
            'mean_error_correction_ip':float(np.mean([r['ip'] for r in records])),
            'mean_correction_norm':float(np.mean([r['norm'] for r in records])),
            'mean_correction_cosine':float(np.mean([r['cos'] for r in records])),
            'grouped_adjusted_ci95':grouped_bootstrap(records,cases,cost_ratio=cost_ratio) if bootstrap else None}

def companion_bases_for_k(k,common=False): return 15+2*k+(2 if common else 0)
def cost_ratio(k,common=False,first_layer_only=False):
    if first_layer_only:
        units=MAIN_BASES*DEPTH+17*TARGET_LAYERS+2*k
    else: units=MAIN_BASES*DEPTH+companion_bases_for_k(k,common)*TARGET_LAYERS
    return units/(FULL_BASES*DEPTH)
def effective_compute(k,common=False,first_layer_only=False):return BASELINE_EFFECTIVE*cost_ratio(k,common,first_layer_only)
def dense_flops(k,common=False,first_layer_only=False):
    units=(MAIN_BASES*DEPTH+17*TARGET_LAYERS+2*k) if first_layer_only else (MAIN_BASES*DEPTH+companion_bases_for_k(k,common)*TARGET_LAYERS)
    return 2*D*D*512*units

def greedy_subsets(dev,nori):
    subset=[FIXED_R3]; out={1:subset.copy()}
    for k in range(2,nori+1):
        best=None
        for j in range(nori):
            if j in subset:continue
            s=subset+[j]
            score=sum(min(c['orientations'][i]['mse_nc'] for i in s) for c in dev)/sum(c['baseline_mse_nc'] for c in dev)
            cand=(score,j)
            if best is None or cand<best:best=cand
        subset.append(best[1]);
        if k in (2,4,8,16) or k==nori:out[k]=subset.copy()
    return out

def select_records(cases,selector):
    rec=[]; ids=[]; conf=[]
    for c in cases:
        oi,cf=selector(c); ids.append(oi);conf.append(cf);rec.append(policy_record(c,oi))
    return rec,ids,np.asarray(conf)

def simple_score(c,subset,kind):
    O=c['orientations']
    if kind=='consensus': vals=[O[i]['p2_consensus_cos'] for i in subset]
    elif kind=='common': vals=[O[i]['p2_common_cos'] for i in subset]
    elif kind=='first_layer': vals=[-O[i]['first_layer_relerr'] for i in subset]
    elif kind=='nested': vals=[O[i]['p1_p2_cos']+O[i]['c1_c2_cos']-O[i]['nested_rel'] for i in subset]
    elif kind=='phase_combo': vals=[O[i]['p2_consensus_cos']+O[i]['p1_p2_cos']+O[i]['c1_c2_cos']+.5*O[i]['c2_p2_cos']-O[i]['nested_rel'] for i in subset]
    else:raise KeyError(kind)
    order=np.argsort(vals)[::-1]; return subset[int(order[0])],float(vals[order[0]]-(vals[order[1]] if len(order)>1 else -1))

def feat_matrix(c,subset):
    X=[]
    for i in subset:
        o=c['orientations'][i]
        row=[]
        for f in FEATURES:
            v=float(o[f]);
            if f in ('p2_norm','c2_norm','nested_rel','first_layer_relerr'):v=math.log1p(max(v,0))
            row.append(v)
        X.append(row)
    return np.asarray(X)

def fit_ridge_ranker(dev,subset):
    bases=sorted(set(c['base_id'] for c in dev)); lambdas=[0,.01,.1,1,10,100]
    def train(train,lam):
        XX=[];yy=[]
        for c in train:
            X=feat_matrix(c,subset); y=np.array([-c['orientations'][i]['mse_nc']/c['baseline_mse_nc'] for i in subset])
            XX.append(X-X.mean(0)); yy.append(y-y.mean())
        X=np.vstack(XX); y=np.concatenate(yy); mu=X.mean(0);sd=X.std(0)+1e-8;Z=(X-mu)/sd
        A=Z.T@Z+lam*np.eye(Z.shape[1]); b=Z.T@y
        w=np.linalg.lstsq(A,b,rcond=None)[0]
        return mu,sd,w
    cv=[]
    for lam in lambdas:
        cm=bm=0
        for b in bases:
            tr=[c for c in dev if c['base_id']!=b];va=[c for c in dev if c['base_id']==b]
            mu,sd,w=train(tr,lam)
            for c in va:
                X=feat_matrix(c,subset);s=((X-mu)/sd)@w;oi=subset[int(np.argmax(s))]
                cm+=c['orientations'][oi]['mse_nc'];bm+=c['baseline_mse_nc']
        cv.append((cm/bm,lam))
    cv.sort();lam=cv[0][1];mu,sd,w=train(dev,lam)
    return {'lambda':lam,'mu':mu.tolist(),'sd':sd.tolist(),'w':w.tolist(),'cv':cv}

def ridge_selector(model,subset):
    mu=np.asarray(model['mu']);sd=np.asarray(model['sd']);w=np.asarray(model['w'])
    def f(c):
        X=feat_matrix(c,subset); X=X-X.mean(0); s=((X-mu)/sd)@w;order=np.argsort(s)[::-1]
        return subset[int(order[0])],float(s[order[0]]-(s[order[1]] if len(order)>1 else -1))
    return f

def calibrate_abstention(cal,base_selector,subset):
    selected,ids,conf=select_records(cal,base_selector)
    qs=np.unique(np.r_[-np.inf,np.quantile(conf,[0,.2,.4,.6,.8,1]),np.inf])
    options=[]
    for fallback in ('fixed','zero'):
        for t in qs:
            rec=[]
            for c,oi,cf in zip(cal,ids,conf):
                use=oi if cf>=t else (FIXED_R3 if fallback=='fixed' else None)
                rec.append(policy_record(c,use))
            s=summarize('tmp',rec,cal,cost_ratio=1,bootstrap=False)
            penalty=max(0,s['worst']-1.15)*100
            options.append((s['pooled_raw_ratio_nc']+penalty,-s['win_rate'],fallback,float(t),s))
    options.sort(key=lambda x:(x[0],x[1],x[2],x[3])); return {'fallback':options[0][2],'threshold':options[0][3],'calibration':options[0][4]}

def abstained_selector(base_selector,rule):
    def f(c):
        oi,cf=base_selector(c)
        if cf<rule['threshold']:oi=FIXED_R3 if rule['fallback']=='fixed' else None
        return oi,cf
    return f

def oracle_selector(subset):return lambda c:(min(subset,key=lambda i:c['orientations'][i]['mse_nc']),1.0)
def fixed_selector(i):return lambda c:(i,1.0)

def preference_stability(cases,subset):
    if len(subset)==1:
        return {'pairwise_identity_agreement':1.0,'mean_modal_fraction':1.0,'mean_rank_correlation':1.0,'n_base_families':len(set(c['base_id'] for c in cases))}
    by={}
    for c in cases:by.setdefault(c['base_id'],[]).append(c)
    agreements=[];rhos=[];modal=[]
    for cs in by.values():
        ids=[];vectors=[]
        for c in cs:
            vals=np.array([c['orientations'][i]['ratio_nc'] for i in subset]);ids.append(subset[int(np.argmin(vals))]);vectors.append(vals)
        agreements.extend([ids[i]==ids[j] for i in range(len(ids)) for j in range(i)])
        modal.append(max(ids.count(x) for x in set(ids))/len(ids))
        for i in range(len(vectors)):
            for j in range(i):
                a=np.argsort(np.argsort(vectors[i]));b=np.argsort(np.argsort(vectors[j]));rhos.append(float(np.corrcoef(a,b)[0,1]))
    return {'pairwise_identity_agreement':float(np.mean(agreements)),'mean_modal_fraction':float(np.mean(modal)),
            'mean_rank_correlation':float(np.mean(rhos)),'n_base_families':len(by)}

def geometry(cases,subset):
    pair=[];pca=[];meanfrac=[];shared_rat=[];oracle_rat=[]
    for c in cases:
        C=np.array([c['orientations'][i]['correction'] for i in subset]);
        for i in range(len(C)):
            for j in range(i):pair.append(float(C[i]@C[j]/(np.linalg.norm(C[i])*np.linalg.norm(C[j])+EPS)))
        cen=C-C.mean(0);s=np.linalg.svd(cen,compute_uv=False);pca.append(1.0 if len(subset)==1 else float(s[0]**2/max(np.sum(s*s),EPS)))
        meanfrac.append(float(np.linalg.norm(C.mean(0))/max(np.mean(np.linalg.norm(C,axis=1)),EPS)))
        y0=np.asarray(c['y0']);truth=np.asarray(c['truth']);noise=c['truth_noise_mse'];cm=C.mean(0)
        shared_rat.append(mse_nc(y0+cm,truth,noise)/c['baseline_mse_nc'])
        oracle_rat.append(min(c['orientations'][i]['mse_nc'] for i in subset)/c['baseline_mse_nc'])
    return {'pairwise_correction_cosine_mean':1.0 if not pair else float(np.mean(pair)),'pairwise_correction_cosine_p10':1.0 if not pair else float(np.quantile(pair,.1)),
            'centered_first_pc_fraction_median':float(np.median(pca)),'mean_correction_norm_fraction_median':float(np.median(meanfrac)),
            'shared_mean_pooled_ratio':float(np.mean(shared_rat)),'oracle_orientation_mean_case_ratio':float(np.mean(oracle_rat))}

def direction_amplitude(cases,subset):
    vals={'fixed_r3':[],'best_orientation_fixed_amp':[],'fixed_r3_oracle_scale':[],'best_orientation_oracle_scale':[]}
    for c in cases:
        vals['fixed_r3'].append(c['orientations'][FIXED_R3]['mse_nc'])
        vals['best_orientation_fixed_amp'].append(min(c['orientations'][i]['mse_nc'] for i in subset))
        # oracle_scale_mse is observed; noise correct here
        vals['fixed_r3_oracle_scale'].append(max(c['orientations'][FIXED_R3]['oracle_scale_mse']-c['truth_noise_mse'],1e-20))
        vals['best_orientation_oracle_scale'].append(min(max(c['orientations'][i]['oracle_scale_mse']-c['truth_noise_mse'],1e-20) for i in subset))
    bm=sum(c['baseline_mse_nc'] for c in cases)
    return {k:float(sum(v)/bm) for k,v in vals.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    manifest,cases=load_cases(a.raw);nori=len(manifest['orientation_seeds'])
    split={s:[c for c in cases if c['split']==s] for s in ('development','calibration','test')};dev=split['development'];cal=split['calibration'];test=split['test']
    subsets=greedy_subsets(dev,nori)
    # Ensure all requested sizes appear.
    ks=sorted(subsets)
    frontier=[];selector_rows=[];models={};frozen_candidates={}
    for k in ks:
        sub=subsets[k]
        for sp,cs in split.items():
            rec,_,_=select_records(cs,oracle_selector(sub));sm=summarize(f'oracle_best_k{k}',rec,cs,cost_ratio=cost_ratio(k));
            frontier.append({'k':k,'subset':sub,'split':sp,**sm,'projected_effective_compute':effective_compute(k),'dense_equivalent_flops':dense_flops(k)})
        # Simple selectors evaluated on development; no calibration/test selection yet.
        kinds=['consensus','common','first_layer','nested','phase_combo']
        candidates=[]
        for kind in kinds:
            fn=lambda c,kind=kind,sub=sub:simple_score(c,sub,kind)
            rec,_,_=select_records(dev,fn);s=summarize(f'{kind}_k{k}',rec,dev,cost_ratio=cost_ratio(k,common=(kind=='common')),bootstrap=False)
            candidates.append((s['pooled_raw_ratio_nc'],s['worst'],kind,fn,s))
        if k>1:
            model=fit_ridge_ranker(dev,sub);models[f'k{k}']=model;rf=ridge_selector(model,sub);rec,_,_=select_records(dev,rf);rs=summarize(f'ridge_k{k}',rec,dev,cost_ratio=cost_ratio(k),bootstrap=False)
            candidates.append((rs['pooled_raw_ratio_nc'],rs['worst'],'ridge',rf,rs))
        candidates.sort(key=lambda z:(z[0],z[1],z[2]));best=candidates[0]
        basefn=best[3];rule=calibrate_abstention(cal,basefn,sub);finalfn=abstained_selector(basefn,rule)
        frozen_candidates[k]={'subset':sub,'selector':best[2],'development':best[4],'abstention':rule}
        for sp,cs in split.items():
            rec,ids,conf=select_records(cs,finalfn);common=(best[2]=='common');cr=cost_ratio(k,common=common,first_layer_only=(best[2]=='first_layer'))
            sm=summarize(f'frozen_{best[2]}_k{k}',rec,cs,cost_ratio=cr)
            selector_rows.append({'k':k,'subset':sub,'selector':best[2],'fallback':rule['fallback'],'threshold':rule['threshold'],'split':sp,
                                  **sm,'selected_orientations':ids,'mean_confidence':float(np.mean(conf)),
                                  'projected_effective_compute':BASELINE_EFFECTIVE*cr,'dense_equivalent_flops':dense_flops(k,common=common,first_layer_only=(best[2]=='first_layer'))})
    # Baselines and universal alternatives.
    baselines=[]
    devbest=min(range(nori),key=lambda i:sum(c['orientations'][i]['mse_nc'] for c in dev))
    for name,fn,k,cr in [('fixed_r3',fixed_selector(FIXED_R3),1,cost_ratio(1)),('development_best_single',fixed_selector(devbest),1,cost_ratio(1)),('zero_correction',fixed_selector(None),1,(MAIN_BASES*DEPTH)/(FULL_BASES*DEPTH))]:
        for sp,cs in split.items():
            rec,ids,conf=select_records(cs,fn);baselines.append({'name':name,'orientation':FIXED_R3 if name=='fixed_r3' else (devbest if name=='development_best_single' else None),'split':sp,
                **summarize(name,rec,cs,cost_ratio=cr),'projected_effective_compute':BASELINE_EFFECTIVE*cr})
    # Stability, geometry, direction/amplitude and saturation.
    all_diag={}
    for k,sub in subsets.items():
        all_diag[f'k{k}']={'subset':sub,'preference_stability_all':preference_stability(cases,sub),'preference_stability_test':preference_stability(test,sub),
                          'geometry_all':geometry(cases,sub),'direction_amplitude_test':direction_amplitude(test,sub)}
    # ceiling saturation using test pooled MSE gains from fixed to oracle K.
    fixed_test=sum(c['orientations'][FIXED_R3]['mse_nc'] for c in test);fullk=max(ks);full_or=sum(min(c['orientations'][i]['mse_nc'] for i in subsets[fullk]) for c in test)
    sat={}
    for k,sub in subsets.items():
        v=sum(min(c['orientations'][i]['mse_nc'] for i in sub) for c in test)
        sat[str(k)]=float((fixed_test-v)/max(fixed_test-full_or,EPS))
    # Pairwise pooled cosine matrix.
    mat=np.zeros((nori,nori));cnt=0
    for c in cases:
        C=np.array([o['correction'] for o in c['orientations']]);N=np.linalg.norm(C,axis=1);mat+=(C@C.T)/np.maximum(N[:,None]*N[None,:],EPS);cnt+=1
    mat/=cnt
    # Flat rows.
    case_rows=[];ori_rows=[]
    for c in cases:
        case_rows.append({k:c[k] for k in ['case_id','base_id','variant','split','truth_noise_mse','baseline_mse','baseline_mse_nc','reduced_base_mse','y0_mse','seconds','peak_rss_kb']})
        for o in c['orientations']:
            ori_rows.append({'case_id':c['case_id'],'base_id':c['base_id'],'variant':c['variant'],'split':c['split'],**{k:o[k] for k in o if k not in ('correction','p1','p2','c1','c2')}})
    def write_csv(path,rows):
        if not rows:return
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys:keys.append(k)
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    write_csv(a.outdir/'CASE_ROWS.csv',case_rows);write_csv(a.outdir/'ORIENTATION_ROWS.csv',ori_rows);write_csv(a.outdir/'CODEBOOK_FRONTIER.csv',frontier);write_csv(a.outdir/'SELECTOR_RESULTS.csv',selector_rows);write_csv(a.outdir/'BASELINES.csv',baselines)
    with (a.outdir/'PAIRWISE_CORRECTION_COSINE.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['orientation']+list(range(nori)))
        for i,row in enumerate(mat):w.writerow([i]+row.tolist())
    results={'manifest':manifest,'subsets':subsets,'development_best_single':devbest,'baselines':baselines,'frontier':frontier,'selector_results':selector_rows,
             'frozen_candidates':frozen_candidates,'ridge_models':models,'diagnostics':all_diag,'test_ceiling_saturation':sat,
             'reference_noise_fraction':{'mean':float(np.mean([c['truth_noise_mse']/c['baseline_mse'] for c in cases])),
                                         'p90':float(np.quantile([c['truth_noise_mse']/c['baseline_mse'] for c in cases],.9)),
                                         'worst':float(np.max([c['truth_noise_mse']/c['baseline_mse'] for c in cases]))},
             'prototype_runtime':{'mean_case_seconds':float(np.mean([c['seconds'] for c in cases])),'p90_case_seconds':float(np.quantile([c['seconds'] for c in cases],.9)),
                                  'peak_rss_kb':int(max(c['peak_rss_kb'] for c in cases))}}
    (a.outdir/'RESULTS_SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'subsets':subsets,'devbest':devbest,'test_saturation':sat,'reference_noise':results['reference_noise_fraction'],
                      'test_selectors':[r for r in selector_rows if r['split']=='test']},indent=2))
if __name__=='__main__':main()
