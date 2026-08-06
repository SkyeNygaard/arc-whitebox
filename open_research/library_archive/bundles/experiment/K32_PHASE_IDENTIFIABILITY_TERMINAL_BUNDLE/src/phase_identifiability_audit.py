#!/usr/bin/env python3
from __future__ import annotations
import json, glob, os, sys, math, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'phase_audit'
OUT.mkdir(exist_ok=True)

def load_files(patterns):
    rows=[]
    for pat in patterns:
        for p in sorted(glob.glob(str(pat))):
            with np.load(p, allow_pickle=True) as z:
                nrot=len(z['rotation_seeds'])
                for j in range(nrot):
                    row={k:z[k][j].copy() for k in ['global_features','token_features','target_delta','beta_bar','probe_indices','probe_directions','q_anchor','sample_prediction','baseline_prediction','oracle_prediction','truth_half1','truth_half2','target_correction']}
                    row.update(network_id=int(z['network_id']), rotation_seed=int(z['rotation_seeds'][j]), split=str(z['split']), source=p)
                    rows.append(row)
    return rows

patterns=[ROOT/'data/train_network_*.npz',ROOT/'data/validation_network_*.npz',ROOT/'final_test_data/test_network_*.npz',ROOT/'rescue/final_test_data/test_network_*.npz']
rows=load_files(patterns)
# unique examples can overlap? no
# Template from training only.
train=[r for r in rows if r['split']=='train']
template=np.mean([r['target_delta'] for r in train],axis=0)
template/=max(np.linalg.norm(template),1e-30)

# annotate targets
for r in rows:
    truth=.5*(r['truth_half1']+r['truth_half2'])
    e=truth-r['sample_prediction']
    corr_template=template@r['beta_bar']
    den=float(corr_template@corr_template)
    s=float(corr_template@e/max(den,1e-30))
    r['template_scale']=s
    r['template_sign']=1 if s>0 else 0
    r['delta_template']=float(r['target_delta']@template)
    r['corr_norm']=float(np.linalg.norm(r['target_correction']))
    r['delta_norm']=float(np.linalg.norm(r['target_delta']))
    r['oracle_gain']=float(np.mean((r['sample_prediction']-truth)**2)/max(np.mean((r['oracle_prediction']-truth)**2),1e-30))

# feature families
families={}
Xg=np.stack([r['global_features'] for r in rows])
T=np.stack([r['token_features'] for r in rows]) # n,32,101
# scalar columns 0:21; node 21:41; aggregate 41:101
families['weight_global']=Xg
families['token_scalar_summary']=np.concatenate([T[:,:,:21].mean(1),T[:,:,:21].std(1),T[:,:,:21].min(1),T[:,:,:21].max(1)],axis=1)
families['node_summary']=np.concatenate([T[:,:,21:41].mean(1),T[:,:,21:41].std(1)],axis=1)
families['cross_output_aggregate']=np.concatenate([T[:,:,41:].mean(1),T[:,:,41:].std(1)],axis=1)
families['all_token_pooled']=np.concatenate([T.mean(1),T.std(1),T.min(1),T.max(1)],axis=1)
families['q_anchor_stats']=np.stack([np.r_[r['q_anchor'], np.mean(r['q_anchor']),np.std(r['q_anchor']),np.linalg.norm(r['q_anchor'])] for r in rows])
families['same_cloud_final']=np.stack([np.r_[r['sample_prediction'],r['baseline_prediction'],r['sample_prediction']-r['baseline_prediction']] for r in rows])
families['all_observable']=np.concatenate([families['weight_global'],families['all_token_pooled'],families['q_anchor_stats'],families['same_cloud_final']],axis=1)

y=np.array([r['template_scale'] for r in rows])
ydelta=np.array([r['delta_template'] for r in rows])
signs=(y>0).astype(int)
ids=np.array([r['network_id'] for r in rows]);rots=np.array([r['rotation_seed'] for r in rows]);splits=np.array([r['split'] for r in rows])

# Evaluate strict preexisting split: train -> validation, terminal1, terminal2.
cohorts={'validation':(splits=='validation'),'primary_terminal':np.array(['/final_test_data/' in r['source'] and '/rescue/' not in r['source'] for r in rows]),'rescue_terminal':np.array(['/rescue/final_test_data/' in r['source'] for r in rows])}
trainmask=splits=='train'
results={}
for name,X in families.items():
    fr={}
    for c,mask in cohorts.items():
        model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        model.fit(X[trainmask],y[trainmask]);pred=model.predict(X[mask])
        p=pearsonr(pred,y[mask]).statistic if np.std(pred)>0 and np.std(y[mask])>0 else float('nan')
        sp=spearmanr(pred,y[mask]).statistic if np.std(pred)>0 and np.std(y[mask])>0 else float('nan')
        r2=r2_score(y[mask],pred)
        acc=float(np.mean((pred>0)==(y[mask]>0)))
        # delta target too
        md=make_pipeline(StandardScaler(),Ridge(alpha=100.0));md.fit(X[trainmask],ydelta[trainmask]);pd=md.predict(X[mask])
        pdcor=pearsonr(pd,ydelta[mask]).statistic if np.std(pd)>0 else float('nan')
        fr[c]={'n':int(mask.sum()),'scale_pearson':float(p),'scale_spearman':float(sp),'scale_r2':float(r2),'sign_accuracy':acc,'delta_template_pearson':float(pdcor)}
    results[name]=fr

# Within-base rotation decomposition.
from collections import defaultdict
byid=defaultdict(list)
for r in rows:byid[r['network_id']].append(r)
within=[]
for nid,rr in byid.items():
    if len(rr)<2:continue
    vals=np.array([x['template_scale'] for x in rr])
    dvals=np.array([x['delta_template'] for x in rr])
    within.append({'network_id':nid,'n':len(rr),'scale_mean':float(vals.mean()),'scale_std':float(vals.std()),'scale_range':float(np.ptp(vals)),'sign_changes':bool(np.any(vals>0) and np.any(vals<0)),'delta_mean':float(dvals.mean()),'delta_std':float(dvals.std()),'rotations':[x['rotation_seed'] for x in rr],'scales':vals.tolist()})

# pairwise rotation-difference predictability: train only fit, test cohorts. For each family difference relative first rotation per base.
def pair_rows(mask):
    pairs=[]
    for nid in sorted(set(ids[mask])):
        ix=np.where(mask & (ids==nid))[0]
        if len(ix)<2:continue
        for a in range(len(ix)):
            for b in range(a+1,len(ix)):
                pairs.append((ix[a],ix[b]))
    return pairs
trainpairs=pair_rows(trainmask)
pairres={}
for name,X in families.items():
    if not trainpairs: continue
    Xtr=np.stack([X[a]-X[b] for a,b in trainpairs]);ytr=np.array([y[a]-y[b] for a,b in trainpairs])
    pr={}
    for c,mask in cohorts.items():
        pp=pair_rows(mask)
        Xt=np.stack([X[a]-X[b] for a,b in pp]);yt=np.array([y[a]-y[b] for a,b in pp])
        m=make_pipeline(StandardScaler(),Ridge(alpha=100.0));m.fit(Xtr,ytr);yp=m.predict(Xt)
        corr=pearsonr(yp,yt).statistic if np.std(yp)>0 else float('nan')
        acc=float(np.mean(np.sign(yp)==np.sign(yt)))
        pr[c]={'n_pairs':len(pp),'difference_pearson':float(corr),'difference_sign_accuracy':acc}
    pairres[name]=pr

# Direct univariate correlation scan on terminal combined, with BH-ish ranking, train and terminal stability.
feature_names=[];Xscan=[]
# token pooled labeled
for fam in ['token_scalar_summary','node_summary','cross_output_aggregate','q_anchor_stats','same_cloud_final']:
    X=families[fam]
    for j in range(X.shape[1]):
        feature_names.append(f'{fam}:{j}');Xscan.append(X[:,j])
Xscan=np.stack(Xscan,axis=1)
term=(cohorts['primary_terminal']|cohorts['rescue_terminal'])
scan=[]
for j,nm in enumerate(feature_names):
    a=Xscan[trainmask,j];b=Xscan[term,j]
    ct=pearsonr(a,y[trainmask]).statistic if np.std(a)>0 else 0
    ce=pearsonr(b,y[term]).statistic if np.std(b)>0 else 0
    # rotation difference corr terminal
    pp=pair_rows(term);xd=np.array([Xscan[a,j]-Xscan[b,j] for a,b in pp]);yd=np.array([y[a]-y[b] for a,b in pp])
    cd=pearsonr(xd,yd).statistic if np.std(xd)>0 else 0
    scan.append({'feature':nm,'train_corr':float(ct),'terminal_corr':float(ce),'terminal_rotation_diff_corr':float(cd),'stable_score':float(np.sign(ct)==np.sign(ce))*min(abs(ct),abs(ce))})
scan=sorted(scan,key=lambda q:max(abs(q['terminal_corr']),abs(q['terminal_rotation_diff_corr'])),reverse=True)

summary={
'n_rows':len(rows),'n_base_networks':len(byid),'split_counts':{s:int(np.sum(splits==s)) for s in sorted(set(splits))},
'template_scale_distribution':{'mean':float(y.mean()),'std':float(y.std()),'positive_fraction':float(np.mean(y>0))},
'within_rotation':{'n_bases':len(within),'sign_change_fraction':float(np.mean([r['sign_changes'] for r in within])),'median_scale_std':float(np.median([r['scale_std'] for r in within])),'median_scale_range':float(np.median([r['scale_range'] for r in within]))},
'family_prediction':results,'rotation_difference_prediction':pairres,'top_univariate':scan[:80],
'per_base_rotation':within,
}
(OUT/'phase_identifiability_audit.json').write_text(json.dumps(summary,indent=2))
# compact report
lines=['# K32 Phase Identifiability Audit','',f"Rows: {len(rows)}; bases: {len(byid)}",'',f"Within-base sign changes: {summary['within_rotation']['sign_change_fraction']:.3f}",f"Median within-base scale SD: {summary['within_rotation']['median_scale_std']:.4g}",'','## Frozen family results','']
for fam,rr in results.items():
    z=rr['primary_terminal'];q=pairres[fam]['primary_terminal'];lines.append(f"- {fam}: terminal scale r={z['scale_pearson']:.3f}, sign={z['sign_accuracy']:.3f}; rotation-difference r={q['difference_pearson']:.3f}, sign={q['difference_sign_accuracy']:.3f}")
lines+=['','## Top stable/univariate diagnostics','']
for x in scan[:20]:lines.append(f"- {x['feature']}: train r={x['train_corr']:.3f}, terminal r={x['terminal_corr']:.3f}, rotation-diff r={x['terminal_rotation_diff_corr']:.3f}")
(OUT/'PHASE_IDENTIFIABILITY_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['n_rows','n_base_networks','split_counts','template_scale_distribution','within_rotation']},indent=2))
for fam in results:
 print(fam,results[fam]['primary_terminal'],pairres[fam]['primary_terminal'])
