#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from nested_partial_mub_experiment import lower_anchor, cosine, mse
ROOT=HERE.parent
DEV=set(range(5000,5009));CAL=set(range(5009,5013));VAL=set(range(5013,5018))
SPLITS={**{n:'development' for n in DEV},**{n:'calibration' for n in CAL},**{n:'validation' for n in VAL}}
PREFIXES=[2,4,8,12,17]

def beta_bar(z):
    fs=z['fit_fold_sizes'].astype(float);return np.tensordot(fs/fs.sum(),z['fit_betas'],axes=(0,0))

def bounded_aitken(c8,c12,c17):
    d1=c12-c8;d2=c17-c12;r=float(d2@d1/max(d1@d1,1e-300));g=float(np.clip(r/max(1-r,1e-12),-.5,.5)) if r<.999999 else .5
    return c17+g*d2,g,r

def bounded_richardson(c8,c12,c17):
    x=np.array([1/8,1/12,1/17.],float);X=np.c_[np.ones(3),x];w=np.linalg.pinv(X)[0]
    raw=w[0]*c8+w[1]*c12+w[2]*c17;delta=raw-c17;cap=.5*np.linalg.norm(c17);dn=np.linalg.norm(delta)
    if dn>cap>0:delta*=cap/dn
    return c17+delta,w,float(dn/max(np.linalg.norm(c17),1e-300))

def geom_orders():
    ids=list(range(17));chirps=np.stack([fr.kerdock_chirp(i) for i in ids]);bits=chirps<0
    selected=[0];remaining=set(ids[1:])
    while remaining:
        def score(j):return min(np.count_nonzero(bits[j]!=bits[s]) for s in selected)
        j=max(remaining,key=lambda q:(score(q),-q));selected.append(j);remaining.remove(j)
    inter=[]
    for j in range(9):
        if j<17-j:inter.extend([j,16-j])
        elif j==17-j:inter.append(j)
    def brev(i):return int(f'{i:05b}'[::-1],2)
    bitrev=sorted(ids,key=lambda i:(brev(i),i))
    return {'natural':ids,'reverse':ids[::-1],'endpoint_interleave':inter[:17],'geometry_maximin':selected,'bit_reversal':bitrev}

def correction_for_mean(z,mu):
    a=lower_anchor(z['sample_center'],mu,z['pair_scaled'],z['probe_indices'],z['probe_directions'])
    return .2*(a@beta_bar(z))

def rich_features(z, rec, corrs, order_name='natural', order=None):
    if order is not None:
        ext=z['ext_block_means'][order];cum=np.cumsum(ext,axis=0)/np.arange(1,18)[:,None];corrs=np.stack([correction_for_mean(z,mu) for mu in cum])
    cs=[corrs[k-1] for k in PREFIXES];c2,c4,c8,c12,c17=cs
    def coss(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-300))
    inc=[c4-c2,c8-c4,c12-c8,c17-c12];B=beta_bar(z);anchors=z['anchors'] if order is None else None
    if anchors is None:
        ext=z['ext_block_means'][order];cum=np.cumsum(ext,axis=0)/np.arange(1,18)[:,None];anchors=np.stack([lower_anchor(z['sample_center'],mu,z['pair_scaled'],z['probe_indices'],z['probe_directions']) for mu in cum])
    aa=[anchors[k-1] for k in PREFIXES];f={'order':order_name}
    for k,c in zip(PREFIXES,cs):
        f[f'proj{k}_on17']=float(c@c17/max(c17@c17,1e-300));f[f'norm{k}_rel17']=float(np.linalg.norm(c)/max(np.linalg.norm(c17),1e-300))
    for i,d in enumerate(inc):
        f[f'incnorm{i}']=float(np.linalg.norm(d)/max(np.linalg.norm(c17),1e-300));f[f'incproj{i}']=float(d@c17/max(c17@c17,1e-300))
    f.update({'cos12_17':coss(c12,c17),'cos8_12':coss(c8,c12),'cos4_8':coss(c4,c8),'min_succ_cos':min(coss(cs[i],cs[i+1]) for i in range(4)),'late_rel':float(np.linalg.norm(c17-c12)/max(np.linalg.norm(c17),1e-300)),'mid_rel':float(np.linalg.norm(c12-c8)/max(np.linalg.norm(c17),1e-300)),'early_rel':float(np.linalg.norm(c8-c4)/max(np.linalg.norm(c17),1e-300)),'inc_cos_late':coss(inc[-1],inc[-2]),'curvature':float(np.linalg.norm(inc[-1]-inc[-2])/max(np.linalg.norm(c17),1e-300))})
    ext=z['ext_block_means'];mu17=ext.mean(0);scatter=np.linalg.norm(ext-mu17,axis=1)
    f['block_scatter_cv']=float(scatter.std()/max(scatter.mean(),1e-300));f['last5_mean_shift']=float(np.linalg.norm(ext[12:].mean(0)-ext[:12].mean(0))/max(np.linalg.norm(z['sample_center']-mu17),1e-300))
    for nm,av in [('c17',aa[-1]),('late',aa[-1]-aa[-2]),('mid',aa[-2]-aa[-3])]:
        contrib=.2*av[:,None]*B;norms=np.linalg.norm(contrib,axis=1);total=np.linalg.norm(contrib.sum(0));totv=contrib.sum(0);signed=contrib@totv
        f[f'{nm}_l1_over_total']=float(norms.sum()/max(total,1e-300));f[f'{nm}_max_share']=float(norms.max()/max(norms.sum(),1e-300));f[f'{nm}_top5_share']=float(np.sort(norms)[-5:].sum()/max(norms.sum(),1e-300));f[f'{nm}_negative_share']=float(np.abs(signed[signed<0]).sum()/max(np.abs(signed).sum(),1e-300))
    aw=np.linalg.norm(B,axis=1);A=np.stack(aa);flips=np.sum(np.sign(A[1:])!=np.sign(A[:-1]),axis=0)
    f['probe_flip_weighted']=float(np.sum(aw*flips)/max(np.sum(aw)*4,1e-300));f['probe_late_rel_weighted']=float(np.sqrt(np.sum(aw*(aa[-1]-aa[-2])**2)/max(np.sum(aw*aa[-1]**2),1e-300)))
    pc=z['paired_correction'];f['paired_cos']=coss(pc,c17);f['paired_norm_ratio']=float(np.linalg.norm(pc)/max(np.linalg.norm(c17),1e-300))
    loo=z['loo_corrections'];infl=np.linalg.norm(loo-c17,axis=1)/max(np.linalg.norm(c17),1e-300);f['jack_max']=float(infl.max());f['jack_cv']=float(infl.std()/max(infl.mean(),1e-300));f['jack_last5']=float(infl[12:].mean()/max(infl[:12].mean(),1e-300))
    return f,corrs

def group_bootstrap(rows,cand,draws=2000,seed=20260729):
    rng=np.random.default_rng(seed);groups=sorted(set(r['network_id'] for r in rows));by={g:[r for r in rows if r['network_id']==g] for g in groups};vals=[]
    for _ in range(draws):
        samp=rng.choice(groups,len(groups),replace=True);num=den=0.
        for g in samp:
            for r in by[int(g)]:num+=r['candidate_sse'][cand];den+=r['base_sse']
        vals.append(num/max(den,1e-300))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def metrics(rows,cand):
    ratios=np.array([r['candidate_sse'][cand]/r['base_sse'] for r in rows]);num=sum(r['candidate_sse'][cand] for r in rows);den=sum(r['base_sse'] for r in rows);corr=[r['candidate_corr'][cand] for r in rows];ideal=[r['ideal'] for r in rows];err=[r['reduced_base']-r['truth'] for r in rows]
    ips=[float(e@c/len(c)) for e,c in zip(err,corr)];norms=[float(np.linalg.norm(c)) for c in corr];coss=[cosine(c,i) for c,i in zip(corr,ideal)]
    unum=sum(r['candidate_sse_unbiased'][cand] for r in rows);uden=sum(r['base_sse_unbiased'] for r in rows)
    return {'candidate':cand,'pooled_ratio':float(num/max(den,1e-300)),'pooled_ratio_unbiased':float(unum/max(uden,1e-300)),'mean_ratio':float(ratios.mean()),'wins':int(np.sum(ratios<1)),'n':len(rows),'median':float(np.median(ratios)),'p90':float(np.quantile(ratios,.9)),'worst':float(ratios.max()),'grouped_bootstrap_95':group_bootstrap(rows,cand),'mean_error_correction_inner_product':float(np.mean(ips)),'mean_correction_norm':float(np.mean(norms)),'mean_correction_cosine':float(np.mean(coss)),'extra_trajectory_flops':0,'diagnostic_flops_note':'No added propagation; reductions are sub-million-scale. Official FlopScope not available in this sandbox.','split_provenance':sorted(set(r['split'] for r in rows))}

def auc_rank(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float);pos=x[y];neg=x[~y]
    if len(pos)==0 or len(neg)==0:return float('nan')
    a=(sum((p>neg).sum()+.5*(p==neg).sum() for p in pos)/(len(pos)*len(neg)));return float(max(a,1-a))

def detection(flag,severe):
    flag=np.asarray(flag,bool);severe=np.asarray(severe,bool);return {'catch':float(np.sum(flag&severe)/max(np.sum(severe),1)),'false_suppression':float(np.sum(flag&~severe)/max(np.sum(~severe),1)),'flagged':int(flag.sum()),'severe':int(severe.sum()),'nonsevere':int((~severe).sum())}

def main():
    t0=time.time();records=[];feat_rows=[];orders=geom_orders();order_rows=[]
    for rp in sorted((ROOT/'results'/'records').glob('network_*_rotation_*.json')):
        rec=json.loads(rp.read_text());z=np.load(ROOT/'results'/'vectors'/rec['vectors_file']);truth=z['truth_y'].astype(float);ref1=z['ref1_y'].astype(float);ref2=z['ref2_y'].astype(float);base=z['basefull'].astype(float);y0=z['reduced_base'].astype(float);corrs=z['corrections'].astype(float);den=float(np.sum((base-truth)**2));uden=float(np.sum((base-ref1)*(base-ref2)));ideal=truth-y0
        c8,c12,c17=corrs[7],corrs[11],corrs[16];ait,g,r=bounded_aitken(c8,c12,c17);rich,w,rich_raw=bounded_richardson(c8,c12,c17)
        cw=(8*c8+12*c12+17*c17)/37.;iv=(64*c8+144*c12+289*c17)/(64+144+289);choose=c12 if np.linalg.norm(c17-c12)>0.5*np.linalg.norm(c17) else c17
        cand={'zero':np.zeros_like(c17),'c2':corrs[1],'c4':corrs[3],'c8':c8,'c12':c12,'c17':c17,'choose_c12_if_unstable':choose,'count_weighted_8_12_17':cw,'inverse_variance_proxy':iv,'bounded_aitken':ait,'bounded_richardson':rich,'trimmed_basis_mean':z['robust_corrections'][0],'median_of_groups':z['robust_corrections'][1],'paired2_substitute':z['paired_correction']}
        f,_=rich_features(z,rec,corrs);f.update({'network_id':rec['network_id'],'rotation_index':rec['rotation_index'],'split':SPLITS[rec['network_id']],'c17_ratio':float(np.sum((y0+c17-truth)**2)/den),'reduced_ratio':float(np.sum((y0-truth)**2)/den),'reference_noise_fraction':rec['reference_noise_fraction'],'aitken_gamma':g,'aitken_r':r})
        # frozen flags
        flags={'rule_A':f['jack_cv']>=.33 or f['probe_late_rel_weighted']<=.50,'rule_B':f['c17_top5_share']>=.255 or f['mid_l1_over_total']<=4.04,'rule_C':f['cos12_17']>=.963 or f['jack_cv']>=.33,'rule_D_paired':f['paired_cos']<=-.05}
        for n,fl in flags.items():cand[n]=np.zeros_like(c17) if fl else c17;f[n+'_flag']=fl
        # oracle ceilings
        candidate_ratios={n:float(np.sum((y0+c-truth)**2)/den) for n,c in cand.items()};bestname=min(candidate_ratios,key=candidate_ratios.get);cand['oracle_nested_chooser']=cand[bestname];cand['oracle_benefit_gate']=np.zeros_like(c17) if candidate_ratios['zero']<candidate_ratios['c17'] else c17
        row={'network_id':rec['network_id'],'rotation_index':rec['rotation_index'],'split':SPLITS[rec['network_id']],'truth':truth,'reduced_base':y0,'ideal':ideal,'base_sse':den,'base_sse_unbiased':uden,'candidate_corr':cand,'candidate_sse':{n:float(np.sum((y0+c-truth)**2)) for n,c in cand.items()},'candidate_sse_unbiased':{n:float(np.sum((y0+c-ref1)*(y0+c-ref2))) for n,c in cand.items()},'runtime_seconds':rec['runtime_seconds'],'peak_rss_kb':rec['peak_rss_kb']};records.append(row);feat_rows.append(f)
        for oname,order in orders.items():
            of,oc=rich_features(z,rec,corrs,oname,order);order_rows.append({'network_id':rec['network_id'],'rotation_index':rec['rotation_index'],'split':SPLITS[rec['network_id']],'order':oname,'c2_rel17':float(np.linalg.norm(oc[1]-oc[16])/max(np.linalg.norm(oc[16]),1e-300)),'c4_rel17':float(np.linalg.norm(oc[3]-oc[16])/max(np.linalg.norm(oc[16]),1e-300)),'c8_rel17':float(np.linalg.norm(oc[7]-oc[16])/max(np.linalg.norm(oc[16]),1e-300)),'c12_rel17':float(np.linalg.norm(oc[11]-oc[16])/max(np.linalg.norm(oc[16]),1e-300)),'cos12_17':of['cos12_17'],'late_rel':of['late_rel'],'jack_cv':of['jack_cv'],'probe_late_rel_weighted':of['probe_late_rel_weighted'],'c17_ratio':f['c17_ratio']})
    feat=pd.DataFrame(feat_rows);ordf=pd.DataFrame(order_rows);feat.to_csv(ROOT/'results'/'ROW_LEVEL_RESULTS.csv',index=False);ordf.to_csv(ROOT/'results'/'ORDERING_DIAGNOSTICS.csv',index=False)
    candidates=list(records[0]['candidate_corr']);all_metrics=[]
    for split in ['development','calibration','validation','all']:
        rr=records if split=='all' else [r for r in records if r['split']==split]
        for c in candidates:
            m=metrics(rr,c);m['evaluation_split']=split;all_metrics.append(m)
    mdf=pd.DataFrame(all_metrics);mdf.to_csv(ROOT/'results'/'CANDIDATE_METRICS.csv',index=False)
    # Detection, feature AUC, and ordering audits.
    detection_out={};auc_out={}
    for split in ['development','calibration','validation','all']:
        q=feat if split=='all' else feat[feat.split==split];sev=q.c17_ratio>1.10
        detection_out[split]={n:detection(q[n+'_flag'],sev) for n in ['rule_A','rule_B','rule_C','rule_D_paired']}
        cols=['cos12_17','late_rel','jack_cv','probe_late_rel_weighted','c17_top5_share','mid_l1_over_total','paired_cos','block_scatter_cv','curvature','min_succ_cos']
        auc_out[split]={c:auc_rank(sev,q[c]) for c in cols}
    ordering={}
    for split in ['development','calibration','validation','all']:
        q=ordf if split=='all' else ordf[ordf.split==split];ordering[split]={}
        for oname in orders:
            u=q[q.order==oname];ordering[split][oname]={'mean_c8_rel17':float(u.c8_rel17.mean()),'mean_c12_rel17':float(u.c12_rel17.mean()),'median_cos12_17':float(u.cos12_17.median()),'late_rel_auc_for_severe':auc_rank(u.c17_ratio>1.10,u.late_rel)}
    # Rotation stability.
    mixed={};flag_stability={}
    for split in ['development','calibration','validation','all']:
        q=feat if split=='all' else feat[feat.split==split];g=q.groupby('network_id');mixed[split]=float(np.mean([(x.c17_ratio.gt(1.10).nunique()>1) for _,x in g]))
        flag_stability[split]={n:float(np.mean([(x[n+'_flag'].nunique()==1) for _,x in g])) for n in ['rule_A','rule_B','rule_C','rule_D_paired']}
    # Core answers.
    val=feat[feat.split=='validation'];sev=val.c17_ratio>1.10;non=~sev
    conv={'validation_median_cos12_17_all':float(val.cos12_17.median()),'validation_median_cos12_17_severe':float(val.loc[sev,'cos12_17'].median()) if sev.any() else None,'validation_median_cos12_17_nonsevere':float(val.loc[non,'cos12_17'].median()) if non.any() else None,'validation_median_norm12_rel17':float(val.norm12_rel17.median()),'validation_mixed_severity_base_fraction':mixed['validation']}
    valm=mdf[mdf.evaluation_split=='validation'].set_index('candidate').to_dict('index')
    summary={'protocol':'nested-partial-mub-uncertainty-v1','status':'CLOSE nested convergence as a standalone free safety certificate','records':len(records),'base_networks':len(set(r['network_id'] for r in records)),'rotations_per_base':3,'splits':{s:int(sum(r['split']==s for r in records)) for s in ['development','calibration','validation']},'reference_samples_per_half':196608,'calibration_selection':'none','gate':{'required_catch':.75,'required_max_false_suppression':.20,'passed':False},'detection':detection_out,'feature_auc':auc_out,'ordering':ordering,'rotation_mixed_severity_fraction':mixed,'flag_rotation_stability':flag_stability,'convergence':conv,'validation_metrics':valm,'runtime':{'mean_record_seconds':float(np.mean([r['runtime_seconds'] for r in records])),'peak_rss_kb':int(max(r['peak_rss_kb'] for r in records)),'analysis_seconds':float(time.time()-t0)},'geometry_orders':orders}
    (ROOT/'results'/'RESULTS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Figures.
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.figure(figsize=(7,5));
    for split,mark in [('development','o'),('calibration','s'),('validation','^')]:
        q=feat[feat.split==split];plt.scatter(q.cos12_17,q.c17_ratio,label=split,marker=mark,alpha=.8)
    plt.axhline(1.10,linestyle='--');plt.xlabel('cos(c12, c17)');plt.ylabel('c17 / full-baseline MSE');plt.legend();plt.tight_layout();plt.savefig(ROOT/'figures'/'cos12_17_vs_ratio.png',dpi=180);plt.close()
    plt.figure(figsize=(7,5));
    for oname in orders:
        q=ordf[(ordf.split=='validation')&(ordf.order==oname)];plt.plot([2,4,8,12],[q.c2_rel17.mean(),q.c4_rel17.mean(),q.c8_rel17.mean(),q.c12_rel17.mean()],marker='o',label=oname)
    plt.xlabel('prefix bases');plt.ylabel('mean relative distance to c17');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(ROOT/'figures'/'ordering_convergence.png',dpi=180);plt.close()
    # Markdown report.
    vm=valm;detv=detection_out['validation'];dcal=detection_out['calibration'];
    report=f'''# Nested Partial-MUB Convergence as a Free Uncertainty Signal\n\n**Date:** 2026-07-29  \n**Status:** **CLOSE as a standalone safety certificate.**  \n**Protected/official holdout:** not opened.\n\n## Executive result\n\nA fresh exact-geometry width-256 experiment used **18 new base networks × 3 predetermined rotations = 54 records**, with all rotations grouped by base network. The frozen 112+17 construction accumulated `c2,c4,c8,c12,c17` in one companion pass. Final targets were uniformly refined to two independent **196,608-sample** aggregates per base network.\n\nAt selection time, the development-selected nested rule failed calibration at **40.0% catch / 71.4% false suppression**, so no rule was selected. After uniform reference refinement, the same calibration block is even worse: **{dcal['rule_A']['catch']:.1%} catch / {dcal['rule_A']['false_suppression']:.1%} false suppression**. On validation, the same frozen rule had catch **{detv['rule_A']['catch']:.1%}** and false suppression **{detv['rule_A']['false_suppression']:.1%}**. It does not meet the required 75% / 20% gate.\n\nThe central negative finding is mechanistic: severe cases often have internally smooth nested convergence. Validation median `cos(c12,c17)` was **{conv['validation_median_cos12_17_all']:.3f}** overall and **{conv['validation_median_cos12_17_severe']:.3f}** on severe records. The trajectory can converge coherently to the wrong externally phased answer.\n\n## Final-output candidates on validation\n\n| Candidate | Pooled ratio | Unbiased pooled | Wins | Median | p90 | Worst | Mean correction cosine |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''
    for c in ['zero','c8','c12','c17','count_weighted_8_12_17','bounded_aitken','bounded_richardson','trimmed_basis_mean','median_of_groups','paired2_substitute','rule_A','rule_D_paired','oracle_benefit_gate','oracle_nested_chooser']:
        x=vm[c];report+=f"| {c} | {x['pooled_ratio']:.4f} | {x['pooled_ratio_unbiased']:.4f} | {x['wins']}/{x['n']} | {x['median']:.4f} | {x['p90']:.4f} | {x['worst']:.4f} | {x['mean_correction_cosine']:.3f} |\n"
    report+=f'''\nThe fixed `c17` package itself scored **{vm['c17']['pooled_ratio']:.4f}** by mean-target MSE and **{vm['c17']['pooled_ratio_unbiased']:.4f}** by the independent-half unbiased estimator on this synthetic validation block; the no-correction reduced 112-basis arm scored **{vm['zero']['pooled_ratio']:.4f}**. This cohort is diagnostic, not a replacement for the canonical partial-MUB validation.\n\n## Priority questions resolved\n\n1. **Do catastrophic tails show nonconvergence?** Not reliably. Smooth late convergence is common in both good and bad records; severe records remain after `c12` and `c17` are highly aligned.\n2. **Does direction stabilize earlier than amplitude?** Direction relative to the terminal estimate stabilizes earlier, but terminal direction itself is not an absolute-truth certificate. Median validation `||c12||/||c17||` is **{conv['validation_median_norm12_rel17']:.3f}**.\n3. **Are later bases signal or variance?** `c8`, `c12`, and `c17` show gradual average convergence, but later bases do not monotonically improve every rotation. Robust trimming, median groups, Aitken, and Richardson do not remove the tail.\n4. **Does basis ordering matter?** It changes early-prefix smoothness but not `c17`. The target-free geometry-maximin order is reported, but no ordering produces a transferable severe-tail certificate.\n5. **Are patterns stable across rotations?** No. **{mixed['validation']:.1%}** of validation base networks have mixed severe/nonsevere outcomes across their three rotations.\n6. **Does per-probe instability find harmful modes?** The strongest validation per-probe concentration AUC was {auc_out['validation']['c17_top5_share']:.3f}, but its development behavior did not transfer into a calibrated rule.\n7. **Can nested estimates replace a paired probe?** No. Nested-only rules fail. The already-free two-basis original/external difference is weak as a certificate too; its frozen disagreement rule catches only **{detv['rule_D_paired']['catch']:.1%}** of severe validation records.\n\n## Compute and implementation\n\nAll nested prefixes are reductions of the same 17 propagated blocks. They add **zero trajectories**. The candidate diagnostics require only prefix sums, 128-probe reductions, and leave-one-basis recombinations. The sandbox did not provide official FlopScope, so this report does not claim official tracked FLOPs. Mean per-record diagnostic harness time was {summary['runtime']['mean_record_seconds']:.2f}s and peak process RSS was {summary['runtime']['peak_rss_kb']/1024:.1f} MiB; these include research materialization and are not subprocess-package timings.\n\n## Decision\n\nClose nested convergence as a standalone free uncertainty signal. Retain the prefix arrays and per-basis influences as low-cost features for Priority 1 paired safety or a later tiny residual gate, but do not suppress, shrink, extrapolate, or choose prefixes based on nested convergence alone. The failure mode is coherent external angular bias, not merely finite-prefix variance.\n\n## Provenance\n\n- Development: base networks 5000–5008.\n- Calibration: 5009–5012.\n- Untouched-until-final validation: 5013–5017.\n- Three predetermined rotations per base; all grouped by base network.\n- Rule thresholds frozen before calibration; calibration selected no rule; validation did not change the decision.\n- Full row-level vectors, records, freeze hashes, reference seeds, candidate tables, ordering diagnostics, and code are included.\n'''
    (ROOT/'REPORT.md').write_text(report)
    # ledger line and hashes
    pd.DataFrame([{'ID':'M129-NESTED','Evidence level':'Fresh exact-geometry synthetic grouped validation','Family':'Legal signed anchor','Experiment':'Nested partial-MUB convergence as free uncertainty','Result':f"No frozen rule passed calibration; validation c17={vm['c17']['pooled_ratio']:.4f}; coherent wrong-bias cases remain",'Verdict':'Close standalone nested certificate; retain features only','Status':'Closed'}]).to_csv(ROOT/'EXPERIMENT_LEDGER_ADDITION.csv',index=False)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'];lines=[]
    for p in sorted(files):lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)))
    (ROOT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':summary['status'],'validation_c17':vm['c17']['pooled_ratio'],'validation_detection':detv,'report':str(ROOT/'REPORT.md')},indent=2))
if __name__=='__main__':main()
