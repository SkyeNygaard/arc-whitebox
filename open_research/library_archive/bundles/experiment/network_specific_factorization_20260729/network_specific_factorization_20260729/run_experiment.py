#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

DEFAULT_SOURCE = Path('/mnt/data/exp3_work/legal_signed_anchor_continuation_20260729/legal_signed_anchor_continuation_20260729/continue_path1/02_centered_analytic_closures/src')
D=256
RANKS=[1,2,4,8,12,16,30,64,128]
ROTATIONS=[3,11,97]
TRAIN_IDS=list(range(4300,4316))
VALID_IDS=list(range(4400,4424))


def canonicalize(V: np.ndarray) -> np.ndarray:
    V=V.copy()
    for j in range(V.shape[1]):
        i=int(np.argmax(np.abs(V[:,j])))
        if V[i,j] < 0: V[:,j]*=-1
    return V

def orth(A: np.ndarray, r: int) -> np.ndarray:
    if A.size==0: return np.zeros((D,0))
    q,_=np.linalg.qr(A)
    return canonicalize(q[:,:min(r,q.shape[1])])

def svd_right(A: np.ndarray, r: int) -> np.ndarray:
    if not np.any(np.isfinite(A)) or np.linalg.norm(A)==0: return np.eye(A.shape[1],r)
    _,_,vt=np.linalg.svd(A,full_matrices=False)
    return canonicalize(vt[:r].T)

def make_kerdock_rot(fr, seed:int)->np.ndarray:
    radius=fr.chi_mean(fr.D); H=fr.walsh_hadamard()/math.sqrt(fr.D); rotation=fr.haar_rotation(seed)
    blocks=[]
    for u in range(128):
        chirp=fr.kerdock_chirp(u); basis=(H*chirp[None,:])@rotation
        blocks.extend([(radius*basis).astype(np.float32),(-radius*basis).astype(np.float32)])
    coordinate=(radius*rotation).astype(np.float32); blocks.extend([coordinate,-coordinate])
    return np.concatenate(blocks,axis=0)

def mse(x,y): return float(np.mean((x-y)**2))
def cosine(a,b): return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def ratio_metrics(rows: list[dict[str,Any]], key:str, groups:dict[int,list[int]], draws:int=5000)->dict[str,Any]:
    b=np.array([r['baseline_mse'] for r in rows]); c=np.array([r[key+'_mse'] for r in rows]); rr=c/np.maximum(b,1e-300)
    full=np.stack([r['full_correction'] for r in rows]); pred=np.stack([r[key+'_correction'] for r in rows])
    cos=[cosine(a,z) for a,z in zip(pred,full)]
    rng=np.random.default_rng(20260729+sum(map(ord,key)))
    gids=sorted(groups); bs=[]
    for _ in range(draws):
        pick=rng.choice(gids,size=len(gids),replace=True); ix=[i for g in pick for i in groups[g]]
        bs.append(c[ix].sum()/max(b[ix].sum(),1e-300))
    return {'aggregate_ratio':float(c.sum()/b.sum()),'grouped_bootstrap_95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
            'wins':int(np.sum(c<b)),'n':len(rows),'win_rate':float(np.mean(c<b)),'median_ratio':float(np.median(rr)),
            'p90_ratio':float(np.quantile(rr,.9)),'worst_ratio':float(np.max(rr)),'mean_correction_cosine':float(np.mean(cos)),
            'median_correction_cosine':float(np.median(cos))}

def ridge_fit(X,Y,lam):
    Xa=np.c_[np.ones(len(X)),X]
    P=np.eye(Xa.shape[1]); P[0,0]=0
    return np.linalg.solve(Xa.T@Xa+lam*P,Xa.T@Y)
def ridge_pred(X,B): return np.c_[np.ones(len(X)),X]@B

def grouped_cv_lambda(X,Y,network_ids,builders,base,truth,lams):
    unique=sorted(set(network_ids)); best=None
    for lam in lams:
        num=den=0.0
        for gid in unique:
            tr=np.array([g!=gid for g in network_ids]); te=~tr
            B=ridge_fit(X[tr],Y[tr],lam); yp=ridge_pred(X[te],B)
            idx=np.flatnonzero(te)
            for local,i in enumerate(idx):
                corr=builders[i](yp[local]); num+=mse(base[i]+corr,truth[i]); den+=mse(base[i],truth[i])
        val=num/max(den,1e-300)
        if best is None or val<best[0]: best=(val,lam)
    return best

def projector_distance(V,W):
    r=min(V.shape[1],W.shape[1]); return float(r-np.linalg.norm(V[:,:r].T@W[:,:r])**2)
def principal_angles(V,W):
    s=np.linalg.svd(V.T@W,compute_uv=False); s=np.clip(s,-1,1); return np.degrees(np.arccos(s))

def greedy_medoids(Vs,k):
    n=len(Vs); dist=np.zeros((n,n))
    for i in range(n):
        for j in range(i): dist[i,j]=dist[j,i]=projector_distance(Vs[i],Vs[j])
    med=[int(np.argmin(dist.mean(1)))]
    while len(med)<min(k,n):
        nearest=np.min(dist[:,med],axis=1); nearest[med]=-1; med.append(int(np.argmax(nearest)))
    return med,dist

def weight_features(ws):
    out=[]
    for li in [0,7,15,23,29,30,31]:
        w=ws[li].numpy().astype(np.float64)
        s=np.linalg.svd(w,compute_uv=False)
        out += [np.linalg.norm(w)/D,s[0],s[1],s[3],s[7],s[15],s[-1],np.mean(w),np.std(w)]
    prod=ws[30].numpy().astype(np.float64)@ws[31].numpy().astype(np.float64)
    s=np.linalg.svd(prod,compute_uv=False); out += list(s[:16])
    return np.asarray(out)

def sample_features(sample):
    beta=sample['beta']; s_beta=np.linalg.svd(beta,compute_uv=False)
    fy=sample['fold_y']; fx=sample['fold_x']; q=sample['sample_anchor']; pw=np.sum(beta*beta,axis=1)
    def stats(x):
        x=np.asarray(x).ravel(); qs=np.quantile(x,[0,.1,.25,.5,.75,.9,1]); return [np.mean(x),np.std(x),np.linalg.norm(x)/math.sqrt(max(len(x),1)),*qs,float(np.mean(x>0))]
    feats=[]
    feats += list(sample['weight_features'])
    feats += list(s_beta[:24]/max(s_beta[0],1e-30))
    feats += stats(q)+stats(pw)+stats(fx.std(0))+stats(fy.std(0))
    feats += list(np.linalg.svd(fy-fy.mean(0),compute_uv=False)[:5])
    feats += stats(sample['gate30'])+stats(sample['gate31'])+stats(sample['baseline'])
    # Signed mode-specific legal observables in the frozen legal basis.
    V=sample['V_legal_max']; Bm=beta@V
    feats += list(q@Bm)
    feats += list(fx.mean(0)@Bm)
    feats += list(np.sqrt(np.mean((fx-fx.mean(0))**2,axis=0))@np.abs(Bm))
    return np.nan_to_num(np.asarray(feats),nan=0,posinf=0,neginf=0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent); ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--truth-n',type=int,default=4096); ap.add_argument('--chunk',type=int,default=512); ap.add_argument('--threads',type=int,default=16)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(a.source)); import frozen_reference_impl as fr
    torch.set_num_threads(min(a.threads,os.cpu_count() or a.threads)); t_all=time.time()
    xks={r:make_kerdock_rot(fr,r) for r in ROTATIONS}
    samples=[]; row_summaries=[]
    for split,ids in [('train',TRAIN_IDS),('validation',VALID_IDS)]:
      for nid in ids:
        tnet=time.time(); ws,whash,wseed=fr.make_weights(nid); wf=weight_features(ws)
        ref1=fr.stream_reference(ws,a.truth_n,88000000+2*nid,a.chunk); ref2=fr.stream_reference(ws,a.truth_n,88000001+2*nid,a.chunk)
        truth=.5*(ref1['y']+ref2['y']); pooled={k:.5*(ref1[k]+ref2[k]) for k in ref1}; refnoise=.25*np.mean((ref1['y']-ref2['y'])**2)
        for rot in ROTATIONS:
            ts=time.time(); hk,yk=fr.forward_target_final(torch.from_numpy(xks[rot]),ws)
            H=hk.numpy().astype(np.float64); Y=yk.numpy().astype(np.float64); m=H.mean(0); base=Y.mean(0); rho=fr.chi_mean(fr.D)
            Q=fr.sample_anchor_matrix(H,m,rho); idx,dirs=fr.sample_row_probes(Q); X=fr.radial_features_sample_rows(H,m,idx,dirs,rho); fit=fr.fit_crossfit(X,Y)
            sample_M=(fr.D/(rho*rho))*(H.T@H/len(H)); sample_raw=((fr.D+1)/(rho*rho))*((H*H).T@H/len(H))
            comps=fr.anchor_component_matrices(pooled['mu'],pooled['M'],pooled['raw'],m,sample_M,sample_raw)
            delta=fr.contract_rows(comps['lower_only']-Q,idx,dirs); weights=fit['fold_sizes']/fit['fold_sizes'].sum(); beta=np.einsum('f,fpd->pd',weights,fit['betas'])
            C=delta[:,None]*beta; U,S,VT=np.linalg.svd(C,full_matrices=False); Vexact=canonicalize(VT.T); full=C.sum(0)
            # legal subspaces up to max 16
            Vbeta=svd_right(beta,16)
            Vfold=svd_right(fit['fold_y_mean']-fit['fold_y_mean'].mean(0),16)
            z30=H@ws[30].numpy(); a30=np.maximum(z30,0); z31=a30@ws[31].numpy(); g30=(z30>0).mean(0); g31=(z31>0).mean(0)
            A=(ws[30].numpy().astype(np.float64)*g30[None,:])@(ws[31].numpy().astype(np.float64)*g31[None,:])
            Vsoft=svd_right(A,16); Wprod=ws[30].numpy().astype(np.float64)@ws[31].numpy().astype(np.float64); Vweight=svd_right(Wprod,16)
            Vunion=orth(np.c_[Vsoft[:,:8],Vfold[:,:5],Vbeta[:,:8]],16)
            sm={'split':split,'network_id':nid,'rotation':rot,'weight_seed':wseed,'weight_hash':whash,'baseline':base,'truth':truth,'baseline_mse':mse(base,truth),
                'reference_noise_mse':float(refnoise),'full_correction':full,'full_mse':mse(base+full,truth),'C':C,'S':S,'Vexact':Vexact,'beta':beta,
                'sample_anchor':fr.contract_rows(Q,idx,dirs),'fold_x':fit['fold_x_mean'],'fold_y':fit['fold_y_mean'],'gate30':g30,'gate31':g31,
                'V_beta':Vbeta,'V_fold':Vfold,'V_softgate':Vsoft,'V_weightprod':Vweight,'V_union':Vunion,'weight_features':wf}
            samples.append(sm)
            row={'split':split,'network_id':nid,'rotation':rot,'baseline_mse':sm['baseline_mse'],'reference_noise_mse':refnoise,'full_exact_ratio':sm['full_mse']/sm['baseline_mse'],'seconds':time.time()-ts}
            benefit_den=max(sm['baseline_mse']-sm['full_mse'],1e-300)
            for r in RANKS:
                Cr=(U[:,:r]*S[:r])@VT[:r]; c=Cr.sum(0); mr=mse(base+c,truth); row[f'exact_rank{r}_ratio']=mr/sm['baseline_mse']; row[f'exact_rank{r}_benefit_retention']=(sm['baseline_mse']-mr)/benefit_den; row[f'exact_rank{r}_energy']=float(np.sum(S[:r]**2)/np.sum(S**2))
            # First-32 selected-slot diagnostic (not historical K32).
            c32=C[:32].sum(0); row['probe32_ratio']=mse(base+c32,truth)/sm['baseline_mse']; row['probe32_cosine']=cosine(c32,full)
            row_summaries.append(row); print(json.dumps({'split':split,'id':nid,'rot':rot,'sec':round(row['seconds'],2),'full':round(row['full_exact_ratio'],4),'r8':round(row['exact_rank8_ratio'],4)}),flush=True)
            del H,Y,Q,X,sample_M,sample_raw,comps,C,U,S,VT,z30,a30,z31,A; gc.collect()
        del ref1,ref2,pooled,ws; gc.collect(); print(f'network {nid} total {time.time()-tnet:.1f}s',flush=True)
    # Freeze rank by training gate.
    train_rows=[r for r in row_summaries if r['split']=='train']
    frozen_rank=12
    for r in [2,4,8,12]:
        ratios=np.array([x[f'exact_rank{r}_ratio'] for x in train_rows]); b=np.array([x['baseline_mse'] for x in train_rows]); agg=float(np.sum(ratios*b)/np.sum(b))
        if agg<=.595 and np.mean(ratios<1)>=.75 and np.max(ratios)<=1.15: frozen_rank=r; break
    # Attach legal basis at frozen rank; select deterministic mechanism on train with oracle coefficients.
    mechanisms=['V_beta','V_fold','V_softgate','V_weightprod','V_union']
    mech_train={}
    for mech in mechanisms:
        num=den=0
        for s in samples:
            if s['split']!='train': continue
            V=s[mech][:,:frozen_rank]; c=V@(V.T@s['full_correction']); num+=mse(s['baseline']+c,s['truth']); den+=s['baseline_mse']
        mech_train[mech]=num/den
    frozen_mech=min(mech_train,key=mech_train.get)
    for s in samples: s['V_legal_max']=s[frozen_mech][:,:frozen_rank]
    # Pooled exact output basis from training C.
    Cstack=np.concatenate([s['C'] for s in samples if s['split']=='train'],axis=0); Vpool=svd_right(Cstack,max(30,frozen_rank))
    # Exact-subspace codebook medoids on training samples.
    tr_samples=[s for s in samples if s['split']=='train']; va_samples=[s for s in samples if s['split']=='validation']
    exact_train=[s['Vexact'][:,:frozen_rank] for s in tr_samples]; med,dist=greedy_medoids(exact_train,8); codebook=[exact_train[i] for i in med]
    # Features and targets.
    Xall=np.stack([sample_features(s) for s in samples]); mean=Xall[[s['split']=='train' for s in samples]].mean(0); std=Xall[[s['split']=='train' for s in samples]].std(0); std[std<1e-10]=1; Xz=(Xall-mean)/std
    train_ix=np.array([i for i,s in enumerate(samples) if s['split']=='train']); val_ix=np.array([i for i,s in enumerate(samples) if s['split']=='validation'])
    gids=np.array([s['network_id'] for s in samples]); base=[s['baseline'] for s in samples]; truth=[s['truth'] for s in samples]
    Ylegal=np.stack([s['V_legal_max'].T@s['full_correction'] for s in samples]); Yexact=np.stack([s['Vexact'][:,:frozen_rank].T@s['full_correction'] for s in samples]); Yvec=np.stack([s['full_correction'] for s in samples])
    lams=[1e-4,1e-3,1e-2,.1,1,10,100,1000,10000]
    builders_legal=[lambda z,s=s: s['V_legal_max']@z for s in samples]; builders_exact=[lambda z,s=s: s['Vexact'][:,:frozen_rank]@z for s in samples]; builders_vec=[lambda z:z for _ in samples]
    cv_legal,lam_legal=grouped_cv_lambda(Xz[train_ix],Ylegal[train_ix],gids[train_ix],[builders_legal[i] for i in train_ix],[base[i] for i in train_ix],[truth[i] for i in train_ix],lams)
    cv_exact,lam_exact=grouped_cv_lambda(Xz[train_ix],Yexact[train_ix],gids[train_ix],[builders_exact[i] for i in train_ix],[base[i] for i in train_ix],[truth[i] for i in train_ix],lams)
    cv_vec,lam_vec=grouped_cv_lambda(Xz[train_ix],Yvec[train_ix],gids[train_ix],[builders_vec[i] for i in train_ix],[base[i] for i in train_ix],[truth[i] for i in train_ix],lams)
    Blegal=ridge_fit(Xz[train_ix],Ylegal[train_ix],lam_legal); Bexact=ridge_fit(Xz[train_ix],Yexact[train_ix],lam_exact); Bvec=ridge_fit(Xz[train_ix],Yvec[train_ix],lam_vec)
    Plegal=ridge_pred(Xz,Blegal); Pexact=ridge_pred(Xz,Bexact); Pvec=ridge_pred(Xz,Bvec)
    # Frozen template + scalar.
    template=Yvec[train_ix].mean(0); template/=max(np.linalg.norm(template),1e-30); scalar=np.array([s['full_correction']@template for s in samples])[:,None]
    _,lam_scalar=grouped_cv_lambda(Xz[train_ix],scalar[train_ix],gids[train_ix],[lambda z,t=template:t*float(z[0]) for _ in train_ix],[base[i] for i in train_ix],[truth[i] for i in train_ix],lams)
    Bscalar=ridge_fit(Xz[train_ix],scalar[train_ix],lam_scalar); Pscalar=ridge_pred(Xz,Bscalar)[:,0]
    # Add candidates to sample dictionaries.
    for i,s in enumerate(samples):
        Vex=s['Vexact'][:,:frozen_rank]; Vleg=s['V_legal_max']; full=s['full_correction'];
        candidates={
          'zero':np.zeros(D),'full_exact_lower':full,
          f'exact_per_network_rank{frozen_rank}':Vex@(Vex.T@full),
          f'pooled_rank{frozen_rank}':Vpool[:,:frozen_rank]@(Vpool[:,:frozen_rank].T@full),
          f'legal_{frozen_mech}_oracle_coeff':Vleg@(Vleg.T@full),
          f'legal_{frozen_mech}_predicted_coeff':Vleg@Plegal[i],
          f'exact_subspace_predicted_coeff':Vex@Pexact[i],
          'direct_anchor_vector_learner':Pvec[i],
          'frozen_template_learned_scalar':template*Pscalar[i],
        }
        # codebook: oracle closest exact and selector using legal subspace.
        exd=[projector_distance(Vex,V) for V in codebook]; seld=[projector_distance(Vleg,V) for V in codebook]
        Vor=codebook[int(np.argmin(exd))]; Vsel=codebook[int(np.argmin(seld))]
        candidates['codebook_oracle_select_oracle_coeff']=Vor@(Vor.T@full)
        candidates['codebook_legal_select_oracle_coeff']=Vsel@(Vsel.T@full)
        # all deterministic mechanisms with oracle coefficients
        for mech in mechanisms:
            V=s[mech][:,:frozen_rank]; candidates[f'{mech}_oracle_coeff']=V@(V.T@full)
        for key,c in candidates.items(): s[key+'_correction']=c; s[key+'_mse']=mse(s['baseline']+c,s['truth'])
    # Metrics separately on validation.
    val=[s for s in samples if s['split']=='validation']; groups={g:[i for i,s in enumerate(val) if s['network_id']==g] for g in VALID_IDS}
    candidate_keys=[k[:-4] for k in val[0] if k.endswith('_mse') and k not in ['baseline_mse','reference_noise_mse','full_mse']]
    metrics={k:ratio_metrics(val,k,groups) for k in sorted(candidate_keys)}
    # exact rank metrics on validation directly
    for r in RANKS:
        key=f'exact_rank{r}'
        for s,row in zip(samples,row_summaries):
            U,S,VT=np.linalg.svd(s['C'],full_matrices=False); c=((U[:,:r]*S[:r])@VT[:r]).sum(0); s[key+'_correction']=c; s[key+'_mse']=mse(s['baseline']+c,s['truth'])
        metrics[key]=ratio_metrics(val,key,groups)
    # rank threshold distributions and rotation angles.
    thresholds=[.70,.80,.90,.95,.99]; rank_need={str(t):[] for t in thresholds}
    for s in val:
        b=s['baseline_mse']; f=s['full_mse']; den=max(b-f,1e-300)
        U,S,VT=np.linalg.svd(s['C'],full_matrices=False)
        vals=[]
        for r in range(1,129):
            c=((U[:,:r]*S[:r])@VT[:r]).sum(0); vals.append((b-mse(s['baseline']+c,s['truth']))/den)
        for t in thresholds: rank_need[str(t)].append(next((i+1 for i,x in enumerate(vals) if x>=t),129))
    angle_rows=[]
    for nid in VALID_IDS:
        grp=[s for s in val if s['network_id']==nid]
        for i in range(3):
            for j in range(i):
                for name,Vkey in [('exact','Vexact'),('legal',frozen_mech)]:
                    V1=grp[i][Vkey][:,:frozen_rank]; V2=grp[j][Vkey][:,:frozen_rank]; ang=principal_angles(V1,V2)
                    angle_rows.append({'network_id':nid,'rotation_a':grp[j]['rotation'],'rotation_b':grp[i]['rotation'],'kind':name,'mean_angle_deg':float(np.mean(ang)),'max_angle_deg':float(np.max(ang)),'projector_distance':projector_distance(V1,V2)})
    # Probe32 diagnostic.
    probe32={'aggregate_ratio':float(sum(r['baseline_mse']*r['probe32_ratio'] for r in row_summaries if r['split']=='validation')/sum(r['baseline_mse'] for r in row_summaries if r['split']=='validation')),
             'mean_cosine':float(np.mean([r['probe32_cosine'] for r in row_summaries if r['split']=='validation'])),'warning':'first 32 of 128 selected slots; not the historical K32 teacher asset'}
    # Save arrays and results.
    pd.DataFrame(row_summaries).to_csv(a.out/'grouped_prediction_rows.csv',index=False); pd.DataFrame(angle_rows).to_csv(a.out/'cross_rotation_angles.csv',index=False)
    np.savez_compressed(a.out/'MODEL_FREEZE.npz',feature_mean=mean,feature_std=std,ridge_legal=Blegal,ridge_exact=Bexact,ridge_vector=Bvec,ridge_scalar=Bscalar,
                        pooled_basis=Vpool,template=template,codebook=np.stack(codebook),frozen_rank=np.array(frozen_rank),frozen_mechanism=np.array(frozen_mech),rotations=np.array(ROTATIONS))
    oracle={'representation':'selected lower-defect slot x downstream-output contribution C=diag(delta_anchor) beta_bar','train_groups':TRAIN_IDS,'validation_groups':VALID_IDS,'rotations':ROTATIONS,
            'frozen_rank':frozen_rank,'train_mechanism_oracle_ratios':mech_train,'validation_rank_metrics':{k:v for k,v in metrics.items() if k.startswith('exact_rank')},
            'rank_needed_for_final_control_benefit':{t:{'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'max':int(max(v)),'values':v} for t,v in rank_need.items()},
            'cross_rotation':{'rows_file':'cross_rotation_angles.csv','summary':pd.DataFrame(angle_rows).groupby('kind')[['mean_angle_deg','max_angle_deg','projector_distance']].mean().to_dict()},
            'probe32_diagnostic':probe32,'reference_noise_over_baseline':{'median':float(np.median([s['reference_noise_mse']/s['baseline_mse'] for s in val])),'p90':float(np.quantile([s['reference_noise_mse']/s['baseline_mse'] for s in val],.9))}}
    mode={'frozen_rank':frozen_rank,'frozen_legal_mechanism':frozen_mech,'train_oracle_subspace_ratios':mech_train,'ridge_cv':{'legal_coeff':{'lambda':lam_legal,'grouped_cv_ratio':cv_legal},'exact_coeff':{'lambda':lam_exact,'grouped_cv_ratio':cv_exact},'direct_vector':{'lambda':lam_vec,'grouped_cv_ratio':cv_vec},'template_scalar':{'lambda':lam_scalar}},
          'validation_metrics':metrics,'unavailable_same_cohort_baselines':{'matched_K32_K128_teacher':'raw matched vectors absent from shared bundle','best_analytic_anchor':'prior exposed six-network validation ratio 0.992662, not mixed into this new cohort','analytic_plus_low_rank_residual':'not run because no frozen legal analytic anchor passed the development gate'},
          'promotion_gate':{'ratio':.595,'wins':.75,'worst':1.15}}
    (a.out/'ORACLE_RANK_CEILING.json').write_text(json.dumps(oracle,indent=2)); (a.out/'MODE_PREDICTION_RESULTS.json').write_text(json.dumps(mode,indent=2))
    hashes={}
    for p in [a.out/'run_experiment.py',a.out/'MODEL_FREEZE.npz',a.out/'grouped_prediction_rows.csv']:
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'model_freeze_hashes.json').write_text(json.dumps(hashes,indent=2))
    cost={'measured_total_seconds':time.time()-t_all,'samples':len(samples),'base_network_groups':len(TRAIN_IDS)+len(VALID_IDS),'rotations_per_group':3,'truth_nodes_per_half':a.truth_n,
          'incremental_deployable_model':{'ridge_feature_count':int(Xall.shape[1]),'coefficient_count':int(Blegal.size),'estimated_dense_inference_flops':int(2*Xall.shape[1]*frozen_rank),'basis_projection_flops':int(2*D*frozen_rank),'asset_bytes':int((a.out/'MODEL_FREEZE.npz').stat().st_size)},
          'qualification':'Feature extraction here reuses Kerdock activations, fold statistics, fitted beta, and suffix gates. Full subprocess wall/FLOP integration was not available in the shared launch pack.'}
    (a.out/'COST_MODEL.json').write_text(json.dumps(cost,indent=2))
    # concise decision data for report generation
    decision={'frozen_rank':frozen_rank,'frozen_mechanism':frozen_mech,'exact_rank_metric':metrics[f'exact_rank{frozen_rank}'],'legal_oracle_metric':metrics[f'legal_{frozen_mech}_oracle_coeff'],'legal_pred_metric':metrics[f'legal_{frozen_mech}_predicted_coeff'],'direct_vector_metric':metrics['direct_anchor_vector_learner'],'full_metric':metrics['full_exact_lower']}
    (a.out/'DECISION_DATA.json').write_text(json.dumps(decision,indent=2))
    print('FINAL',json.dumps(decision,indent=2),flush=True)

if __name__=='__main__': main()
