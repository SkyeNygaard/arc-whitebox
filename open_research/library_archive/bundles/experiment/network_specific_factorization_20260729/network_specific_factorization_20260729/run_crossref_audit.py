#!/usr/bin/env python3
from __future__ import annotations
import gc, json, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
OUT=Path('/mnt/data/paths/01_legal_signed_anchor/network_specific_factorization_20260729')
SRC=Path('/mnt/data/exp3_work/legal_signed_anchor_continuation_20260729/legal_signed_anchor_continuation_20260729/continue_path1/02_centered_analytic_closures/src')
sys.path.insert(0,str(SRC)); sys.path.insert(0,str(OUT))
import frozen_reference_impl as fr
import run_experiment as re
D=256; IDS=list(range(4400,4424)); ROTS=[3,11,97]; N=16384; CHUNK=512

def urisk(p,y1,y2): return float(np.mean((p-y1)*(p-y2)))
def pmse(p,y1,y2): return float(np.mean((p-.5*(y1+y2))**2))
def build_C(H,Y,mom):
    m=H.mean(0); rho=fr.chi_mean(fr.D); Q=fr.sample_anchor_matrix(H,m,rho); idx,dirs=fr.sample_row_probes(Q)
    X=fr.radial_features_sample_rows(H,m,idx,dirs,rho); fit=fr.fit_crossfit(X,Y)
    sM=(fr.D/(rho*rho))*(H.T@H/len(H)); sraw=((fr.D+1)/(rho*rho))*((H*H).T@H/len(H))
    comps=fr.anchor_component_matrices(mom['mu'],mom['M'],mom['raw'],m,sM,sraw)
    delta=fr.contract_rows(comps['lower_only']-Q,idx,dirs); fw=fit['fold_sizes']/fit['fold_sizes'].sum(); beta=np.einsum('f,fpd->pd',fw,fit['betas'])
    return delta[:,None]*beta,beta,fr.contract_rows(Q,idx,dirs),fit

def metrics(rows,key):
    b=np.array([r['baseline_urisk'] for r in rows]); c=np.array([r[key+'_urisk'] for r in rows]); bp=np.array([r['baseline_pmse'] for r in rows]); cp=np.array([r[key+'_pmse'] for r in rows])
    pr=cp/np.maximum(bp,1e-300); valid=b>0
    rng=np.random.default_rng(20260730+sum(map(ord,key))); groups=sorted(set(r['network_id'] for r in rows)); bs=[]
    for _ in range(5000):
        gs=rng.choice(groups,len(groups),replace=True); ix=[i for g in gs for i,r in enumerate(rows) if r['network_id']==g]
        bs.append(c[ix].sum()/max(b[ix].sum(),1e-300))
    return {'aggregate_unbiased_ratio':float(c.sum()/b.sum()),'grouped_bootstrap_95_unbiased':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
            'aggregate_independent_pooled_ratio':float(cp.sum()/bp.sum()),'wins_unbiased':int(np.sum(c[valid]<b[valid])),'positive_baseline_rows':int(valid.sum()),
            'wins_pooled':int(np.sum(cp<bp)),'n':len(rows),'median_pooled_ratio':float(np.median(pr)),'p90_pooled_ratio':float(np.quantile(pr,.9)),'worst_pooled_ratio':float(np.max(pr))}

def main():
    torch.set_num_threads(min(16,os.cpu_count() or 16)); t0=time.time(); xks={r:re.make_kerdock_rot(fr,r) for r in ROTS}
    z=np.load(OUT/'MODEL_FREEZE.npz',allow_pickle=True); rank=int(z['frozen_rank']); mech=str(z['frozen_mechanism']); mean=z['feature_mean']; std=z['feature_std']; Bleg=z['ridge_legal']; Bex=z['ridge_exact']; Bvec=z['ridge_vector']; Bsc=z['ridge_scalar']; Vpool=z['pooled_basis']; template=z['template']; codebook=z['codebook']
    rows=[]; angles=[]
    for nid in IDS:
        tn=time.time(); ws,_,_=fr.make_weights(nid); wf=re.weight_features(ws)
        # A streams define anchor moments. B streams define output targets and are independent.
        A1=fr.stream_reference(ws,N,91000000+4*nid,CHUNK); A2=fr.stream_reference(ws,N,91000001+4*nid,CHUNK)
        B1=fr.stream_reference(ws,N,92000000+4*nid,CHUNK); B2=fr.stream_reference(ws,N,92000001+4*nid,CHUNK)
        mom={k:.5*(A1[k]+A2[k]) for k in A1}; prev=[]
        for rot in ROTS:
            ts=time.time(); hk,yk=fr.forward_target_final(torch.from_numpy(xks[rot]),ws); H=hk.numpy().astype(np.float64); Y=yk.numpy().astype(np.float64); base=Y.mean(0)
            C,beta,q,fit=build_C(H,Y,mom); U,S,VT=np.linalg.svd(C,full_matrices=False); Vex=re.canonicalize(VT.T); full=C.sum(0)
            Vbeta=re.svd_right(beta,16); Vfold=re.svd_right(fit['fold_y_mean']-fit['fold_y_mean'].mean(0),16)
            z30=H@ws[30].numpy(); a30=np.maximum(z30,0); z31=a30@ws[31].numpy(); g30=(z30>0).mean(0); g31=(z31>0).mean(0)
            A=(ws[30].numpy().astype(np.float64)*g30[None,:])@(ws[31].numpy().astype(np.float64)*g31[None,:]); Vsoft=re.svd_right(A,16); Wprod=ws[30].numpy().astype(np.float64)@ws[31].numpy().astype(np.float64); Vweight=re.svd_right(Wprod,16); Vunion=re.orth(np.c_[Vsoft[:,:8],Vfold[:,:5],Vbeta[:,:8]],16)
            sm={'beta':beta,'fold_y':fit['fold_y_mean'],'fold_x':fit['fold_x_mean'],'sample_anchor':q,'gate30':g30,'gate31':g31,'baseline':base,'weight_features':wf,
                'V_beta':Vbeta,'V_fold':Vfold,'V_softgate':Vsoft,'V_weightprod':Vweight,'V_union':Vunion,'V_legal_max':{'V_beta':Vbeta,'V_fold':Vfold,'V_softgate':Vsoft,'V_weightprod':Vweight,'V_union':Vunion}[mech][:,:rank]}
            x=(re.sample_features(sm)-mean)/std; pleg=np.r_[1,x]@Bleg; pex=np.r_[1,x]@Bex; pvec=np.r_[1,x]@Bvec; psc=float((np.r_[1,x]@Bsc)[0])
            Vleg=sm['V_legal_max']; Vxr=Vex[:,:rank]
            candidates={'zero':np.zeros(D),'full_exact_lower':full,'exact_rank1':((U[:,:1]*S[:1])@VT[:1]).sum(0),'exact_rank2':((U[:,:2]*S[:2])@VT[:2]).sum(0),'exact_rank4':((U[:,:4]*S[:4])@VT[:4]).sum(0),'exact_rank8':((U[:,:8]*S[:8])@VT[:8]).sum(0),'exact_rank12':((U[:,:12]*S[:12])@VT[:12]).sum(0),
              'legal_oracle_coeff':Vleg@(Vleg.T@full),'legal_predicted_coeff':Vleg@pleg,'exact_subspace_predicted_coeff':Vxr@pex,'direct_anchor_vector_learner':pvec,'frozen_template_learned_scalar':template*psc,'pooled_rank':Vpool[:,:rank]@(Vpool[:,:rank].T@full)}
            for name,V in [('beta',Vbeta),('fold',Vfold),('softgate',Vsoft),('weightprod',Vweight),('union',Vunion)]: candidates[name+'_oracle_coeff']=V[:,:rank]@(V[:,:rank].T@full)
            seld=[re.projector_distance(Vleg,V) for V in codebook]; Vc=codebook[int(np.argmin(seld))]; candidates['codebook_legal_select_oracle_coeff']=Vc@(Vc.T@full)
            row={'network_id':nid,'rotation':rot,'baseline_urisk':urisk(base,B1['y'],B2['y']),'baseline_pmse':pmse(base,B1['y'],B2['y']),'target_noise_mse':float(.25*np.mean((B1['y']-B2['y'])**2)),'anchor_half_correction_relerr':None,'seconds':time.time()-ts}
            # Anchor half noise diagnostic through full correction.
            C1,_,_,_=build_C(H,Y,A1); C2,_,_,_=build_C(H,Y,A2); f1=C1.sum(0); f2=C2.sum(0); row['anchor_half_correction_relerr']=float(np.linalg.norm(f1-f2)/max(np.linalg.norm(full),1e-30)); row['anchor_half_correction_cosine']=re.cosine(f1,f2)
            for k,c in candidates.items(): p=base+c; row[k+'_urisk']=urisk(p,B1['y'],B2['y']); row[k+'_pmse']=pmse(p,B1['y'],B2['y'])
            rows.append(row); prev.append((rot,Vex[:,:rank],Vleg)); print(json.dumps({'id':nid,'rot':rot,'sec':round(row['seconds'],2),'base_u':row['baseline_urisk'],'rank2_u_ratio':row['exact_rank2_urisk']/max(row['baseline_urisk'],1e-300),'legalpred_u_ratio':row['legal_predicted_coeff_urisk']/max(row['baseline_urisk'],1e-300)}),flush=True)
            del H,Y,C,C1,C2,U,S,VT,z30,a30,z31,A; gc.collect()
        for i in range(3):
            for j in range(i):
                for kind,pos in [('exact',1),('legal',2)]:
                    ang=re.principal_angles(prev[i][pos],prev[j][pos]); angles.append({'network_id':nid,'rotation_a':prev[j][0],'rotation_b':prev[i][0],'kind':kind,'mean_angle_deg':float(np.mean(ang)),'max_angle_deg':float(np.max(ang)),'projector_distance':re.projector_distance(prev[i][pos],prev[j][pos])})
        print('network',nid,'total',round(time.time()-tn,1),flush=True); del A1,A2,B1,B2,mom,ws; gc.collect()
    keys=[k[:-6] for k in rows[0] if k.endswith('_urisk') and k!='baseline_urisk']
    result={'protocol':{'anchor_streams_per_network':2,'target_streams_per_network':2,'nodes_per_stream':N,'independence':'anchor moments and output targets use disjoint scrambled Sobol seeds','validation_groups':IDS,'rotations':ROTS,'frozen_rank':rank,'frozen_mechanism':mech},
            'metrics':{k:metrics(rows,k) for k in keys},
            'noise':{'median_target_noise_over_baseline_pmse':float(np.median([r['target_noise_mse']/r['baseline_pmse'] for r in rows])),'median_anchor_half_correction_relerr':float(np.median([r['anchor_half_correction_relerr'] for r in rows])),'median_anchor_half_correction_cosine':float(np.median([r['anchor_half_correction_cosine'] for r in rows]))},
            'cross_rotation':pd.DataFrame(angles).groupby('kind')[['mean_angle_deg','max_angle_deg','projector_distance']].mean().to_dict(), 'measured_seconds':time.time()-t0}
    pd.DataFrame(rows).to_csv(OUT/'CROSSREF_ROWS.csv',index=False); pd.DataFrame(angles).to_csv(OUT/'CROSSREF_ANGLES.csv',index=False); (OUT/'INDEPENDENT_CROSSREF_VALIDATION.json').write_text(json.dumps(result,indent=2)); print('FINAL',json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
