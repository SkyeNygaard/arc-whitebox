from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
SRC=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE')
sys.path.insert(0,str(SRC))
import run_layer31_residual as r

D=r.D; N_BASES=r.N_BASES; ROWS=r.ROWS_PER_BASIS
FROZEN_THRESHOLD=0.00279066317598334

def sobol_seed(label:str,network_seed:int,stream:str,scheme:str='hash')->int:
    if scheme=='audited':
        mod=2_147_483_647
        if stream=='anchor1': return (1_300_000_000+2*network_seed)%mod
        if stream=='anchor2': return (1_300_000_001+2*network_seed)%mod
        if stream=='target1': return (1_700_000_000+2*network_seed)%mod
        if stream=='target2': return (1_700_000_001+2*network_seed)%mod
        raise ValueError(stream)
    h=hashlib.sha256(f'{label}:{network_seed}:{stream}'.encode()).digest()
    return 1+int.from_bytes(h[:8],'big')%(2_147_483_646)

def seed_block(label:str,count:int,exclude:set[int])->list[int]:
    out=[]; i=0
    while len(out)<count:
        h=hashlib.sha256(f'{label}:network:{i}'.encode()).digest(); i+=1
        s=100_000_000+int.from_bytes(h[:4],'big')%1_800_000_000
        if s not in exclude and s not in out: out.append(s)
    return out

def stream_or_load(ws,seed:int,label:str,n:int,cache:Path,scheme:str):
    p=cache/f'references_{seed}_n{n}.npz'
    if p.exists():
        z=np.load(p)
        return {k:np.asarray(z[k]) for k in z.files},p
    streams={}
    for name in ['target1','target2','anchor1','anchor2']:
        q=sobol_seed(label,seed,name,scheme)
        print(json.dumps({'stage':'reference','seed':seed,'stream':name,'qmc_seed':q,'n':n}),flush=True)
        streams[name]=r.stream_reference(ws,n,q)
    data={
      'target_final1':streams['target1']['final'],'target_final2':streams['target2']['final'],
      'target_pen1':streams['target1']['penultimate'],'target_pen2':streams['target2']['penultimate'],
      'anchor_mu1':streams['anchor1']['mu_h'],'anchor_mu2':streams['anchor2']['mu_h'],
      'anchor_M1':streams['anchor1']['M_h'],'anchor_M2':streams['anchor2']['M_h'],
      'anchor_raw1':streams['anchor1']['raw_h'],'anchor_raw2':streams['anchor2']['raw_h'],
      'qmc_n':np.array(n,dtype=np.int64),
      'target_seed1':np.array(sobol_seed(label,seed,'target1',scheme),dtype=np.int64),
      'target_seed2':np.array(sobol_seed(label,seed,'target2',scheme),dtype=np.int64),
      'anchor_seed1':np.array(sobol_seed(label,seed,'anchor1',scheme),dtype=np.int64),
      'anchor_seed2':np.array(sobol_seed(label,seed,'anchor2',scheme),dtype=np.int64),
    }
    np.savez_compressed(p,**data)
    return data,p

def forward_rotation(ws,rotation:int):
    x=r.get_kerdock(rotation); bm8=None; H=A30=Y=None
    with torch.no_grad():
        for li,w in enumerate(ws):
            x=torch.relu(x@w)
            if li==8:
                bm8=x.reshape(N_BASES,ROWS,D).double().mean(1).cpu().numpy()
            elif li==r.TARGET_FEATURE_LAYER: H=x.double().cpu().numpy()
            elif li==r.PENULTIMATE_LAYER: A30=x.double().cpu().numpy()
            elif li==r.FINAL_LAYER: Y=x.double().cpu().numpy()
    assert bm8 is not None and H is not None and A30 is not None and Y is not None
    return bm8,H,A30,Y

def risk(bm:np.ndarray,n:int=129)->float:
    z=bm[:n]; f=np.stack([z[ix].mean(0) for ix in np.array_split(np.arange(n),6)])
    c=f.mean(0); return float(np.mean(np.linalg.norm(f-c,axis=1)/(np.linalg.norm(c)+1e-12)))

def quadratic(base, pred, h1, h2):
    d=pred-base
    e1=base-h1; e2=base-h2
    a0=float(np.mean(e1*e2)); lin=float(np.mean(d*(e1+e2))); q=float(np.mean(d*d))
    alpha=float(-lin/(2*q)) if q>1e-30 else 0.0
    return {'unbiased_a0':a0,'unbiased_linear':lin,'correction_norm2':q,'optimal_alpha_unconstrained':alpha,
            'optimal_unbiased_mse':float(a0+alpha*lin+alpha*alpha*q),
            'alpha_minus1_mse':float(a0-lin+q),'alpha_0_mse':a0,'alpha_plus1_mse':float(a0+lin+q)}

def evaluate_variant(H,A30,Y,m,E,truth,h1,h2,ws,p:int):
    Q=r.sample_anchor_matrix(H,m,r.chi_mean(D)); order=np.argsort(np.linalg.norm(Q,axis=1))[::-1]; ix=order[:p]
    rr=Q[ix].copy(); rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30)
    L=np.eye(D,dtype=np.float64)[:,ix]; R=rr.T
    X=r.radial_features(H,m,L,R,r.chi_mean(D)); anc=np.einsum('ir,ij,jr->r',L,E,R)
    direct,_=r.crossfit_mean(X,Y,anc)
    pen,_=r.crossfit_mean(X,A30,anc); replay=r.replay_final(A30,pen,ws[-1])
    base=Y.mean(0); bm=r.mse(base,truth); bu=r.unbiased_mse(base,h1,h2)
    out={
      'support_indices':ix.tolist(),
      'direct_mse':r.mse(direct,truth),'direct_ratio':r.mse(direct,truth)/bm,
      'direct_unbiased_mse':r.unbiased_mse(direct,h1,h2),'direct_unbiased_ratio':r.unbiased_mse(direct,h1,h2)/max(bu,1e-30),
      'replay_mse':r.mse(replay,truth),'replay_ratio':r.mse(replay,truth)/bm,
      'replay_unbiased_mse':r.unbiased_mse(replay,h1,h2),'replay_unbiased_ratio':r.unbiased_mse(replay,h1,h2)/max(bu,1e-30),
      'quadratic':quadratic(base,direct,h1,h2),
    }
    return out,direct,replay

def run_seed(seed:int,label:str,n:int,rotations:list[int],cache:Path,vectors:Path,scheme:str):
    t=time.time(); ws=r.make_weights(seed); ref,refpath=stream_or_load(ws,seed,label,n,cache,scheme)
    h1=np.asarray(ref['target_final1'],float); h2=np.asarray(ref['target_final2'],float); truth=.5*(h1+h2)
    pen_truth=.5*(np.asarray(ref['target_pen1'],float)+np.asarray(ref['target_pen2'],float))
    mu=.5*(np.asarray(ref['anchor_mu1'],float)+np.asarray(ref['anchor_mu2'],float))
    M=.5*(np.asarray(ref['anchor_M1'],float)+np.asarray(ref['anchor_M2'],float))
    raw=.5*(np.asarray(ref['anchor_raw1'],float)+np.asarray(ref['anchor_raw2'],float))
    rows=[]
    for rot in rotations:
        rp=cache/f'label_{seed}_r{rot}_n{n}.json'; vp=vectors/f'vectors_{seed}_r{rot}_n{n}.npz'
        if rp.exists() and vp.exists():
            rows.append(json.load(open(rp))); continue
        rt=time.time(); bm8,H,A30,Y=forward_rotation(ws,rot); base=Y.mean(0); m=H.mean(0)
        E=r.exact_anchor_matrix(mu,M,raw,m)
        bm=r.mse(base,truth); bu=r.unbiased_mse(base,h1,h2)
        oracle=r.replay_final(A30,pen_truth,ws[-1])
        row={'network_seed':seed,'rotation_seed':rot,'qmc_n_per_half':n,
             'baseline_mse':bm,'baseline_unbiased_mse':bu,
             'truth_noise_mse':float(.5*np.mean((h1-h2)**2)),
             'oracle_mse':r.mse(oracle,truth),'oracle_ratio':r.mse(oracle,truth)/bm,
             'oracle_unbiased_mse':r.unbiased_mse(oracle,h1,h2),'oracle_unbiased_ratio':r.unbiased_mse(oracle,h1,h2)/max(bu,1e-30),
             'risk_n032':risk(bm8,32),'risk_n129':risk(bm8,129),
             'frozen_gate_apply_n129':bool(risk(bm8,129)<=FROZEN_THRESHOLD),
             'rotation_runtime_seconds':time.time()-rt,'reference_file':str(refpath)}
        vec={'base':base,'truth_half1':h1,'truth_half2':h2,'oracle':oracle}
        for p in [32,128]:
            v,direct,replay=evaluate_variant(H,A30,Y,m,E,truth,h1,h2,ws,p); row[f'k{p}']=v; vec[f'direct{p}']=direct; vec[f'replay{p}']=replay
        np.savez_compressed(vp,**vec); row['vectors_file']=str(vp)
        rp.write_text(json.dumps(row,indent=2)); rows.append(row)
        print(json.dumps({'stage':'rotation_done','seed':seed,'rot':rot,'runtime':round(row['rotation_runtime_seconds'],2),
                          'risk':row['risk_n129'],'gate':row['frozen_gate_apply_n129'],
                          'oracle':round(row['oracle_ratio'],3),'k32':round(row['k32']['direct_ratio'],3),'k128':round(row['k128']['direct_ratio'],3),
                          'a32':round(row['k32']['quadratic']['optimal_alpha_unconstrained'],3)}),flush=True)
    return rows,time.time()-t

def flatten(rows):
    out=[]
    for z in rows:
        q32=z['k32']['quadratic']; q128=z['k128']['quadratic']
        out.append({k:z[k] for k in ['network_seed','rotation_seed','qmc_n_per_half','baseline_mse','baseline_unbiased_mse','truth_noise_mse','oracle_ratio','oracle_unbiased_ratio','risk_n032','risk_n129','frozen_gate_apply_n129','rotation_runtime_seconds']}|
                   {'k32_ratio':z['k32']['direct_ratio'],'k32_unbiased_ratio':z['k32']['direct_unbiased_ratio'],'k128_ratio':z['k128']['direct_ratio'],'k128_unbiased_ratio':z['k128']['direct_unbiased_ratio'],
                    'alpha32':q32['optimal_alpha_unconstrained'],'alpha32_opt_mse':q32['optimal_unbiased_mse'],'alpha128':q128['optimal_alpha_unconstrained'],'alpha128_opt_mse':q128['optimal_unbiased_mse']})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',default='Path2-K32-independent-screen-v1'); ap.add_argument('--count',type=int,default=8); ap.add_argument('--qmc-n',type=int,default=65536); ap.add_argument('--threads',type=int,default=5); ap.add_argument('--rotations',type=int,nargs='+',default=[3,11,97]); ap.add_argument('--seed-scheme',choices=['hash','audited'],default='hash'); ap.add_argument('--seeds',type=int,nargs='*'); ap.add_argument('--outdir',type=Path,default=ROOT/'fresh_screen_v1'); args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    args.outdir.mkdir(exist_ok=True); cache=args.outdir/'cache'; vectors=args.outdir/'vectors'; cache.mkdir(exist_ok=True); vectors.mkdir(exist_ok=True)
    old=set(pd.read_csv(ROOT/'legal_features_and_labels.csv').network_seed.astype(int))|set(range(51032,51048))|set(range(52000,52024))
    seeds=args.seeds if args.seeds else seed_block(args.label,args.count,old)
    manifest={'label':args.label,'network_seeds':seeds,'rotations':args.rotations,'qmc_n_per_independent_half':args.qmc_n,
              'four_independent_streams':['target1','target2','anchor1','anchor2'],'supports':[32,128],
              'frozen_gate_feature':'post-ReLU layer index 8, 129-basis six-fold relative dispersion mean',
              'frozen_gate_threshold':FROZEN_THRESHOLD,'frozen_before_labels':True,
              'continuous_scale_terms_saved':True,'seed_scheme':args.seed_scheme,'created_utc':'2026-07-29T20:31:00Z'}
    mp=args.outdir/'IMMUTABLE_SCREEN_MANIFEST.json'
    if mp.exists():
        prior=json.load(open(mp));
        if prior!=manifest: raise RuntimeError('manifest mismatch')
    else: mp.write_text(json.dumps(manifest,indent=2))
    allrows=[]
    for i,s in enumerate(seeds):
        rows,rt=run_seed(int(s),args.label,args.qmc_n,args.rotations,cache,vectors,args.seed_scheme); allrows.extend(rows)
        flat=flatten(allrows); flat.to_csv(args.outdir/'fresh_screen_rows.csv',index=False)
        (args.outdir/'fresh_screen_results.json').write_text(json.dumps({'manifest':manifest,'rows':allrows},indent=2))
        print(json.dumps({'stage':'network_done','done':i+1,'total':len(seeds),'seed':s,'runtime':round(rt,2)}),flush=True)
    print(flatten(allrows).to_string(index=False))
if __name__=='__main__': main()
