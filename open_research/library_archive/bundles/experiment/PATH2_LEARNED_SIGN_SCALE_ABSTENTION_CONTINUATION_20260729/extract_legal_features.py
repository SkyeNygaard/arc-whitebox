from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

D=256; DEPTH=32; N_BASES=129; ROWS_PER_BASIS=512
ROOT=Path(__file__).resolve().parent
REPRO=Path('/mnt/data/whest_path2/additional/agent3_agent4_repro/agent3_agent4_repro')
sys.path.insert(0,str(REPRO))
from sampling import full_real_kerdock_bases, haar_rotation, chi_mean
from agent34_screen_fast import make_weights

SELECTED={0,4,8,12,16,20,24,28,29,30,31}

def qstats(x: np.ndarray, prefix: str, out: dict[str,float]):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x): return
    out[prefix+'_mean']=float(np.mean(x)); out[prefix+'_std']=float(np.std(x))
    out[prefix+'_min']=float(np.min(x)); out[prefix+'_max']=float(np.max(x))
    for q in (0.1,0.25,0.5,0.75,0.9): out[f'{prefix}_q{int(100*q)}']=float(np.quantile(x,q))

def make_kerdock(rotation_seed:int)->torch.Tensor:
    Q=haar_rotation(D,rotation_seed); r=chi_mean(D)
    return torch.stack([torch.cat([b@Q,-(b@Q)],0)*r for b in full_real_kerdock_bases(D)],0).reshape(-1,D).contiguous()

def sample_anchor_matrix(H: np.ndarray, m: np.ndarray, rho: float)->np.ndarray:
    raw=(H*H).T@H/len(H); M=H.T@H/len(H); m2=np.diag(M)
    return (raw/(rho*rho)-D/(D+1)*m2[:,None]*m[None,:]/(rho*rho)
            -2*D/(D+1)*m[:,None]*M/(rho*rho)+2/(D+1)*(m*m)[:,None]*m[None,:])

def layer_features(x:torch.Tensor, li:int, out:dict[str,float]):
    a=x.detach().cpu().numpy().astype(np.float32,copy=False)
    p=f'l{li:02d}'
    out[p+'_mean']=float(a.mean()); out[p+'_std']=float(a.std()); out[p+'_rms']=float(np.sqrt(np.mean(a*a)))
    out[p+'_zero']=float(np.mean(a==0)); out[p+'_max']=float(a.max())
    nm=a.mean(0); nz=(a==0).mean(0); nr=np.sqrt(np.mean(a*a,axis=0))
    qstats(nm,p+'_neuron_mean',out); qstats(nz,p+'_neuron_zero',out); qstats(nr,p+'_neuron_rms',out)
    blocks=a.reshape(N_BASES,ROWS_PER_BASIS,D)
    bm=blocks.mean(1)
    bnorm=np.linalg.norm(bm,axis=1)/math.sqrt(D)
    qstats(bnorm,p+'_block_mean_norm',out)
    disp=np.linalg.norm(bm-bm.mean(0,keepdims=True),axis=1)/math.sqrt(D)
    qstats(disp,p+'_block_disp',out)
    pos=blocks[:,:D,:]; neg=blocks[:,D:,:]
    pair_sum=pos+neg
    pair_imb=np.sqrt(np.mean(pair_sum*pair_sum,axis=(1,2)))
    qstats(pair_imb,p+'_antipodal_imb',out)
    # Six complete-basis groups: internal stability of the actual baseline trajectory.
    fold_means=[]
    for ids in np.array_split(np.arange(N_BASES),6): fold_means.append(bm[ids].mean(0))
    F=np.stack(fold_means); fm=F.mean(0)
    rel=np.linalg.norm(F-fm,axis=1)/(np.linalg.norm(fm)+1e-12)
    qstats(rel,p+'_fold_rel',out)
    cos=[]
    for i in range(len(F)):
        for j in range(i): cos.append(float(F[i]@F[j]/((np.linalg.norm(F[i])*np.linalg.norm(F[j]))+1e-12)))
    qstats(np.array(cos),p+'_fold_cos',out)

def weight_features(ws:list[torch.Tensor],out:dict[str,float]):
    per=[]
    for li,w in enumerate(ws):
        a=w.detach().cpu().numpy().astype(np.float64,copy=False)
        rn=np.linalg.norm(a,axis=1); cn=np.linalg.norm(a,axis=0)
        vals={'mean':a.mean(),'std':a.std(),'abs':np.abs(a).mean(),'pos':(a>0).mean(),
              'rn_mean':rn.mean(),'rn_std':rn.std(),'rn_max':rn.max(),
              'cn_mean':cn.mean(),'cn_std':cn.std(),'cn_max':cn.max(),
              'trace':np.trace(a)/D}
        per.append(vals)
        if li in (0,28,29,30,31):
            for k,v in vals.items(): out[f'w{li:02d}_{k}']=float(v)
    for lo,hi,name in [(0,8,'early'),(8,16,'mid1'),(16,24,'mid2'),(24,29,'late'),(29,32,'suffix'),(0,32,'all')]:
        for k in per[0]: qstats(np.array([per[i][k] for i in range(lo,hi)]),f'w_{name}_{k}',out)

def extract(seed:int,rotation:int,xk:torch.Tensor)->dict[str,float]:
    t=time.time(); ws=make_weights(seed); out={'network_seed':seed,'rotation_seed':rotation}
    weight_features(ws,out)
    x=xk
    h29=None
    with torch.no_grad():
        for li,w in enumerate(ws):
            x=torch.relu(x@w)
            if li in SELECTED: layer_features(x,li,out)
            if li==29: h29=x.clone()
    assert h29 is not None
    H=h29.cpu().numpy().astype(np.float64,copy=False); m=H.mean(0); rho=float(np.linalg.norm(m))
    A=sample_anchor_matrix(H,m,max(rho,1e-12))
    rown=np.linalg.norm(A,axis=1); coln=np.linalg.norm(A,axis=0)
    qstats(rown,'anchor_row_norm',out); qstats(coln,'anchor_col_norm',out)
    s=np.linalg.svd(A,compute_uv=False)
    qstats(s[:32],'anchor_sing_top32',out)
    e=s*s; out['anchor_effrank']=float((e.sum()**2)/(np.sum(e*e)+1e-30)); out['anchor_r90']=float(np.searchsorted(np.cumsum(e)/(e.sum()+1e-30),.9)+1)
    out['anchor_frob']=float(np.linalg.norm(A)); out['anchor_trace']=float(np.trace(A)); out['anchor_rho']=rho
    out['feature_runtime_seconds']=time.time()-t
    return out

def build_examples()->pd.DataFrame:
    base=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE')
    c=pd.read_csv(base/'UNIFORM_INDEPENDENT_LAYER31_RESIDUAL_ROWS.csv')
    c=c[c.probe_count==32].copy(); c['rotation_seed']=3; c['domain']='canonical'; c['oracle_headroom']=c.oracle_ratio<1
    c=c[['network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain']]
    h=pd.read_csv(base/'UNIFORM_HARD_ROTATION_PANEL.csv'); h=h[h.probe_count==32].copy(); h['domain']='hard'
    h=h[['network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain']]
    # canonical row wins when duplicated; hard marker retained separately later by key.
    extra=h.merge(c[['network_seed','rotation_seed']],on=['network_seed','rotation_seed'],how='left',indicator=True)
    extra=extra[extra._merge=='left_only'].drop(columns='_merge')
    df=pd.concat([c,extra],ignore_index=True)
    hardkeys=set(map(tuple,h[['network_seed','rotation_seed']].to_numpy()))
    df['in_hard_panel']=[(int(a),int(b)) in hardkeys for a,b in df[['network_seed','rotation_seed']].to_numpy()]
    df['harm']=(df.candidate_ratio>1).astype(int); df['no_headroom']=(df.oracle_ratio>=1).astype(int)
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--threads',type=int,default=16); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    df=build_examples();
    if args.limit: df=df.iloc[:args.limit]
    cache=ROOT/'cache'; cache.mkdir(exist_ok=True)
    rotations={r:make_kerdock(int(r)) for r in sorted(df.rotation_seed.unique())}
    rows=[]
    for i,r in df.iterrows():
        p=cache/f"features_{int(r.network_seed)}_r{int(r.rotation_seed)}.json"
        if p.exists(): feat=json.loads(p.read_text())
        else:
            feat=extract(int(r.network_seed),int(r.rotation_seed),rotations[int(r.rotation_seed)])
            p.write_text(json.dumps(feat,sort_keys=True))
        row={**r.to_dict(),**feat}; rows.append(row)
        print(json.dumps({'done':i+1,'n':len(df),'seed':int(r.network_seed),'rot':int(r.rotation_seed),'runtime':feat['feature_runtime_seconds']}),flush=True)
    out=pd.DataFrame(rows); out.to_csv(ROOT/'legal_features_and_labels.csv',index=False)
    meta={'n_examples':len(out),'n_groups':int(out.network_seed.nunique()),'n_features':len(out.columns)-len(df.columns),'threads':args.threads}
    (ROOT/'feature_manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
