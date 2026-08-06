#!/usr/bin/env python3
"""Fit validation-only scale calibration for factorized-K3 x1/x1a features.

The CoefNet was trained on exact k21 features. This script measures how ARC's
recursive factorized-K3 estimate differs and fits conservative layerwise scales.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def connected_21(mu, m11, m21, m2):
    return m21 - 2*mu[:,None]*m11 - m2[:,None]*mu[None,:] + 2*(mu[:,None]**2)*mu[None,:]


def load_weights(path, device, dtype):
    w=np.asarray(np.load(path),np.float64)
    if w.shape != (32,256,256): raise ValueError((path,w.shape))
    return torch.as_tensor(w.transpose(0,2,1).copy(),device=device,dtype=dtype)


def safe_scale(pred, true, ridge=1e-8, cap=5.0):
    num=float(np.dot(pred,true)); den=float(np.dot(pred,pred))+ridge*len(pred)
    return float(np.clip(num/den,-cap,cap))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-json',type=Path,required=True)
    p.add_argument('--moments-dir',type=Path,required=True)
    p.add_argument('--weights-dir',type=Path,required=True)
    p.add_argument('--split',choices=['valid'],default='valid')
    p.add_argument('--pairs-per-layer',type=int,default=4096)
    p.add_argument('--start-mlp',type=int,default=0)
    p.add_argument('--max-mlps',type=int,default=0)
    p.add_argument('--seed',type=int,default=20260820)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float64','float32'],default='float64')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from mlp_kprop.kprop_harmonic import SIMPLE,coerce_input,linear_kprop,nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef
    cfg=json.loads(a.results_json.read_text()); ids=list(map(int,cfg['valid_ids']))
    ids=ids[a.start_mlp:]
    ids=ids[:a.max_mlps] if a.max_mlps else ids
    device=torch.device(a.device); dtype=torch.float64 if a.dtype=='float64' else torch.float32
    rng=np.random.default_rng(a.seed)
    pred_s=[[] for _ in range(31)]; true_s=[[] for _ in range(31)]
    pred_a=[[] for _ in range(31)]; true_a=[[] for _ in range(31)]
    per_file=[]
    for count,idx in enumerate(ids,1):
        with np.load(a.moments_dir/f'mlp_{idx:05d}.npz') as d:
            pre_mean=np.asarray(d['pre_mean'],np.float64)
            pre_M11=np.asarray(d['pre_M11'],np.float64)
            pre_M21=np.asarray(d['pre_M21'],np.float64)
            pre_m2=np.asarray(d['pre_m2'],np.float64)
        weights=load_weights(a.weights_dir/f'mlp_{idx:05d}.npy',device,dtype)
        K=coerce_input({1:torch.zeros(256,device=device,dtype=dtype),2:torch.eye(256,device=device,dtype=dtype)},k_max=3,kind=SIMPLE)
        rows=0
        with torch.no_grad():
            for layer in range(31):
                Kpre=linear_kprop(K,weights[layer],k_max=3)
                # Layer 0 pre-activations are exactly Gaussian, so there is no K3
                # term to propagate and the predicted features are identically zero.
                if 3 in Kpre:
                    kp=Kpre[3].get_dslice((2,1)).detach().cpu().numpy().astype(np.float64)
                else:
                    kp=np.zeros((256,256),np.float64)
                mu=pre_mean[layer]; cov=pre_M11[layer]-np.outer(mu,mu)
                sig=np.sqrt(np.maximum(np.diag(cov),1e-12))
                kt=connected_21(mu,pre_M11[layer],pre_M21[layer],pre_m2[layer])
                total=256*255//2; take=min(a.pairs_per_layer,total)
                flat=rng.choice(total,take,replace=False)
                iu,ju=np.triu_indices(256,1); i,j=iu[flat],ju[flat]
                den=np.maximum(sig[i]**3+sig[j]**3,1e-12)
                ps=(kp[i,j]+kp[j,i])/den; pa=(kp[i,j]-kp[j,i])/den
                ts=(kt[i,j]+kt[j,i])/den; ta=(kt[i,j]-kt[j,i])/den
                mask=np.isfinite(ps)&np.isfinite(pa)&np.isfinite(ts)&np.isfinite(ta)
                pred_s[layer].append(ps[mask]); true_s[layer].append(ts[mask])
                pred_a[layer].append(pa[mask]); true_a[layer].append(ta[mask]); rows+=int(mask.sum())
                K=nonlin_kprop(Kpre,nonlin_wick_coef=relu_wick_coef,k_max=3,kind=SIMPLE,use_pK=True,factor=True)
        per_file.append({'global_index':idx,'rows':rows})
        print(json.dumps({'loaded':count,'global_index':idx,'rows':rows}),flush=True)
    ps_all=np.concatenate([np.concatenate(x) for x in pred_s]); ts_all=np.concatenate([np.concatenate(x) for x in true_s])
    pa_all=np.concatenate([np.concatenate(x) for x in pred_a]); ta_all=np.concatenate([np.concatenate(x) for x in true_a])
    gs=safe_scale(ps_all,ts_all); ga=safe_scale(pa_all,ta_all)
    layer_s=[]; layer_a=[]; diagnostics=[]
    # Shrink each layer estimate toward global scale to avoid fitting noise.
    prior_rows=20000
    for l in range(31):
        ps=np.concatenate(pred_s[l]); ts=np.concatenate(true_s[l]); pa=np.concatenate(pred_a[l]); ta=np.concatenate(true_a[l])
        ls=safe_scale(ps,ts); la=safe_scale(pa,ta); weight=len(ps)/(len(ps)+prior_rows)
        layer_s.append(weight*ls+(1-weight)*gs); layer_a.append(weight*la+(1-weight)*ga)
        diagnostics.append({'layer':l,'rows':len(ps),'x1_scale_raw':ls,'x1a_scale_raw':la,
          'x1_corr':float(np.corrcoef(ps,ts)[0,1]),'x1a_corr':float(np.corrcoef(pa,ta)[0,1]),
          'x1_pred_rms':float(np.sqrt(np.mean(ps*ps))),'x1_true_rms':float(np.sqrt(np.mean(ts*ts))),
          'x1a_pred_rms':float(np.sqrt(np.mean(pa*pa))),'x1a_true_rms':float(np.sqrt(np.mean(ta*ta)))})
    # final layer has no ReLU; repeat layer 30 for array length 32
    result={'x1_scale':layer_s+[layer_s[-1]],'x1a_scale':layer_a+[layer_a[-1]],
      'global_x1_scale':gs,'global_x1a_scale':ga,'rows':len(ps_all),'layers':diagnostics,'files':per_file}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(a.output),'global_x1_scale':gs,'global_x1a_scale':ga,'rows':len(ps_all)}))
if __name__=='__main__':main()
