#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import bootstrap_ratio
D=fr.D
TOTAL_SIZES=(512,1024,2048,4096)
ALPHAS=(0.0,0.05,0.1,0.2,0.35,0.5,0.75,1.0)

def capture_baseline_and_affine(ws,xk):
    x=torch.from_numpy(xk.copy())
    d=ws[0].shape[0]
    A=np.eye(d,dtype=np.float64); c=np.zeros(d,dtype=np.float64)
    layer_stats=[]
    with torch.no_grad():
        for l,wt in enumerate(ws):
            pre=x@wt; post=torch.relu(pre)
            pre_mean=pre.double().mean(0).cpu().numpy()
            post_mean=post.double().mean(0).cpu().numpy()
            p=(pre>0).double().mean(0).cpu().numpy()
            w=wt.double().cpu().numpy()
            b=post_mean-p*pre_mean
            A=(A@w)*p[None,:]
            c=(c@w)*p+b
            # By construction c should match the Kerdock post mean.
            layer_stats.append({'layer':l,'mean_identity_error':float(np.max(np.abs(c-post_mean))),
                                'gate_min':float(p.min()),'gate_max':float(p.max()),
                                'A_fro':float(np.linalg.norm(A))})
            x=post
    base=x.double().mean(0).cpu().numpy()
    return base,A,c,layer_stats

def antithetic_normal(n,seed):
    assert n%2==0
    eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed)
    u=eng.draw(n//2,dtype=torch.float32).clamp_(1e-7,1-1e-7)
    z=math.sqrt(2)*torch.erfinv(2*u-1)
    return torch.cat([z,-z],0)

def exact_network(z,ws):
    x=z
    with torch.no_grad():
        for w in ws:x=torch.relu(x@w)
    return x.double().cpu().numpy()

def pilot_group(n,seed,ws,A,c):
    z=antithetic_normal(n,seed)
    f=exact_network(z,ws)
    g=z.double().cpu().numpy()@A+c[None,:]
    r=f-g
    return r.mean(0), {'residual_rms':float(np.sqrt(np.mean(r*r))),
                       'output_rms':float(np.sqrt(np.mean(f*f))),
                       'surrogate_rms':float(np.sqrt(np.mean(g*g)))}

def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

def run(network,xk,truth_n,chunk):
    t=time.perf_counter();ws,wh,seed=fr.make_weights(network)
    base,A,c,layers=capture_baseline_and_affine(ws,xk)
    max_half=max(TOTAL_SIZES)//2
    # Two independent antithetic groups, each half of total size.
    # Generate each requested group independently to avoid nesting artifacts.
    refs1=fr.stream_reference(ws,truth_n,151_000_000+2*network,chunk)
    refs2=fr.stream_reference(ws,truth_n,151_000_001+2*network,chunk)
    truth=.5*(refs1['y']+refs2['y'])
    mse=lambda p:float(np.mean((p-truth)**2))
    base_mse=mse(base); methods={}
    true_delta=truth-base
    for total in TOTAL_SIZES:
        half=total//2
        d1,s1=pilot_group(half,152_000_000+20*network+total,ws,A,c)
        d2,s2=pilot_group(half,152_000_001+20*network+total,ws,A,c)
        d=.5*(d1+d2)
        grid=[mse(base+a*d) for a in ALPHAS]
        methods[str(total)]={'mse_grid':grid,'pilot_mutual_cosine':cosine(d1,d2),
                             'pilot_disagreement':float(np.linalg.norm(d1-d2)/max(np.linalg.norm(d),1e-30)),
                             'correction_cosine_truth':cosine(d,true_delta),
                             'delta_norm':float(np.linalg.norm(d)),
                             'true_delta_norm':float(np.linalg.norm(true_delta)),
                             'group1':s1,'group2':s2}
    return {'network_id':network,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,
            'baseline_mse':base_mse,'alpha_grid':list(ALPHAS),'methods':methods,
            'affine_final_identity_error':float(np.max(np.abs(c-base))),
            'max_layer_identity_error':max(x['mean_identity_error'] for x in layers),
            'layer_stats':layers,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
    rs=sorted(rs,key=lambda r:r['network_id']);base=np.array([r['baseline_mse'] for r in rs]);alph=np.array(rs[0]['alpha_grid'])
    ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
    for s in rs[0]['methods']:
        cm=np.array([r['methods'][s]['mse_grid'] for r in rs])
        for j,a in enumerate(alph):
            rat=cm[ti,j]/base[ti]
            rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(s),j))
    safe=[x for x in rows if x[1]<=1.15]
    sel=min(safe or rows);_,_,_,s,j=sel
    cm=np.array([r['methods'][str(s)]['mse_grid'] for r in rs])
    mutual=np.array([r['methods'][str(s)]['pilot_mutual_cosine'] for r in rs])
    dis=np.array([r['methods'][str(s)]['pilot_disagreement'] for r in rs])
    # One bounded gate search: raw candidate only when mutual agreement high and normalized disagreement low.
    mths=np.unique(np.r_[-1.0,np.quantile(mutual[ti],[.25,.5,.75])])
    dths=np.unique(np.r_[np.inf,np.quantile(dis[ti],[.25,.5,.75])])
    gates=[]
    for mt in mths:
      for dt in dths:
        apply=(mutual[ti]>=mt)&(dis[ti]<=dt);cand=np.where(apply,cm[ti,j],cm[ti,0]);rat=cand/base[ti]
        gates.append((float(cand.sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),float(mt),float(dt)))
    gs=[x for x in gates if x[1]<=1.15];gsel=min(gs or gates);_,_,_,mt,dt=gsel
    def block(ix,seed):
      raw=cm[ix,j];apply=(mutual[ix]>=mt)&(dis[ix]<=dt);cand=np.where(apply,raw,cm[ix,0]);rat=cand/base[ix]
      return {'n':len(ix),'raw_over_base':float(raw.sum()/base[ix].sum()),'raw_wins':int((raw<base[ix]).sum()),'raw_worst':float(np.max(raw/base[ix])),
              'gated_over_base':float(cand.sum()/base[ix].sum()),'gated_wins':int((cand<base[ix]).sum()),'gated_worst':float(rat.max()),
              'gated_ci95':bootstrap_ratio(base[ix],cand,seed),'applied':int(apply.sum()),'per_network':rat.tolist()}
    return {'selected_total_pilot':s,'selected_alpha':float(alph[j]),'mutual_threshold':mt,'disagreement_threshold':dt,
            'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],
            'tuning':block(ti,20261201),'validation':block(vi,20261202) if len(vi) else {},
            'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'total':x[3],'alpha':float(alph[x[4]])} for x in sorted(safe)[:30]],
            'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'total':x[3],'alpha':float(alph[x[4]])} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=4);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'identity':z['max_layer_identity_error'],'best':{s:round(min(v['mse_grid'])/z['baseline_mse'],3) for s,v in z['methods'].items()},'cos':{s:round(v['correction_cosine_truth'],3) for s,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
