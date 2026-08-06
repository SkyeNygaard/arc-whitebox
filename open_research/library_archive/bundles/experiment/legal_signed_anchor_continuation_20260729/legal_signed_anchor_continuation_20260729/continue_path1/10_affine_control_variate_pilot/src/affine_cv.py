#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,observed_covariance,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET;SIZES=(256,512,1024,2048);GAMMAS=(0,.1,.25,.5,.75,1.0)

def baseline(n,xk):
 ws,wh,seed=fr.make_weights(n);x=torch.from_numpy(xk.copy());g=[];means=[];first=None;Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);g.append((pre>0).double().mean(0).cpu().numpy());means.append(post.double().mean(0).cpu().numpy())
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET:Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 return ws,wh,seed,Ht,Yt,g,means,first

def affine_map(ws,g):
 P=np.eye(D)
 for l in range(TARGET+1):
  P=P@ws[l].double().numpy();P*=g[l][None,:]
 return P

def pilot_group(ws,P,m,n,seed):
 assert n%2==0;eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed);u=eng.draw(n//2,dtype=torch.float32).clamp_(1e-7,1-1e-7);z=math.sqrt(2)*torch.erfinv(2*u-1);z=torch.cat([z,-z],0);x=z.clone()
 with torch.no_grad():
  for l,w in enumerate(ws):
   x=torch.relu(x@w)
   if l==TARGET:break
 H=x.double().numpy();Z=z.double().numpy();S=m[None,:]+Z@P
 return Z,H,S

def cv_anchor(H,S,m,idx,dirs,P):
 mi=m[idx];mv=dirs@m;var_i=np.sum(P[:,idx]*P[:,idx],axis=0);Pv=P@dirs.T;cov_iv=np.sum(P[:,idx]*Pv,axis=0);known_diag=mi*mi+var_i;known_row=mi*mv+cov_iv
 mu_i=mi+(H[:,idx]-S[:,idx]).mean(0);Hv=H@dirs.T;Sv=S@dirs.T;mu_v=mv+(Hv-Sv).mean(0);diag=known_diag+(H[:,idx]**2-S[:,idx]**2).mean(0);row=known_row+(H[:,idx]*Hv-S[:,idx]*Sv).mean(0);di=mu_i-mi;dv=mu_v-mv
 return (diag*dv+2*di*row+2*(mi*mi-mu_i*mu_i)*mu_v)/(D+1)

def q128_first(ws,g,first,H,idx,dirs):
 U=np.eye(D)[:,idx];V=dirs.T
 for l in range(TARGET-1,-1,-1):
  A=ws[l+1].double().numpy()*g[l+1][None,:];U=A@U;V=A@V
 dm0,dc0=first;dmi=dm0@U;dmv=dm0@V;dcii=np.einsum('ip,ij,jp->p',U,dc0,U);dciv=np.einsum('ip,ij,jp->p',U,dc0,V);m=H.mean(0);mi=m[idx];mv=dirs@m;mei=mi+dmi;mev=mv+dmv;sc=D/(fr.chi_mean(D)**2);sd=(H*H).mean(0)*sc;sr=(H[:,idx]*(H@dirs.T)).mean(0)*sc;diag=sd[idx]+dcii+mei*mei-mi*mi;row=sr+dciv+mei*mev-mi*mv;return (diag*dmv+2*dmi*row+2*(mi*mi-mei*mei)*mev)/(D+1)

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed,Ht,Yt,g,means,first=baseline(n,xk);H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa);P=affine_map(ws,g);a0=q128_first(ws,g,first,H,idx,dirs)
 maxg=max(SIZES)//2;Z1,H1,S1=pilot_group(ws,P,m,maxg,131_000_000+2*n);Z2,H2,S2=pilot_group(ws,P,m,maxg,131_000_001+2*n)
 r1=fr.stream_reference(ws,truth_n,131_500_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,131_500_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,1),idx,dirs);mse=lambda p:float(np.mean((p-ty)**2));methods={}
 for size in SIZES:
  gn=size//2;k=gn//2
  def sel(Z,A,B):return np.concatenate([A[:k],A[maxg//2:maxg//2+k]],0),np.concatenate([B[:k],B[maxg//2:maxg//2+k]],0)
  h1,s1=sel(Z1,H1,S1);h2,s2=sel(Z2,H2,S2);aa1=cv_anchor(h1,s1,m,idx,dirs,P);aa2=cv_anchor(h2,s2,m,idx,dirs,P);acv=.5*(aa1+aa2);grid=[]
  for gam in GAMMAS:
   a=a0+gam*(acv-a0);p,_=fr.estimate_from_fit(fit,sa+a);grid.append(mse(p))
  p1,_=fr.estimate_from_fit(fit,sa+aa1);p2,_=fr.estimate_from_fit(fit,sa+aa2);db=p1-base;dc=p2-base;methods[str(size)]={'mse_grid':grid,'pilot_output_cosine':float(db@dc/max(np.linalg.norm(db)*np.linalg.norm(dc),1e-30)),'anchor_disagreement':float(np.linalg.norm(aa1-aa2)/max(np.linalg.norm(acv),1e-30)),'cv_anchor_cosine':float(acv@oracle/max(np.linalg.norm(acv)*np.linalg.norm(oracle),1e-30)),'first_anchor_cosine':float(a0@oracle/max(np.linalg.norm(a0)*np.linalg.norm(oracle),1e-30))}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'gamma_grid':list(GAMMAS),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['gamma_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for s in rs[0]['methods']:
  cm=np.array([r['methods'][s]['mse_grid'] for r in rs]);
  for j,g in enumerate(grid):
   rat=cm[ti,j]/base[ti];rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(s),j))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,s,j=sel;cm=np.array([r['methods'][str(s)]['mse_grid'] for r in rs]);cos=np.array([r['methods'][str(s)]['pilot_output_cosine'] for r in rs]);ths=np.unique(np.r_[np.quantile(cos[ti],[.25,.5,.75]),-1]);gates=[];zero=0
 for th in ths:
  c=np.where(cos[ti]>=th,cm[ti,j],cm[ti,zero]);rat=c/base[ti];gates.append((float(c.sum()/base[ti].sum()),float(rat.max()),float(th)))
 _,_,th=min([x for x in gates if x[1]<=1.15] or gates)
 def block(ix,seed):
  raw=cm[ix,j];c=np.where(cos[ix]>=th,raw,cm[ix,zero]);rat=c/base[ix];return {'n':len(ix),'raw_over_base':float(raw.sum()/base[ix].sum()),'raw_wins':int((raw<base[ix]).sum()),'raw_worst':float(np.max(raw/base[ix])),'gated_over_base':float(c.sum()/base[ix].sum()),'gated_wins':int((c<base[ix]).sum()),'gated_worst':float(rat.max()),'gated_ci95':bootstrap_ratio(base[ix],c,seed),'applied':int((cos[ix]>=th).sum()),'per_network':rat.tolist()}
 return {'selected_size':s,'selected_gamma':float(grid[j]),'cosine_threshold':th,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260910),'validation':block(vi,20260911) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'gamma':float(grid[x[4]])} for x in sorted(safe)[:30]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'gamma':float(grid[x[4]])} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'best':{s:round(min(v['mse_grid'])/z['baseline_mse'],3) for s,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
