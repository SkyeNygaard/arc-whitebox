#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,relu_univariate,observed_covariance,observed_marginals,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET;SIZES=(512,1024,2048);LAM=(.05,.1,.2,.35,.5,.75);THRESH=(-.5,-.25,0,.1,.25,.5)

def capture(n,xk):
 ws,wh,seed=fr.make_weights(n);x=torch.from_numpy(xk.copy());g=[];means=[];sm=[];sv=[];first=None;Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);kpm,kpv=observed_marginals(pre);kom,kov=observed_marginals(post);gm,gv,_=relu_univariate(kpm,kpv);g.append((pre>0).double().mean(0).cpu().numpy());means.append(kom);sm.append(gm-kom);sv.append(gv-kov)
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET:Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 return ws,wh,seed,Ht,Yt,g,means,sm,sv,first

def back(ws,g,idx,dirs):
 U=np.eye(D)[:,idx];V=dirs.T;Us=[None]*(TARGET+1);Vs=[None]*(TARGET+1);Us[TARGET]=U;Vs[TARGET]=V
 for l in range(TARGET-1,-1,-1):
  A=ws[l+1].double().numpy()*g[l+1][None,:];U=A@U;V=A@V;Us[l]=U;Vs[l]=V
 return Us,Vs

def anchor(H,idx,dirs,v):
 dmi,dmv,dcii,dciv=v;m=H.mean(0);mi=m[idx];mv=dirs@m;mei=mi+dmi;mev=mv+dmv;sc=D/(fr.chi_mean(D)**2);sd=(H*H).mean(0)*sc;sr=(H[:,idx]*(H@dirs.T)).mean(0)*sc;diag=sd[idx]+dcii+mei*mei-mi*mi;row=sr+dciv+mei*mev-mi*mv;return (diag*dmv+2*dmi*row+2*(mi*mi-mei*mei)*mev)/(D+1)
def add(a,b):return tuple(x+y for x,y in zip(a,b))

def affine_map(ws,g):
 P=np.eye(D)
 for l in range(TARGET+1):P=(P@ws[l].double().numpy())*g[l][None,:]
 return P

def pilot(ws,P,m,n,seed):
 eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed);u=eng.draw(n//2,dtype=torch.float32).clamp_(1e-7,1-1e-7);z=math.sqrt(2)*torch.erfinv(2*u-1);z=torch.cat([z,-z]);x=z.clone()
 with torch.no_grad():
  for l,w in enumerate(ws):
   x=torch.relu(x@w)
   if l==TARGET:break
 Z=z.double().numpy();H=x.double().numpy();S=m+Z@P;return H,S

def cv_anchor(H,S,m,idx,dirs,P):
 mi=m[idx];mv=dirs@m;Pv=P@dirs.T;known_diag=mi*mi+np.sum(P[:,idx]**2,0);known_row=mi*mv+np.sum(P[:,idx]*Pv,0);Hv=H@dirs.T;Sv=S@dirs.T;mui=mi+(H[:,idx]-S[:,idx]).mean(0);muv=mv+(Hv-Sv).mean(0);diag=known_diag+(H[:,idx]**2-S[:,idx]**2).mean(0);row=known_row+(H[:,idx]*Hv-S[:,idx]*Sv).mean(0);di=mui-mi;dv=muv-mv;return (diag*dv+2*di*row+2*(mi*mi-mui*mui)*muv)/(D+1)
def cos(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed,Ht,Yt,g,means,sm,sv,first=capture(n,xk);H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa);U,V=back(ws,g,idx,dirs);dm0,dc0=first;init=(dm0@U[0],dm0@V[0],np.einsum('ip,ij,jp->p',U[0],dc0,U[0]),np.einsum('ip,ij,jp->p',U[0],dc0,V[0]));src=(np.zeros(len(idx)),)*4
 for l in range(1,TARGET+1):src=add(src,(sm[l]@U[l],sm[l]@V[l],np.sum(U[l]*U[l]*sv[l][:,None],0),np.sum(U[l]*V[l]*sv[l][:,None],0)))
 a0=anchor(H,idx,dirs,init);a1=anchor(H,idx,dirs,add(init,src));p0,_=fr.estimate_from_fit(fit,sa+a0);p1,_=fr.estimate_from_fit(fit,sa+a1);d0=p0-base;ds=p1-p0;P=affine_map(ws,g)
 r1=fr.stream_reference(ws,truth_n,141_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,141_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mse=lambda p:float(np.mean((p-ty)**2));methods={}
 for size in SIZES:
  h1,s1=pilot(ws,P,m,size//2,141_500_000+2*n);h2,s2=pilot(ws,P,m,size//2,141_500_001+2*n);ac1=cv_anchor(h1,s1,m,idx,dirs,P);ac2=cv_anchor(h2,s2,m,idx,dirs,P);pc1,_=fr.estimate_from_fit(fit,sa+ac1);pc2,_=fr.estimate_from_fit(fit,sa+ac2);rpil1=pc1-p0;rpil2=pc2-p0;conf=min(cos(rpil1,ds),cos(rpil2,ds));mut=cos(rpil1,rpil2);grid={}
  for th in THRESH:
   for lam in LAM:
    apply=conf>=th;pred=p0+lam*ds if apply else p0;grid[f't{th}_l{lam}']=mse(pred)
  methods[str(size)]={'confidence':conf,'pilot_mutual_cosine':mut,'base_first_mse':mse(p0),'grid':grid}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for s in rs[0]['methods']:
  for key in rs[0]['methods'][s]['grid']:
   cm=np.array([r['methods'][s]['grid'][key] for r in rs]);rat=cm[ti]/base[ti];rows.append((float(cm[ti].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(s),key))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,s,key=sel;cm=np.array([r['methods'][str(s)]['grid'][key] for r in rs])
 def block(ix,seed):
  c=cm[ix];rat=c/base[ix];return {'n':len(ix),'candidate_over_base':float(c.sum()/base[ix].sum()),'ci95':bootstrap_ratio(base[ix],c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist(),'applied':int(sum(r['methods'][str(s)]['grid'][key]!=r['methods'][str(s)]['base_first_mse'] for r in [rs[i] for i in ix]))}
 return {'selected_size':s,'selected_rule':key,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260920),'validation':block(vi,20260921) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'rule':x[4]} for x in sorted(safe)[:30]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'rule':x[4]} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'conf':{s:(round(v['confidence'],3),round(v['pilot_mutual_cosine'],3)) for s,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
