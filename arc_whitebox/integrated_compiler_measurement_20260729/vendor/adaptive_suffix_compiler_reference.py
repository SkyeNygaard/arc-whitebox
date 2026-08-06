#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,time,gc
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'/mnt/data/whest_work')
import nonk3_suffix_compiler_fast as sc
D=sc.D; DEP=sc.DEP; R1=sc.R1; BASE=sc.BASE

def classify_layer(hp,rare=1):
 p=np.sum(hp>0,axis=0); n=len(hp)-p
 st=np.minimum(p,n)<=rare; on=st&(p>=n); off=st&~on; kink=~st
 return np.flatnonzero(on),np.flatnonzero(off),np.flatnonzero(kink)

def exact_prefix_suffix(W,X,k):
 anchor=X
 for w in W[:DEP-k]:
  anchor=anchor@w; np.maximum(anchor,0,out=anchor)
 pre=[]; act=anchor
 for w in W[DEP-k:]:
  h=act@w; pre.append(h); act=np.maximum(h,0)
 return anchor,pre,act

def compile_predict(anchor,Ws,C,I,Y):
 # Symbolic stable-on representation: a_O = X B + sum_j R_j C_j
 N=len(anchor); P=len(I); k=len(Ws)
 X=anchor.astype(np.float64,copy=False); Xp=X[I]
 meanX=X.mean(axis=0,dtype=np.float64)
 Rs=[]; Rps=[]
 O0,_,K0=C[0]
 if len(K0):
  h=X@Ws[0][:,K0].astype(np.float64); R=np.maximum(h,0); Rp=R[I]
 else: R=np.empty((N,0)); Rp=np.empty((P,0))
 Rs.append(R);Rps.append(Rp)
 B=Ws[0][:,O0].astype(np.float64)
 Cs=[]
 # each next layer
 for l in range(1,k):
  Oprev=C[l-1][0]; Kprev=C[l-1][2]; O,_,K=C[l]
  W=Ws[l].astype(np.float64)
  final=(l==k-1)
  # helper representation to target T from prior stable O plus previous kinks
  def rep_to(T):
   if len(T)==0:return np.zeros((D,0)), [np.zeros((len(Rs[j][0]) if Rs[j].ndim else 0,0)) for j in range(len(Rs))]
   if len(Oprev):
    Wot=W[np.ix_(Oprev,T)]
    Bt=B@Wot
    Ct=[Cj@Wot for Cj in Cs]
   else:
    Bt=np.zeros((D,len(T)));Ct=[np.zeros((R.shape[1],len(T))) for R in Rs[:-1]]
   Ct.append(W[np.ix_(Kprev,T)] if len(Kprev) else np.zeros((0,len(T))))
   return Bt,Ct
  if final:
   # full rows only for final kink outputs
   Bk,Ck=rep_to(K)
   if len(K):
    hk=X@Bk
    for R,Cj in zip(Rs,Ck):
     if R.shape[1]: hk += R@Cj
    yk=np.maximum(hk,0); meanK=yk.mean(axis=0,dtype=np.float64)
   else: yk=np.empty((N,0));meanK=np.empty(0)
   # stable-on output mean and pilot rows
   Bo,Co=rep_to(O)
   meanO=meanX@Bo
   for R,Cj in zip(Rs,Co):
    if R.shape[1]:meanO += R.mean(axis=0,dtype=np.float64)@Cj
   pred_mean=np.zeros(D,np.float64);pred_mean[O]=meanO;pred_mean[K]=meanK
   # compiled pilot all outputs
   qp=np.zeros((P,D),np.float64)
   if len(O):
    ho=Xp@Bo
    for Rp,Cj in zip(Rps,Co):
     if Rp.shape[1]:ho += Rp@Cj
    qp[:,O]=ho
   if len(K): qp[:,K]=yk[I]
   pred=R1*(pred_mean+(Y[I].astype(np.float64)-qp).mean(axis=0,dtype=np.float64))
   return pred
  # build kink activation at this layer and stable representation
  Bk,Ck=rep_to(K)
  if len(K):
   hk=X@Bk
   for R,Cj in zip(Rs,Ck):
    if R.shape[1]:hk += R@Cj
   R=np.maximum(hk,0);Rp=R[I]
  else:R=np.empty((N,0));Rp=np.empty((P,0))
  Bo,Co=rep_to(O)
  B=Bo;Cs=Co;Rs.append(R);Rps.append(Rp)
 raise RuntimeError

def cost_proxy(k,C,P,N):
 z=[len(c[2]) for c in C]
 # Mirrors two/three-layer report: k pilot full-layer equivalents + full-row kink propagation.
 ce=k*P/N
 for l in range(k):
  ce += z[l]/D
  for j in range(l):ce += z[j]*z[l]/D**2
 return (DEP-k+ce)/DEP

def eval_network(W,X,I,ks=range(2,7),rare=1):
 # retain only the maximum requested suffix to limit memory
 ks=list(ks); maxk=max(ks); a=X
 for w in W[:DEP-maxk]:
  a=a@w; np.maximum(a,0,out=a)
 acts=[a]; pres=[]
 for w in W[DEP-maxk:]:
  h=a@w;pres.append(h);a=np.maximum(h,0);acts.append(a)
 Y=acts[-1];truth=R1*Y.mean(axis=0,dtype=np.float64)
 out=[]
 for k in ks:
  offset=maxk-k
  anchor=acts[offset]
  pre=pres[offset:]
  C=[classify_layer(h[I],rare) for h in pre]
  pred=compile_predict(anchor,W[DEP-k:],C,I,Y)
  e=float(np.mean((pred-truth)**2));cr=cost_proxy(k,C,len(I),len(X))
  out.append({'k':k,'kinks':[len(c[2]) for c in C],'ons':[len(c[0]) for c in C], 'add_mse':e,'cost_ratio':cr,'score_proxy':cr*(1+e/BASE),'max_abs':float(np.max(np.abs(pred-truth)))})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--networks',type=int,default=4);ap.add_argument('--seed',type=int,default=2026080501);ap.add_argument('--cols',type=int,default=8);ap.add_argument('--rare',type=int,default=1);ap.add_argument('--max-k',type=int,default=6);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 X=sc.points(a.asset);I=sc.pidx(a.cols);rows=[];t=time.time()
 for n in range(a.start,a.start+a.networks):
  print('network',n,flush=True);r=eval_network(sc.net(a.seed+n*1009),X,I,range(2,a.max_k+1),a.rare)
  for x in r:x['network']=n;rows.append(x);print(json.dumps(x),flush=True)
  gc.collect()
 a.out.write_text(json.dumps({'config':{**vars(a),'asset':str(a.asset),'out':str(a.out)},'runtime':time.time()-t,'rows':rows},indent=2,default=str))
if __name__=='__main__':main()
