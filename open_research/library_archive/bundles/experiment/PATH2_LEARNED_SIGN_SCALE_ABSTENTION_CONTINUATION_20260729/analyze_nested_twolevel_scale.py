from __future__ import annotations
import json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'legal_features_and_labels.csv')
CANP=ROOT/'canonical24_quadratics_audited';HARDP=ROOT/'hardpanel_quadratics_audited'
A=[]
for _,z in D.iterrows():
 panel=CANP if z.domain=='canonical' else HARDP
 v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;dd=p-b
 A.append([np.mean(e*e),2*np.mean(e*dd),np.mean(dd*dd)])
A=np.array(A);G=D.network_seed.to_numpy();Z=D.l08_fold_rel_mean.to_numpy();CAN=D.domain.eq('canonical').to_numpy();HARD=D.in_hard_panel.astype(bool).to_numpy();LOGO=LeaveOneGroupOut()

def eval_alpha(alpha,ix):
 m=A[ix,0]+alpha*A[ix,1]+alpha*alpha*A[ix,2];r=m/A[ix,0];can=CAN[ix];hard=HARD[ix]
 cp=m[can].sum()/A[ix,0][can].sum() if can.any() else m.sum()/A[ix,0].sum();hw=r[hard].max() if hard.any() else 1.;hp=np.quantile(r[hard],.9) if hard.any() else 1.
 return float(cp+.8*max(0,hw-1.02)+.2*max(0,hp-1.0)),float(cp),float(hw)

def oof_for_cfg(ix,q,lo,hi):
 # Within supplied rows, leave each base group out to set threshold using other groups.
 out=np.zeros(len(ix));g=G[ix];z=Z[ix]
 for tr,te in LeaveOneGroupOut().split(ix,groups=g):
  th=np.quantile(z[tr],q);out[te]=np.where(z[te]>th,lo,hi)
 return out
pred=np.zeros(len(D));choices=[]
qs=[.9,.925,.95,.975];los=[.2,.25,.3,.35,.4,.45];his=[.45,.5,.55]
for otr,ote in LOGO.split(D,groups=G):
 best=None
 for q,lo,hi in itertools.product(qs,los,his):
  if lo>hi:continue
  a=oof_for_cfg(otr,q,lo,hi);sc,cp,hw=eval_alpha(a,otr);key=(sc,hw,cp,q,lo,hi)
  if best is None or key<best[0]:best=(key,q,lo,hi)
 _,q,lo,hi=best;th=np.quantile(Z[otr],q);pred[ote]=np.where(Z[ote]>th,lo,hi)
 choices.append({'held_seed':int(G[ote][0]),'q':q,'high_risk_alpha':lo,'ordinary_alpha':hi,'threshold':float(th),'predictions':pred[ote].tolist(),'inner_score':float(best[0][0])})

def metrics(a):
 m=A[:,0]+a*A[:,1]+a*a*A[:,2];r=m/A[:,0]
 def sub(mask):return {'pooled_ratio':float(m[mask].sum()/A[:,0][mask].sum()),'worst':float(r[mask].max()),'p90':float(np.quantile(r[mask],.9)),'wins':int((r[mask]<1).sum()),'alpha_mean':float(a[mask].mean())}
 return {'canonical':sub(CAN),'hard':sub(HARD),'all':sub(np.ones(len(D),bool)),'alphas':a.tolist(),'ratios':r.tolist()}
# Fixed externally interpretable comparisons, threshold recomputed LOGO q95.
def fixed(lo,hi,q=.95):
 a=np.zeros(len(D))
 for tr,te in LOGO.split(D,groups=G):a[te]=np.where(Z[te]>np.quantile(Z[tr],q),lo,hi)
 return a
pol={'nested_twolevel':metrics(pred),'q95_04_05':metrics(fixed(.4,.5)),'q95_03_05':metrics(fixed(.3,.5)),'fixed_045':metrics(np.full(len(D),.45)),'fixed_05':metrics(np.full(len(D),.5))}
out={'protocol':{'outer':'leave-one-base-network-out','inner':'leave-one-base-network-out threshold evaluation inside each outer training set','q_grid':qs,'high_risk_alpha_grid':los,'ordinary_alpha_grid':his,'all_rotations_grouped':True},'policies':pol,'choices':choices}
(ROOT/'results'/'NESTED_TWOLEVEL_SCALE_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:{'canonical':v['canonical'],'hard':v['hard']} for k,v in pol.items()},indent=2));print(pd.Series([f"q{x['q']}_lo{x['high_risk_alpha']}_hi{x['ordinary_alpha']}" for x in choices]).value_counts())
