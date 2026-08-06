from __future__ import annotations
import glob,json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/competition_relevance_20260730')
from nonlinear_basis_aggregation import methods
ROOT=Path('/mnt/data/competition_relevance_20260730/nonlinear_basis_aggregation')
files=sorted(glob.glob(str(ROOT/'basis_cases/*.npz')))
B={}
for f in files:
 m=re.search(r'n(\d+)_r(\d+)',f);B[(int(m.group(1)),int(m.group(2)))]=np.load(f)['B']
nets=sorted({n for n,r in B});rots=[3,11,97]
cases=[]
for n in nets:
 if not all((n,r) in B for r in rots):continue
 means={r:B[n,r].mean(0) for r in rots}
 for r in rots:
  ref=np.mean([means[q] for q in rots if q!=r],0);base,mm=methods(B[n,r]);cases.append({'network':n,'rot':r,'base':base,'ref':ref,'methods':mm})

def ratio(c,p):return float(np.mean((p-c['ref'])**2)/np.mean((c['base']-c['ref'])**2))
def fit_alpha(train,name):
 num=den=0.
 for c in train:
  e=c['base']-c['ref'];s=c['methods'][name]-c['base'];num+=float(e@s);den+=float(s@s)
 return float(np.clip(-num/max(den,1e-30),-5,5))
res={}
names=sorted(cases[0]['methods'])
for name in names:
 rows=[];orows=[]
 for n in sorted({c['network'] for c in cases}):
  tr=[c for c in cases if c['network']!=n];te=[c for c in cases if c['network']==n];a=fit_alpha(tr,name)
  for c in te:
   s=c['methods'][name]-c['base'];rows.append({'network':n,'rot':c['rot'],'alpha':a,'ratio':ratio(c,c['base']+a*s)})
   e=c['base']-c['ref'];ao=float(np.clip(-e@s/max(float(s@s),1e-30),-10,10));orows.append(ratio(c,c['base']+ao*s))
 rr=np.array([x['ratio'] for x in rows]);res[name]={'mean_ratio':float(rr.mean()),'median_ratio':float(np.median(rr)),'wins':int(np.sum(rr<1)),'n':len(rr),'worst':float(rr.max()),'oracle_scalar_mean':float(np.mean(orows)),'rows':rows}
# Multi-source ridge, nested lambda; use all method deltas but reduce correlated power family using handpicked dictionary.
sel=['median','trim0.1','huber1.5','geomed','central64','power0.5','power1.5','power2','power3']
def fit_multi(train,lam):
 # Accumulate output-coordinate observations without materializing giant design.
 G=np.zeros((len(sel),len(sel)));b=np.zeros(len(sel))
 for c in train:
  S=np.stack([c['methods'][k]-c['base'] for k in sel],1);e=c['base']-c['ref'];G+=S.T@S;b+=S.T@e
 return np.linalg.solve(G+lam*np.trace(G)/max(len(sel),1)*np.eye(len(sel))+1e-12*np.eye(len(sel)),-b)
def score_multi(train,lam):
 rr=[]
 for n in sorted({c['network'] for c in train}):
  tr=[c for c in train if c['network']!=n];te=[c for c in train if c['network']==n];coef=fit_multi(tr,lam)
  for c in te:
   S=np.stack([c['methods'][k]-c['base'] for k in sel],1);rr.append(ratio(c,c['base']+S@coef))
 return np.mean(rr)
rows=[]
for n in sorted({c['network'] for c in cases}):
 tr=[c for c in cases if c['network']!=n];te=[c for c in cases if c['network']==n];lam=min([1e-6,1e-4,1e-2,.1,1,10],key=lambda l:score_multi(tr,l));coef=fit_multi(tr,lam)
 for c in te:
  S=np.stack([c['methods'][k]-c['base'] for k in sel],1);rows.append({'network':n,'rot':c['rot'],'ratio':ratio(c,c['base']+S@coef),'lambda':lam})
rr=np.array([x['ratio'] for x in rows]);res['multi_nested']={'mean_ratio':float(rr.mean()),'median_ratio':float(np.median(rr)),'wins':int(np.sum(rr<1)),'n':len(rr),'worst':float(rr.max()),'rows':rows,'sources':sel}
out={'networks':nets,'cases':len(cases),'results':res};(ROOT/'ANALYSIS.json').write_text(json.dumps(out,indent=2));
for k,v in sorted(res.items(),key=lambda kv:kv[1]['mean_ratio']):print(k,{x:v[x] for x in ['mean_ratio','median_ratio','wins','n','worst'] if x in v},'oracle',v.get('oracle_scalar_mean'))
