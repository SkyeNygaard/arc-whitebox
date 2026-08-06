from __future__ import annotations
import json,time,math
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT=Path('/mnt/data/oracle_continuation_20260730/edge_dws'); DATA=OUT/'edge_dws_labels_raw.npz'
SPL=json.loads((OUT/'split_registry.json').read_text())['splits']; t0=time.time(); D=256; DEPTH=32
with np.load(DATA,allow_pickle=False) as z:
 ids=np.asarray(z['base_network_id']).astype(str); e0=np.asarray(z['baseline_error'],float); j=np.asarray(z['replay_jacobian'],float)[:,:,0]
 anchor=np.asarray(z['anchor_coeffs'],float)[:,0]; y=np.asarray(z['target_coeffs'],float)[:,0]
# Generate exact weights but retain only cheap layer invariants: mean, std, absmean, row-norm mean/std, col-norm mean/std.
X=[]; scale=math.sqrt(2.0/D)
for s in map(int,ids):
 rng=np.random.default_rng(s); row=[]
 for _ in range(DEPTH):
  w=(rng.standard_normal((D,D))*scale).astype(np.float32)
  rn=np.sqrt(np.sum(w*w,axis=1,dtype=np.float64)); cn=np.sqrt(np.sum(w*w,axis=0,dtype=np.float64))
  row.extend([float(w.mean()),float(w.std()),float(np.mean(np.abs(w))),float(rn.mean()),float(rn.std()),float(cn.mean()),float(cn.std())])
 X.append(row)
X=np.asarray(X,float); idx={k:np.flatnonzero(np.isin(ids,np.asarray(v,str))) for k,v in SPL.items()}
def loss(ix,c): return float(np.mean((e0[ix]+c[:,None]*j[ix])**2))
def fit_ridge():
 best=None
 for a in [0.01,.1,1,10,100,1000,10000]:
  m=make_pipeline(StandardScaler(),Ridge(alpha=a));m.fit(X[idx['train']],y[idx['train']]);p=np.clip(m.predict(X[idx['calibration']]),-20,20);l=loss(idx['calibration'],p)
  if best is None or l<best[0]:best=(l,a,m)
 return best
_,ra,ridge=fit_ridge()
extra=ExtraTreesRegressor(n_estimators=200,max_depth=3,min_samples_leaf=4,max_features=.5,random_state=20260730,n_jobs=-1).fit(X[idx['train']],y[idx['train']])
rf=RandomForestRegressor(n_estimators=200,max_depth=3,min_samples_leaf=4,max_features=.5,random_state=20260730,n_jobs=-1).fit(X[idx['train']],y[idx['train']])
cal=idx['calibration'];const=float(-np.sum(e0[cal]*j[cal])/np.sum(j[cal]*j[cal]))
def boot(base,cand,n=5000):
 rng=np.random.default_rng(20260730);N=len(base);v=[]
 for _ in range(n):
  q=rng.integers(0,N,N);v.append(cand[q].sum()/base[q].sum())
 return [float(x) for x in np.quantile(v,[.025,.5,.975])]
def met(ix,c,cr=1.05405):
 b=np.mean(e0[ix]**2,1);d=np.mean((e0[ix]+c[:,None]*j[ix])**2,1);r=d/b;p=float(d.sum()/b.sum())
 return dict(pooled_candidate_over_baseline=p,raw_gain=1/p,wins=int((d<b).sum()),median=float(np.median(r)),p90=float(np.quantile(r,.9)),worst=float(r.max()),bootstrap_ratio_ci95=boot(b,d),adjusted_candidate_over_baseline=p*cr,coefficient_mean=float(c.mean()),coefficient_std=float(c.std()),coefficient_corr_with_oracle=float(np.corrcoef(c,y[ix])[0,1]) if c.std()>0 else 0)
rep={'status':'completed','feature_dim':X.shape[1],'ridge_alpha':ra,'constant_coefficient':const,'splits':{},'runtime_seconds':None,'scope':'cheap permutation-invariant layerwise weight summaries'}
for sp in ['validation','test']:
 ix=idx[sp];rep['splits'][sp]={'anchor_frozen':met(ix,anchor[ix]),'constant_calibrated':met(ix,np.full(len(ix),const)),'ridge':met(ix,np.clip(ridge.predict(X[ix]),-20,20)),'extra_trees':met(ix,np.clip(extra.predict(X[ix]),-20,20)),'random_forest':met(ix,np.clip(rf.predict(X[ix]),-20,20)),'oracle_scalar':met(ix,y[ix],1.0)}
rep['runtime_seconds']=time.time()-t0
rep['gate']={}
for n,r in rep['splits']['test'].items():
 if n=='oracle_scalar':continue
 g={'raw_gain_ge_1_15':r['raw_gain']>=1.15,'ratio_ci_upper_lt_1':r['bootstrap_ratio_ci95'][2]<1,'worst_le_1_10':r['worst']<=1.10,'adjusted_lt_1':r['adjusted_candidate_over_baseline']<1};g['pass']=all(g.values());rep['gate'][n]=g
(OUT/'WEIGHT_SUMMARY_PHASE_BASELINES.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
