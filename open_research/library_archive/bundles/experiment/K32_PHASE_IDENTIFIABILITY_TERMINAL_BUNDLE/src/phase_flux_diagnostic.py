#!/usr/bin/env python3
from pathlib import Path
import glob,json,sys
import numpy as np
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# load helper by exec? replicate model load
M=np.load(ROOT/'hierarchical_phase_flux/hierarchical_phase_flux_model.npz')
template=M['template'];xm=M['x_mean'];xs=M['x_std'];pc=M['phase_coef'];gm=M['global_mean'];gs=M['global_std'];oc=M['offset_coef'];oi=float(M['offset_intercept']);cal=float(M['calibration'])

def load(pattern):
 rows=[]
 for p in sorted(glob.glob(str(pattern))):
  with np.load(p,allow_pickle=True) as z:
   for j,rot in enumerate(z['rotation_seeds']):
    r={'id':int(z['network_id']),'rot':int(rot),'g':z['global_features'][j].astype(float),'x':np.r_[z['sample_prediction'][j],z['baseline_prediction'][j],z['sample_prediction'][j]-z['baseline_prediction'][j]].astype(float),'delta':z['target_delta'][j].astype(float),'beta':z['beta_bar'][j].astype(float),'sample':z['sample_prediction'][j].astype(float),'truth':.5*(z['truth_half1'][j]+z['truth_half2'][j]),'base':float(z['base_mse'][j])}
    c=template@r['beta'];r['corr']=c;r['scale']=float(c@(r['truth']-r['sample'])/max(c@c,1e-30));rows.append(r)
 return rows

def predparts(rows):
 X=np.stack([r['x'] for r in rows]);G=np.stack([r['g'] for r in rows]);b=((X-xm)/xs)@pc;a=((G-gm)/gs)@oc+oi;return a,b,cal*(a+b)
def metrics(rows,s):
 ms=[];bs=[];rr=[]
 for r,v in zip(rows,s):
  m=np.mean((r['sample']+v*r['corr']-r['truth'])**2);ms.append(m);bs.append(r['base']);rr.append(m/r['base'])
 ms=np.array(ms);bs=np.array(bs);rr=np.array(rr);y=np.array([r['scale'] for r in rows])
 return {'n':len(rows),'aggregate':float(ms.sum()/bs.sum()),'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max()),'corr':float(pearsonr(s,y).statistic) if np.std(s)>0 else None,'sign':float(np.mean((s>0)==(y>0)))}
coh={'validation':load(ROOT/'data/validation_network_*.npz'),'primary':load(ROOT/'final_test_data/test_network_*.npz'),'rescue':load(ROOT/'rescue/final_test_data/test_network_*.npz')}
out={}
for name,rows in coh.items():
 a,b,p=predparts(rows);true=np.array([r['scale'] for r in rows])
 # oracle decomposition per base: true base mean and true rotation deviation
 means={nid:np.mean([r['scale'] for r in rows if r['id']==nid]) for nid in set(r['id'] for r in rows)}
 om=np.array([means[r['id']] for r in rows]);od=true-om
 variants={'full':p,'offset_only':cal*a,'phase_only':cal*b,'oracle_offset_plus_pred_phase':om+cal*b,'pred_offset_plus_oracle_phase':cal*a+od,'oracle_offset_only':om,'oracle_phase_only':od}
 o={k:metrics(rows,v) for k,v in variants.items()}
 for rot in sorted(set(r['rot'] for r in rows)):
  ix=[i for i,r in enumerate(rows) if r['rot']==rot];o[f'rotation_{rot}']={k:metrics([rows[i] for i in ix],v[ix]) for k,v in variants.items()}
 out[name]=o
(ROOT/'hierarchical_phase_flux/diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
