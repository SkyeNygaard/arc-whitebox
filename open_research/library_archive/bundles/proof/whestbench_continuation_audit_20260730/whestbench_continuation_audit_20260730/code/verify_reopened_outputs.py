from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/data/whestbench_continuation_20260730/reopened/reopened_paths_repro_20260730')
OUT=ROOT/'results'
LAMBDAS=(1e-4,1e-3,1e-2,1e-1,1.0,10.0)

def raw_mse(pred,ref): return float(np.mean((pred-ref)**2))
def cross_mse(pred,a,b): return float(np.mean((pred-a)*(pred-b)))
def fit_scalar(pairs):
    num=sum(float(np.dot(p,e)) for p,e in pairs); den=sum(float(np.dot(p,p)) for p,e in pairs)
    return num/max(den,1e-30)
def metric_summary(preds,arrs,seeds):
    raw=[];cross=[];br=[];bc=[]
    for s,p in zip(seeds,preds):
        a=arrs[s];raw.append(raw_mse(p,a['ref']));cross.append(cross_mse(p,a['ref_a'],a['ref_b']))
        br.append(raw_mse(a['base'],a['ref']));bc.append(cross_mse(a['base'],a['ref_a'],a['ref_b']))
    raw=np.array(raw);cross=np.array(cross);br=np.array(br);bc=np.array(bc)
    return {'pooled_raw_ratio':float(raw.sum()/br.sum()),'pooled_cross_ratio':float(cross.sum()/bc.sum()),
            'mean_raw_ratio':float(np.mean(raw/br)),'wins_raw':int(np.sum(raw<br)),'worst_raw_ratio':float(np.max(raw/br)),
            'per_seed_raw_ratio':{str(s):float(x) for s,x in zip(seeds,raw/br)},
            'candidate_raw':raw.tolist(),'candidate_cross':cross.tolist()}
def close(a,b,tol=2e-12):
    return abs(float(a)-float(b)) <= tol*max(1,abs(float(a)),abs(float(b)))
def assert_metrics(got,exp,label):
    for k in ['pooled_raw_ratio','pooled_cross_ratio','mean_raw_ratio','worst_raw_ratio']:
        assert close(got[k],exp[k]),(label,k,got[k],exp[k])
    assert got['wins_raw']==exp['wins_raw'],(label,'wins',got['wins_raw'],exp['wins_raw'])
    assert np.allclose(got['candidate_raw'],exp['candidate_raw'],rtol=2e-12,atol=1e-18),(label,'candidate_raw')
    assert np.allclose(got['candidate_cross'],exp['candidate_cross'],rtol=2e-12,atol=1e-18),(label,'candidate_cross')

# Load all raw vectors.
seeds=list(range(3000,3068))
arrs={s:dict(np.load(OUT/f'network_{s}.npz')) for s in seeds}
result={'passed':True,'checks':{}}

# Reproduce 48-network projected-ReLU terminal extension.
sel=json.load(open(OUT/'PROJECTED_RELU_EXTENSION_RESULTS.json'))
ss=list(range(3020,3068)); alpha=-2.0; key='nonlinear_k4_m64_b0_0.001'
preds=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in ss]
m=metric_summary(preds,arrs,ss)
for k,sk in [('pooled_raw_ratio','pooled_raw_ratio'),('pooled_cross_ratio','pooled_cross_ratio'),('mean_raw_ratio','mean_raw_ratio'),('worst_raw_ratio','worst')]:
    assert close(m[k],sel[sk]),('projected',k,m[k],sel[sk])
assert m['wins_raw']==sel['wins']
result['checks']['projected_relu_48']={k:m[k] for k in ['pooled_raw_ratio','pooled_cross_ratio','mean_raw_ratio','wins_raw','worst_raw_ratio']}

# Reproduce preregistered terminal descendant candidates.
term=json.load(open(OUT/'TERMINAL_DESCENDANT_RESULTS.json'))
pre=json.load(open(OUT/'DESCENDANT_PREREGISTRATION.json'))
ss=pre['terminal_synthetic_holdout']
for c in pre['candidates']:
    key=c['stored_prediction_key']; alpha=c['global_alpha']
    preds=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in ss]
    got=metric_summary(preds,arrs,ss); exp=term['candidates'][c['name']]['metrics']
    assert_metrics(got,exp,'terminal:'+c['name'])
result['checks']['terminal_descendants']={name:{k:v['metrics'][k] for k in ['pooled_raw_ratio','pooled_cross_ratio','wins_raw','worst_raw_ratio']} for name,v in term['candidates'].items()}

# Reproduce the extended train/validation ranking and the selected signed grid.
ext=json.load(open(OUT/'EXTENDED_VALIDATION_SUMMARY.json'))
train=list(range(3000,3004)); test=list(range(3004,3020))
keys=[k for k in arrs[3000] if k.startswith('poisson_') or k.startswith('nonlinear_')]
calc={}
for key in keys:
    alpha=float(np.clip(fit_scalar([(arrs[s][key]-arrs[s]['base'],arrs[s]['ref']-arrs[s]['base']) for s in train]),-2,2))
    tr=metric_summary([arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in train],arrs,train)
    va=metric_summary([arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in test],arrs,test)
    calc[key]=(alpha,tr,va)
    source=ext['poisson_ranked_train'] if key.startswith('poisson_') else ext['nonlinear_ranked_train']
    assert close(alpha,source[key]['alpha'])
    assert_metrics(tr,source[key]['train'],'train:'+key); assert_metrics(va,source[key]['validation'],'valid:'+key)
result['checks']['extended_families']={'count':len(calc),'best_poisson':next(iter(ext['poisson_ranked_train'])),'best_nonlinear':next(iter(ext['nonlinear_ranked_train']))}

# Reproduce frozen global signed validation (ridge=1e-4, total coefficient L1 cap=0.1).
signed_saved=json.load(open(OUT/'SIGNED_64_VALIDATION.json'))
for fam,akey in [('network','signed_network'),('random','signed_random')]:
    M=arrs[3000][akey].shape[0]; G=np.zeros((M,M)); b=np.zeros(M)
    for s in train:
        D=arrs[s][akey]; err=arrs[s]['ref']-arrs[s]['base']; G += D@D.T; b += D@err
    scale=max(np.trace(G)/M,1e-30); w=np.linalg.solve(G+1e-4*scale*np.eye(M),b)
    w *= min(1.0,0.1/max(float(np.sum(np.abs(w))),1e-30))
    ss=list(range(3004,3068))
    got=metric_summary([arrs[s]['base']+w@arrs[s][akey] for s in ss],arrs,ss)
    assert_metrics(got,signed_saved[fam]['fixed'],'signed:'+fam)
    result['checks']['signed_'+fam]={k:got[k] for k in ['pooled_raw_ratio','pooled_cross_ratio','mean_raw_ratio','wins_raw','worst_raw_ratio']}

Path('/mnt/data/whestbench_continuation_20260730/local_verification/reopened_independent_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
