#!/usr/bin/env python3
import json, glob, os, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

ROOT=Path('/mnt/data/continued_review_inputs/t4_orientation/full/T4_legal_layer31_anchor_hedge_20260729/T41_EXACT_ROTATIONS/results')
OUT=Path('/mnt/data/continued_review_outputs/orientation_odd_t4')
OUT.mkdir(parents=True,exist_ok=True)
BOUNDS=np.array([[0.,2.],[0.,1.],[0.,1.]])
ALPHAS=[1e-6,1e-4,1e-2,1e-1,1.,10.,100.,1000.]

rows=[]
coefmap={}
with open(ROOT/'T41_ORACLE_CEILINGS.json') as f:
    oc=json.load(f)
for x in oc['per_rotation_coefficients']:
    coefmap[(x['network_id'],x['rotation_index'])]=np.array(x['coefficients'],float)

# deterministic cosine reference axes over canonical output coordinate ordering
idx=np.arange(256)
axes=[]
for k in (1,2):
    v=np.cos(np.pi*(idx+0.5)*k/256.0); v=v/np.linalg.norm(v); axes.append(v)

def safe_cos(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else 0.0

def odd_features(c, reduced, main_delta):
    n=np.linalg.norm(c)
    if n==0: return [0.0]*9
    j=int(np.argmax(np.abs(c)))
    # all values flip under c -> -c while the runtime reference remains fixed
    return [
        float(c[0]/n),
        float(c[-1]/n),
        float(c.mean()*np.sqrt(c.size)/n),
        float(c[j]/n),
        safe_cos(c,reduced),
        safe_cos(c,main_delta),
        float(np.dot(c,axes[0])/n),
        float(np.dot(c,axes[1])/n),
        float((np.sum(c[:128])-np.sum(c[128:]))/(np.sqrt(c.size)*n)),
    ]

for vp in sorted(glob.glob(str(ROOT/'vectors/*.npz'))):
    z=np.load(vp)
    nid=int(z['network_id']); rot=int(z['rotation_index'])
    rp=ROOT/'records'/Path(vp).with_suffix('.json').name
    rec=json.load(open(rp))
    dirs=[z['p128_projection'].astype(float), z['p2_correction'].astype(float), z['p4_correction'].astype(float)]
    reduced=z['reduced_base'].astype(float); base=z['basefull'].astype(float); truth=z['truth_y'].astype(float)
    even=np.array(list(rec['features'].values()),float)
    odd=np.array(sum((odd_features(c,reduced,base-reduced) for c in dirs),[]),float)
    rows.append(dict(nid=nid,rot=rot,even=even,odd=odd,dirs=np.stack(dirs),reduced=reduced,base=base,truth=truth,
                     target=coefmap[(nid,rot)], base_sse=float(np.sum((base-truth)**2))))
rows=sorted(rows,key=lambda r:(r['nid'],r['rot']))
groups=sorted(set(r['nid'] for r in rows))

# exact final SSE helpers
def sse_rows(rs, coeffs):
    total=0.0
    for r,a in zip(rs,coeffs):
        pred=r['reduced']+np.tensordot(a,r['dirs'],axes=1)
        total += float(np.sum((pred-r['truth'])**2))
    return total

def base_sse(rs): return sum(r['base_sse'] for r in rs)

def fit_constant(rs):
    # bounded pooled least-squares coefficients
    def fun(a): return sse_rows(rs,[a]*len(rs))
    res=minimize(fun, x0=np.array([.3,.1,.1]), bounds=[tuple(x) for x in BOUNDS], method='L-BFGS-B')
    return np.clip(res.x,BOUNDS[:,0],BOUNDS[:,1])

def mat(rs,fs):
    if fs=='even': return np.stack([r['even'] for r in rs])
    if fs=='odd': return np.stack([r['odd'] for r in rs])
    if fs=='all': return np.stack([np.r_[r['even'],r['odd']] for r in rs])
    raise ValueError(fs)

def fit_predict(train,test,fs,alpha):
    X=mat(train,fs); Xt=mat(test,fs); y=np.stack([r['target'] for r in train])
    sc=StandardScaler().fit(X)
    model=Ridge(alpha=alpha,fit_intercept=True).fit(sc.transform(X),y)
    pred=model.predict(sc.transform(Xt))
    return np.clip(pred,BOUNDS[:,0],BOUNDS[:,1])

def nested_oof(fs):
    out=[]; choices=[]
    for g in groups:
        train=[r for r in rows if r['nid']!=g]; test=[r for r in rows if r['nid']==g]
        inner_groups=sorted(set(r['nid'] for r in train))
        scores=[]
        for alpha in ALPHAS:
            s=0.0
            for ig in inner_groups:
                itr=[r for r in train if r['nid']!=ig]; iva=[r for r in train if r['nid']==ig]
                p=fit_predict(itr,iva,fs,alpha)
                s+=sse_rows(iva,p)
            scores.append(s)
        alpha=ALPHAS[int(np.argmin(scores))]
        choices.append(alpha)
        p=fit_predict(train,test,fs,alpha)
        for r,a in zip(test,p): out.append((r,a))
    out=sorted(out,key=lambda x:(x[0]['nid'],x[0]['rot']))
    return [x[0] for x in out],np.stack([x[1] for x in out]),choices

def constant_oof():
    out=[]
    for g in groups:
        train=[r for r in rows if r['nid']!=g]; test=[r for r in rows if r['nid']==g]
        a=fit_constant(train)
        for r in test: out.append((r,a.copy()))
    out=sorted(out,key=lambda x:(x[0]['nid'],x[0]['rot']))
    return [x[0] for x in out],np.stack([x[1] for x in out])

def metrics(rs,pred):
    b=base_sse(rs); c=sse_rows(rs,pred)
    ratios=[]
    wins=0
    for r,a in zip(rs,pred):
        cs=sse_rows([r],[a]); rr=cs/r['base_sse']; ratios.append(rr); wins+=rr<1
    return dict(candidate_over_baseline=c/b,raw_gain=b/c,wins=wins,n=len(rs),median=float(np.median(ratios)),p90=float(np.quantile(ratios,.9)),worst=float(np.max(ratios)),coeff_mean=np.mean(pred,axis=0).tolist(),coeff_sd=np.std(pred,axis=0).tolist())

results={}
crs,cp=constant_oof(); results['constant_oof']=metrics(crs,cp)
for fs in ['even','odd','all']:
    rs,p,ch=nested_oof(fs); results[fs+'_ridge_oof']=metrics(rs,p); results[fs+'_ridge_oof']['selected_alpha_counts']={str(a):ch.count(a) for a in ALPHAS if a in ch}
# all-data diagnostics only
oracle=np.stack([r['target'] for r in rows]); results['per_case_box_oracle']=metrics(rows,oracle)
zero=np.zeros((len(rows),3)); results['reduced_no_correction']=metrics(rows,zero)
# direct target predictability diagnostics under outer OOF predictions
for fs in ['even','odd','all']:
    rs,p,_=nested_oof(fs); y=np.stack([r['target'] for r in rs])
    diag=[]
    for j,name in enumerate(['c17','p2','p4']):
        ssr=float(np.sum((p[:,j]-y[:,j])**2)); sst=float(np.sum((y[:,j]-y[:,j].mean())**2))
        corr=float(np.corrcoef(p[:,j],y[:,j])[0,1]) if np.std(p[:,j])>0 else 0.0
        diag.append(dict(source=name,r2=1-ssr/sst,corr=corr,positive_action_accuracy=float(np.mean((p[:,j]>0.05)==(y[:,j]>0.05)))))
    results[fs+'_coefficient_diagnostics']=diag

payload={
 'status':'DEVELOPMENT_ONLY_DIAGNOSTIC',
 'n_rows':len(rows),'n_base_networks':len(groups),'rotations_per_base':3,
 'feature_dimensions':{'even':len(rows[0]['even']),'odd':len(rows[0]['odd']),'all':len(rows[0]['even'])+len(rows[0]['odd'])},
 'odd_feature_definition':'Nine deterministic sign-odd contractions per source: fixed coordinates, normalized mean, signed max-magnitude pivot, contractions with reduced output and primary-delta references, two cosine axes, and half-block contrast.',
 'bounds':BOUNDS.tolist(), 'ridge_grid':ALPHAS, 'results':results,
 'interpretation':'This is a grouped development diagnostic on already-exposed rows. It can falsify claims about the sufficiency of the original even feature map, but cannot promote an estimator without a newly frozen validation cohort.'
}
out=OUT/'ORIENTATION_ODD_T4_DIAGNOSTIC.json'; out.write_text(json.dumps(payload,indent=2))
# row-level predictions for audit
with open(OUT/'ORIENTATION_ODD_T4_SUMMARY.txt','w') as f:
    for k,v in results.items(): f.write(f'{k}: {v}\n')
print(json.dumps(payload,indent=2))
