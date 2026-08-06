import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/whest_reopened')
import reopened_path_experiments as r
OUT=r.OUT
train=list(range(3000,3004));test=list(range(3004,3020));allseeds=train+test
arrs={s:dict(np.load(OUT/f'network_{s}.npz')) for s in allseeds}
summary={'train':train,'frozen_validation':test,'selection':'all hyperparameters and scalar coefficients selected on seeds 3000-3003 only'}
# Branch 2/3
families={}
keys=[k for k in arrs[train[0]] if k.startswith('poisson_') or k.startswith('nonlinear_')]
for key in keys:
    alpha=float(np.clip(r.fit_scalar([(arrs[s][key]-arrs[s]['base'],arrs[s]['ref']-arrs[s]['base']) for s in train]),-2,2))
    trpred=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in train]
    tepred=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in test]
    families[key]={'alpha':alpha,'train':r.metric_summary(trpred,arrs,train),'validation':r.metric_summary(tepred,arrs,test)}
summary['poisson_ranked_train']=dict(sorted(((k,v) for k,v in families.items() if k.startswith('poisson_')),key=lambda kv:kv[1]['train']['pooled_raw_ratio']))
summary['nonlinear_ranked_train']=dict(sorted(((k,v) for k,v in families.items() if k.startswith('nonlinear_')),key=lambda kv:kv[1]['train']['pooled_raw_ratio']))
# Branch 4, exactly same sweep and selection rule as original.
summary['signed']={}
for fam,akey in [('network','signed_network'),('random','signed_random')]:
    M=arrs[train[0]][akey].shape[0];G=np.zeros((M,M));b=np.zeros(M)
    for s in train:
        D=arrs[s][akey];err=arrs[s]['ref']-arrs[s]['base'];G+=D@D.T;b+=D@err
    scale=max(np.trace(G)/M,1e-30);cand={}
    for lam in r.LAMBDAS:
        w=np.linalg.solve(G+lam*scale*np.eye(M),b);mass=float(np.sum(np.abs(w)))
        for cap in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0]:
            wc=w*min(1.0,cap/max(mass,1e-30));key=f'lam{lam}_cap{cap}'
            trpred=[arrs[s]['base']+wc@arrs[s][akey] for s in train]
            tepred=[arrs[s]['base']+wc@arrs[s][akey] for s in test]
            cand[key]={'ridge':lam,'cap':cap,'negative_mass':float(np.sum(np.abs(wc))),
                       'train':r.metric_summary(trpred,arrs,train),'validation':r.metric_summary(tepred,arrs,test)}
    summary['signed'][fam]={'ranked_train':dict(sorted(cand.items(),key=lambda kv:kv[1]['train']['pooled_raw_ratio']))}
(OUT/'EXTENDED_VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2))
for name,d in [('poisson',summary['poisson_ranked_train']),('nonlinear',summary['nonlinear_ranked_train'])]:
    k,v=next(iter(d.items()));print(name,k,'alpha',v['alpha'],'train',v['train']['pooled_raw_ratio'],'valid',v['validation']['pooled_raw_ratio'],'wins',v['validation']['wins_raw'],'worst',v['validation']['worst_raw_ratio'],'cross',v['validation']['pooled_cross_ratio'])
for fam in ['network','random']:
    k,v=next(iter(summary['signed'][fam]['ranked_train'].items()));print('signed',fam,k,'train',v['train']['pooled_raw_ratio'],'valid',v['validation']['pooled_raw_ratio'],'wins',v['validation']['wins_raw'],'worst',v['validation']['worst_raw_ratio'],'cross',v['validation']['pooled_cross_ratio'])
