from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
ROOT=Path(__file__).resolve().parent

def compact_cols(d):
 return [c for c in d if re.match(r'l(00|04|08|12|16|20|24|28|29|30|31)_(mean|std|rms|zero|max|fold_rel_mean|fold_rel_max|block_disp_mean|antipodal_imb_mean)$',c)]

def fit_scores(Xtr,Xte,method):
 imp=SimpleImputer(); A=imp.fit_transform(Xtr); B=imp.transform(Xte); sc=StandardScaler(); A=sc.fit_transform(A); B=sc.transform(B)
 u,s,vt=np.linalg.svd(A,full_matrices=False); V=vt[:min(6,len(vt))].T; P=A@V; Q=B@V
 if method=='mahalanobis':
  m=LedoitWolf().fit(P); return m.mahalanobis(P),m.mahalanobis(Q)
 m=IsolationForest(n_estimators=1000,random_state=20260729,contamination='auto').fit(P)
 return -m.score_samples(P),-m.score_samples(Q)

def grouped_policy(d,ratio_col,method,q=.95):
 cols=compact_cols(d); X=d[cols].to_numpy(); g=d.network_seed.to_numpy(); apply=np.ones(len(d),bool); score=np.zeros(len(d)); thresholds=[]
 for seed in np.unique(g):
  tr=g!=seed; te=g==seed; a,b=fit_scores(X[tr],X[te],method); th=float(np.quantile(a,q)); apply[te]=b<=th; score[te]=b
  thresholds.append({'seed':int(seed),'threshold':th,'scores':[float(x) for x in b],'apply':[bool(x) for x in apply[te]]})
 return apply,score,thresholds

def metrics(d,apply,ratio_col,can=None,hard=None):
 def one(mask):
  z=d[mask]; a=apply[mask]; pol=np.where(a,z[ratio_col],1.); b=z.baseline_mse.to_numpy();
  return {'n':len(z),'coverage':float(a.mean()),'pooled_ratio':float(np.sum(b*pol)/np.sum(b)),'wins':int(np.sum(pol<1)),'worst':float(np.max(pol)),'p90':float(np.quantile(pol,.9))}
 out={'all':one(np.ones(len(d),bool))}
 if can is not None: out['canonical']=one(can)
 if hard is not None: out['hard']=one(hard)
 return out

def frozen_train_apply(train,test,method,q=.95):
 cols=compact_cols(train); cols=[c for c in cols if c in test]
 a,b=fit_scores(train[cols].to_numpy(),test[cols].to_numpy(),method); th=float(np.quantile(a,q)); return b<=th,b,th

def main():
 panel=pd.read_csv(ROOT/'legal_features_and_labels.csv'); radial=pd.read_csv(ROOT/'radial16_full_features.csv')
 hardbases={205215497,493891104,422494190,680708219}; hard=panel.network_seed.isin(hardbases).to_numpy(); can=panel.domain.eq('canonical').to_numpy()
 out={'panel':{},'radial_grouped':{},'cross_cohort_k32_to_radial':{},'combined_panel':{}}
 policies={}
 for m in ['mahalanobis','isoforest']:
  ap,score,detail=grouped_policy(panel,'candidate_ratio',m,.95); policies[m]=ap
  out['panel'][m]={**metrics(panel,ap,'candidate_ratio',can,hard),'details':detail,
                   'catastrophic_scores':panel.assign(score=score,apply=ap).sort_values('candidate_ratio',ascending=False)[['network_seed','rotation_seed','candidate_ratio','score','apply']].head(6).to_dict('records')}
  ar,sr,dr=grouped_policy(radial,'ratio_alpha04',m,.95)
  out['radial_grouped'][m]={**metrics(radial,ar,'ratio_alpha04'),'details':dr,
    'negative_case':radial.assign(score=sr,apply=ar).sort_values('alpha_opt').head(1)[['network_seed','alpha_opt','ratio_alpha04','score','apply']].to_dict('records')[0]}
  train=panel[panel.domain.eq('canonical')].copy(); ax,sx,th=frozen_train_apply(train,radial,m,.95)
  out['cross_cohort_k32_to_radial'][m]={**metrics(radial,ax,'ratio_alpha04'),'threshold':th,
    'abstained':radial.assign(score=sx,apply=ax).loc[~ax,['network_seed','alpha_opt','ratio_alpha04','score']].to_dict('records')}
 # combine original l08 grouped policy from results decisions with fixed one-class OOF.
 dec=pd.read_csv(ROOT/'results'/'oof_decisions.csv'); key=['network_seed','rotation_seed']; tmp=panel.merge(dec[key+['fixed_l08_q95_apply']],on=key,how='left')
 for m,ap in policies.items():
  comb=tmp.fixed_l08_q95_apply.to_numpy(bool)&ap
  out['combined_panel'][m]=metrics(panel,comb,'candidate_ratio',can,hard)
 (ROOT/'results'/'ONECLASS_ABSTENTION_RESULTS.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
