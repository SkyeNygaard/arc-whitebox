from __future__ import annotations
import json, re, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'legal_features_and_labels.csv')
G=D.network_seed.to_numpy(); Y=D.candidate_ratio.to_numpy(); LOGY=np.log(Y); B=D.baseline_mse.to_numpy()
CAN=D.domain.eq('canonical').to_numpy(); HARD=D.in_hard_panel.astype(bool).to_numpy()
META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds'}
NUM=[c for c in D if c not in META and np.issubdtype(D[c].dtype,np.number)]
LOGO=LeaveOneGroupOut()

def metrics(apply:np.ndarray)->dict:
    pol=np.where(apply,Y,1.0)
    def pooled(mask): return float(np.sum(B[mask]*pol[mask])/np.sum(B[mask]))
    return {
      'canonical_candidate_over_baseline':pooled(CAN),
      'canonical_coverage':float(apply[CAN].mean()),
      'canonical_wins':int(np.sum(pol[CAN]<1.0)),
      'canonical_worst':float(np.max(pol[CAN])),
      'hard_candidate_over_baseline':pooled(HARD),
      'hard_coverage':float(apply[HARD].mean()),
      'hard_worst':float(np.max(pol[HARD])),
      'hard_p90':float(np.quantile(pol[HARD],.9)),
      'all_candidate_over_baseline':pooled(np.ones(len(D),bool)),
    }

def bootstrap_canonical(apply:np.ndarray,n=50000,seed=20260729):
    rng=np.random.default_rng(seed); ids=np.flatnonzero(CAN); pol=np.where(apply,Y,1.0); vals=np.empty(n)
    for i in range(n):
        ix=rng.choice(ids,size=len(ids),replace=True)
        vals[i]=np.sum(B[ix]*pol[ix])/np.sum(B[ix])
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]

def fixed_quantile_rule(feature='l08_fold_rel_mean',q=.95):
    apply=np.ones(len(D),bool); thresholds=[]
    vals=D[feature].to_numpy()
    for tr,te in LOGO.split(D,Y,G):
        th=float(np.quantile(vals[tr],q)); apply[te]=vals[te]<=th
        thresholds.append({'held_seed':int(G[te][0]),'threshold':th,'applied':[bool(x) for x in apply[te]]})
    return apply,thresholds

def nested_fold_family():
    feats=[c for c in D if re.match(r'l(00|04|08|12|16|20|24|28|29|30|31)_fold_rel_(mean|q50|q75|q90|max)$',c)]
    outer=np.ones(len(D),bool); choices=[]
    for otr,ote in LOGO.split(D,Y,G):
        td=D.iloc[otr].reset_index(drop=True); tg=td.network_seed.to_numpy(); ty=td.candidate_ratio.to_numpy(); tb=td.baseline_mse.to_numpy()
        tc=td.domain.eq('canonical').to_numpy(); thard=td.in_hard_panel.astype(bool).to_numpy()
        best=None
        for f in feats:
            vv=td[f].to_numpy()
            for q in (.85,.9,.925,.95,.975):
                ia=np.ones(len(td),bool)
                for itr,ite in LeaveOneGroupOut().split(td,ty,tg): ia[ite]=vv[ite]<=np.quantile(vv[itr],q)
                pol=np.where(ia,ty,1.0)
                cr=np.sum(tb[tc]*pol[tc])/np.sum(tb[tc]); cov=ia[tc].mean()
                hw=np.max(pol[thard]) if np.any(thard) else 1.; hp=np.quantile(pol[thard],.9) if np.any(thard) else 1.
                score=cr+.5*max(0.,hw-1.1)+.15*max(0.,hp-1.05)+.3*max(0.,.7-cov)
                key=(score,-q,f)
                if best is None or key<best[0]: best=(key,f,q)
        _,f,q=best; threshold=float(np.quantile(D[f].to_numpy()[otr],q)); outer[ote]=D[f].to_numpy()[ote]<=threshold
        choices.append({'held_seed':int(G[ote][0]),'feature':f,'q':q,'threshold':threshold,'applied':[bool(x) for x in outer[ote]]})
    return outer,choices

def ridge_suite():
    sets={
      'rotation':[c for c in NUM if ('block_' in c or 'fold_' in c or 'antipodal_' in c or c.startswith('anchor_'))],
      'anchor_suffix':[c for c in NUM if c.startswith('anchor_') or re.match(r'l(2[89]|3[01])_',c) or c.startswith('w_suffix_') or re.match(r'w(28|29|30|31)_',c)],
      'late':[c for c in NUM if re.match(r'l(20|24|28|29|30|31)_',c) or c.startswith('anchor_')],
    }
    rows=[]
    for sn,cs in sets.items():
      X=D[cs].to_numpy()
      for k in (4,8,16,32):
       for alpha in (1.,10.,100.,1000.):
        pred=np.zeros(len(D))
        for tr,te in LOGO.split(X,LOGY,G):
          m=Pipeline([('imp',SimpleImputer()),('scale',StandardScaler()),('select',SelectKBest(f_regression,k=k)),('ridge',Ridge(alpha=alpha))])
          m.fit(X[tr],LOGY[tr]); pred[te]=np.ravel(m.predict(X[te]))
        rows.append({'feature_set':sn,'k':k,'alpha':alpha,'rmse_log':float(np.sqrt(np.mean((pred-LOGY)**2))),
                     'pearson':float(np.corrcoef(pred,LOGY)[0,1]),'spearman':float(spearmanr(pred,LOGY).statistic),
                     'worst_case_predicted_ratio':float(np.exp(pred[np.argmax(Y)]))})
    return sorted(rows,key=lambda x:x['rmse_log'])

def main():
    full=np.ones(len(D),bool)
    fixed,fixeddetail=fixed_quantile_rule()
    nested,nesteddetail=nested_fold_family()
    ridge=ridge_suite()
    decisions=D[['network_seed','rotation_seed','domain','in_hard_panel','baseline_mse','oracle_ratio','candidate_ratio']].copy()
    decisions['fixed_l08_q95_apply']=fixed; decisions['fixed_l08_q95_policy_ratio']=np.where(fixed,Y,1.)
    decisions['nested_family_apply']=nested; decisions['nested_family_policy_ratio']=np.where(nested,Y,1.)
    decisions.to_csv(ROOT/'results'/'oof_decisions.csv',index=False)
    out={
      'protocol':{'examples':len(D),'base_network_groups':int(D.network_seed.nunique()),'canonical_examples':int(CAN.sum()),'hard_panel_examples':int(HARD.sum()),
                  'split':'leave-one-base-network-out; all rotations grouped','fixed_exploratory_rule':'apply K32 iff layer-8 six-fold basis-block relative dispersion <= training 95th percentile'},
      'label_support':{'harmful_examples':int(D.harm.sum()),'harmful_groups':int(D.loc[D.harm==1,'network_seed'].nunique()),'no_headroom_examples':int(D.no_headroom.sum()),'no_headroom_groups':int(D.loc[D.no_headroom==1,'network_seed'].nunique())},
      'unabstained_k32':metrics(full),
      'exploratory_fixed_l08_q95':{**metrics(fixed),'canonical_bootstrap_ci95':bootstrap_canonical(fixed),'details':fixeddetail},
      'fully_nested_fold_family':{**metrics(nested),'canonical_bootstrap_ci95':bootstrap_canonical(nested),'selection_counts':dict(Counter(x['feature'] for x in nesteddetail)),'details':nesteddetail},
      'ridge_log_ratio_suite':{'best':ridge[0],'all':ridge},
      'interpretation':{'fixed_rule_posthoc':True,'promotion_evidence':False,'continuous_alpha_identifiable_from_archived_tables':False}
    }
    (ROOT/'results'/'PATH2_RESULTS.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('ridge_log_ratio_suite',)},indent=2))
    print('best ridge',ridge[0])
if __name__=='__main__':main()
