from __future__ import annotations
import json,re,warnings
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'legal_features_and_labels.csv')
CANP=ROOT/'canonical24_quadratics_audited';HARDP=ROOT/'hardpanel_quadratics_audited'
terms=[]
for _,z in D.iterrows():
 panel=CANP if z.domain=='canonical' else HARDP
 v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz')
 b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;d=p-b
 a0=float(np.mean(e*e));lin=float(2*np.mean(e*d));q=float(np.mean(d*d));ao=float(-lin/(2*q))
 terms.append({'raw_a0':a0,'raw_lin':lin,'raw_q':q,'alpha_opt':ao})
D=pd.concat([D,pd.DataFrame(terms)],axis=1)
G=D.network_seed.to_numpy();CAN=D.domain.eq('canonical').to_numpy();HARD=D.in_hard_panel.astype(bool).to_numpy()
META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds','raw_a0','raw_lin','raw_q','alpha_opt'}
NUM=[c for c in D if c not in META and np.issubdtype(D[c].dtype,np.number)]
sets={
 'compact':[c for c in NUM if re.match(r'l(08|16|24|28|29|30|31)_fold_rel_(mean|q50|q90|max)$',c) or c in ['anchor_effrank','anchor_frob','anchor_r90','anchor_rho','anchor_trace'] or re.match(r'w_suffix_(fro|op|std|trace)_(mean|q50|q90|max)$',c)],
 'rotation':[c for c in NUM if ('fold_rel_' in c or 'block_disp_' in c or 'antipodal_imb_' in c)],
 'late':[c for c in NUM if re.match(r'l(20|24|28|29|30|31)_',c) or c.startswith('anchor_')],
 'weights':[c for c in NUM if c.startswith('w')],
 'all':NUM,
}
configs=[{'kind':'constant'}]
for sn,cs in sets.items():
 for k in [4,8,16]:
  if len(cs)<k:continue
  for reg in [1.,10.,100.,1000.]: configs.append({'kind':'ridge','set':sn,'k':k,'reg':reg})

def opt_constant(ix):
 lin=D.raw_lin.to_numpy()[ix].sum();q=D.raw_q.to_numpy()[ix].sum();return float(np.clip(-lin/(2*max(q,1e-30)),0,1.5))
def make_model(cfg):
 cs=sets[cfg['set']]
 return cs,Pipeline([('imp',SimpleImputer()),('scale',StandardScaler()),('select',SelectKBest(f_regression,k=cfg['k'])),('ridge',Ridge(alpha=cfg['reg']))])
def fit_predict(cfg,tr,te):
 if cfg['kind']=='constant':return np.full(len(te),opt_constant(tr))
 cs,m=make_model(cfg);X=D[cs].to_numpy();w=D.raw_q.to_numpy()[tr]/np.mean(D.raw_q.to_numpy()[tr])
 m.fit(X[tr],np.clip(D.alpha_opt.to_numpy()[tr],0,1.5),ridge__sample_weight=w)
 return np.clip(np.ravel(m.predict(X[te])),0,1.5)
def score(pred,ix):
 a0=D.raw_a0.to_numpy()[ix];lin=D.raw_lin.to_numpy()[ix];q=D.raw_q.to_numpy()[ix];m=a0+pred*lin+pred*pred*q;rat=m/a0
 can=CAN[ix];hard=HARD[ix]
 cp=float(m[can].sum()/a0[can].sum()) if np.any(can) else float(m.sum()/a0.sum())
 hw=float(rat[hard].max()) if np.any(hard) else 1.;hp=float(np.quantile(rat[hard],.9)) if np.any(hard) else 1.
 return cp+.75*max(0,hw-1.05)+.2*max(0,hp-1.0)+.05*np.mean((pred-np.clip(D.alpha_opt.to_numpy()[ix],0,1.5))**2)

def nested():
 logo=LeaveOneGroupOut();out=np.zeros(len(D));choices=[]
 for otr,ote in logo.split(D,G,G):
  ug=np.unique(G[otr]);inner=GroupKFold(n_splits=min(5,len(ug)));best=None
  for cfg in configs:
   ip=np.zeros(len(otr));subg=G[otr]
   for itr,ite in inner.split(otr,groups=subg):
    ip[ite]=fit_predict(cfg,otr[itr],otr[ite])
   sc=score(ip,otr);key=(sc,str(cfg))
   if best is None or key<best[0]:best=(key,cfg)
  cfg=best[1];out[ote]=fit_predict(cfg,otr,ote);choices.append({'held_seed':int(G[ote][0]),'config':cfg,'predictions':out[ote].tolist(),'inner_score':float(best[0][0])})
 return out,choices

def metrics(a):
 a=np.asarray(a);b=D.raw_a0.to_numpy();m=b+a*D.raw_lin.to_numpy()+a*a*D.raw_q.to_numpy();rat=m/b
 def sub(mask):return {'pooled_ratio':float(m[mask].sum()/b[mask].sum()),'mean_ratio':float(rat[mask].mean()),'worst':float(rat[mask].max()),'p90':float(np.quantile(rat[mask],.9)),'wins':int(np.sum(rat[mask]<1)),'coverage_nonzero':float(np.mean(a[mask]!=0))}
 return {'canonical':sub(CAN),'hard':sub(HARD),'all':sub(np.ones(len(D),bool)),'alpha_min':float(a.min()),'alpha_mean':float(a.mean()),'alpha_max':float(a.max()),'alphas':a.tolist(),'ratios':rat.tolist()}

pred,choices=nested();pol={'nested_model':metrics(pred),'fixed_0.4':metrics(np.full(len(D),.4)),'fixed_0.5':metrics(np.full(len(D),.5)),'l08_abstain':metrics(np.where(D.l08_fold_rel_mean<=0.00279066317598334,1.,0.)),'oracle_alpha':metrics(D.alpha_opt.to_numpy())}
out={'protocol':{'rows':len(D),'groups':int(D.network_seed.nunique()),'outer':'leave-one-base-network-out; rotations grouped','inner':'5-fold grouped configuration selection; final-MSE plus hard-tail objective','target':'exact direct-K32 raw-MSE optimal scale from four independent 262144-point streams'},'policies':pol,'selection_counts':dict(Counter(json.dumps(x['config'],sort_keys=True) for x in choices)),'choices':choices,'rows':D[['network_seed','rotation_seed','domain','in_hard_panel','alpha_opt','l08_fold_rel_mean']].assign(pred_alpha=pred).to_dict('records')}
(ROOT/'results'/'SCALE_MODEL_RESULTS.json').write_text(json.dumps(out,indent=2))
for k,v in pol.items():print(k,v['canonical'],v['hard'],'alpha',v['alpha_min'],v['alpha_mean'],v['alpha_max'])
print('choices',out['selection_counts'])
