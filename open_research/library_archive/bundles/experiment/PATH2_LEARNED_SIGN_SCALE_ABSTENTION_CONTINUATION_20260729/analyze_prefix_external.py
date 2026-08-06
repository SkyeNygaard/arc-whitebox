from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'prefix_external_features.csv')
COUNTS=[8,16,24,32,48,64,80,96,112,129]
# Frozen after original model selection, before these external labels were inspected.
FROZEN_FULL_THRESHOLD=0.00279066317598334

def pooled(d,apply,ratio='candidate_ratio'):
    return float(np.sum(d.baseline_mse*np.where(apply,d[ratio],1.0))/np.sum(d.baseline_mse))
def met(d,apply,ratio='candidate_ratio'):
    pol=np.where(apply,d[ratio],1.)
    return {'n':len(d),'coverage':float(np.mean(apply)),'pooled_ratio':pooled(d,apply,ratio),
            'wins':int(np.sum(pol<1)),'worst':float(np.max(pol)),'p90':float(np.quantile(pol,.9))}

out={'frozen_full_threshold':FROZEN_FULL_THRESHOLD,'consistency':{},'prefix_panel':{},'external':{},'teacher':{}}
# consistency with original extractor
old=pd.read_csv(ROOT/'legal_features_and_labels.csv')[['network_seed','rotation_seed','l08_fold_rel_mean']]
p=D[D.domain.isin(['canonical','hard'])].merge(old,on=['network_seed','rotation_seed'])
diff=p.n129_risk_mean-p.l08_fold_rel_mean
out['consistency']={'max_abs':float(np.max(np.abs(diff))),'mean_abs':float(np.mean(np.abs(diff)))}
# Panel prefix counts: fixed quantile per LOGO, threshold trained on all other groups, exactly analogous to original screen.
panel=D[D.domain.isin(['canonical','hard'])].copy().reset_index(drop=True)
g=panel.network_seed.to_numpy(); can=panel.domain.eq('canonical').to_numpy(); hard=panel.domain.isin(['canonical','hard']).to_numpy() & panel.network_seed.isin([205215497,493891104,422494190,680708219]).to_numpy()
# hard membership: include canonical rotation for hard bases plus the 8 extra rotations.
for n in COUNTS:
    f=f'n{n:03d}_risk_mean'; vals=panel[f].to_numpy(); apply=np.ones(len(panel),bool)
    for seed in np.unique(g):
        tr=g!=seed; te=g==seed; apply[te]=vals[te]<=np.quantile(vals[tr],.95)
    # compute metrics with masks locally
    def mm(mask):
        d=panel[mask]; a=apply[mask]; return met(d,a)
    out['prefix_panel'][str(n)]={'canonical':mm(can),'hard':mm(hard),
      'spearman_log_ratio':float(spearmanr(vals,np.log(panel.candidate_ratio)).statistic)}
# External unchanged full gate, plus ranking diagnostics.
for domain in ['radial16','sparse24']:
    d=D[D.domain==domain].copy(); apply=d.n129_risk_mean.to_numpy()<=FROZEN_FULL_THRESHOLD
    obj={'full_gate':met(d,apply),'raw':met(d,np.ones(len(d),bool)),
         'risk_spearman_log_ratio':float(spearmanr(d.n129_risk_mean,np.log(d.candidate_ratio)).statistic),
         'risk_pearson_log_ratio':float(pearsonr(d.n129_risk_mean,np.log(d.candidate_ratio)).statistic),
         'abstained':[{'seed':int(s),'risk':float(r),'ratio':float(y)} for s,r,y in d.loc[~apply,['network_seed','n129_risk_mean','candidate_ratio']].to_numpy()]}
    if domain=='sparse24':
        obj['connected_full_gate']=met(d,apply,'connected_ratio'); obj['connected_raw']=met(d,np.ones(len(d),bool),'connected_ratio')
        obj['lower_full_gate']=met(d,apply,'lower_ratio'); obj['lower_raw']=met(d,np.ones(len(d),bool),'lower_ratio')
    out['external'][domain]=obj
# K128 teacher on panel: merge K128 labels and assess simple l08 relationship / rescue policy.
canon_path=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE/UNIFORM_INDEPENDENT_LAYER31_RESIDUAL_ROWS.csv')
hard_path=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE/UNIFORM_HARD_ROTATION_PANEL.csv')
c=pd.read_csv(canon_path); c=c[c.probe_count.isin([32,128])].pivot(index='network_seed',columns='probe_count',values='candidate_ratio').reset_index(); c['rotation_seed']=3
h=pd.read_csv(hard_path); h=h[h.probe_count.isin([32,128])].pivot(index=['network_seed','rotation_seed'],columns='probe_count',values='candidate_ratio').reset_index()
t=pd.concat([c,h[~h.set_index(['network_seed','rotation_seed']).index.isin(c.set_index(['network_seed','rotation_seed']).index)]],ignore_index=True)
t=t.rename(columns={32:'ratio32',128:'ratio128'}).merge(D[['network_seed','rotation_seed','domain','baseline_mse','n129_risk_mean']],on=['network_seed','rotation_seed'])
t['teacher_log_advantage']=np.log(t.ratio32/t.ratio128)
out['teacher']={'n':len(t),'k128_better':int(np.sum(t.ratio128<t.ratio32)),
 'risk_spearman_k128_advantage':float(spearmanr(t.n129_risk_mean,t.teacher_log_advantage).statistic),
 'risk_spearman_ratio32':float(spearmanr(t.n129_risk_mean,np.log(t.ratio32)).statistic),
 'risk_spearman_ratio128':float(spearmanr(t.n129_risk_mean,np.log(t.ratio128)).statistic)}
(ROOT/'results'/'PATH2_CONTINUATION_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
