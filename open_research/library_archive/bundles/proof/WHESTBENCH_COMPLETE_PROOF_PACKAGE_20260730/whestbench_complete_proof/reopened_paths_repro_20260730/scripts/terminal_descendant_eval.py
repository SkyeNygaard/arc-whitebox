import sys,json
import numpy as np
from pathlib import Path
sys.path.insert(0,'/mnt/data/whest_reopened')
import reopened_path_experiments as r
OUT=r.OUT
pre=json.load(open(OUT/'DESCENDANT_PREREGISTRATION.json'))
seeds=pre['terminal_synthetic_holdout'];arrs={s:dict(np.load(OUT/f'network_{s}.npz')) for s in seeds}
res={'preregistration_sha256':'a760d61708c917cbc2d0346168bed3e034be232744f9d4bbd7bacbb8a0156190','seeds':seeds,'decision_rule':pre['decision_rule'],'candidates':{}}
for c in pre['candidates']:
    key=c['stored_prediction_key'];alpha=c['global_alpha']
    preds=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in seeds]
    m=r.metric_summary(preds,arrs,seeds)
    passed=(m['pooled_raw_ratio']<.95 and m['pooled_cross_ratio']<.95 and m['wins_raw']>=10 and m['worst_raw_ratio']<1.25)
    res['candidates'][c['name']]={'alpha':alpha,'metrics':m,'passed':passed}
(OUT/'TERMINAL_DESCENDANT_RESULTS.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
