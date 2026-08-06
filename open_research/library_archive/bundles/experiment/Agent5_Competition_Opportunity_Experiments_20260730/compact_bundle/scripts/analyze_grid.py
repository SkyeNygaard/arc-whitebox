#!/usr/bin/env python3
import glob,json,math
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/mnt/data/work/new_opportunities/grid'); files=sorted(P.glob('grid_network_*.json')); rows=[json.loads(p.read_text()) for p in files]
keys=sorted(set.intersection(*(set(r['configs']) for r in rows)))
rng=np.random.default_rng(20260730)
out=[]
for mode in ('cf4','full'):
 for k in keys:
  base=np.array([r['base_unbiased_mse'] for r in rows]); m=np.array([r['configs'][k][mode]['unbiased_mse'] for r in rows]); inn=np.array([r['configs'][k][mode]['inner'] for r in rows]); nn=np.array([r['configs'][k][mode]['norm_sq'] for r in rows])
  alpha=float(np.clip(-inn.sum()/max(nn.sum(),1e-30),-2,2)); ms=base+2*alpha*inn+alpha*alpha*nn
  # leave-one-network-out global scalar shrink
  al=[]; ml=[]
  for i in range(len(rows)):
   tr=np.arange(len(rows))!=i; a=float(np.clip(-inn[tr].sum()/max(nn[tr].sum(),1e-30),-2,2));al.append(a);ml.append(base[i]+2*a*inn[i]+a*a*nn[i])
  ml=np.array(ml)
  def boot_gain(vals,B=10000):
   ix=rng.integers(0,len(vals),(B,len(vals))); g=base[ix].sum(1)/vals[ix].sum(1);return [float(np.quantile(g,.025)),float(np.quantile(g,.975))]
  out.append({'config':k,'mode':mode,'unshrunk_gain':float(base.sum()/m.sum()),'unshrunk_wins':int((m<base).sum()),'unshrunk_worst':float(np.max(m/base)),
              'global_alpha':alpha,'global_shrink_gain':float(base.sum()/ms.sum()),'global_shrink_ci':boot_gain(ms),'global_shrink_wins':int((ms<base).sum()),'global_shrink_worst':float(np.max(ms/base)),
              'lono_alpha_mean':float(np.mean(al)),'lono_shrink_gain':float(base.sum()/ml.sum()),'lono_shrink_ci':boot_gain(ml),'lono_shrink_wins':int((ml<base).sum()),'lono_shrink_worst':float(np.max(ml/base))})
df=pd.DataFrame(out).sort_values('lono_shrink_gain',ascending=False)
df.to_csv(P/'GRID_AGGREGATE.csv',index=False);(P/'GRID_AGGREGATE.json').write_text(json.dumps({'networks':[r['network_id'] for r in rows],'rows':df.to_dict('records')},indent=2))
print(df.head(30).to_string(index=False))
