import glob,json
from pathlib import Path
import numpy as np,pandas as pd
P=Path('/mnt/data/work/new_opportunities/aligned');R=[json.loads(Path(p).read_text()) for p in sorted(P.glob('screen_network_*.json'))];K=sorted(set.intersection(*(set(r['configs']) for r in R)));rng=np.random.default_rng(5);out=[]
for mode in ('full','cf4'):
 for k in K:
  b=np.array([r['base_unbiased_mse'] for r in R]);m=np.array([r['configs'][k][mode]['unbiased_mse'] for r in R]);inn=np.array([r['configs'][k][mode]['inner'] for r in R]);nn=np.array([r['configs'][k][mode]['norm_sq'] for r in R]);
  grid=np.arange(0,2.01,.25);loss=[np.sum(b+2*a*inn+a*a*nn) for a in grid];a=float(grid[np.argmin(loss)]);ms=b+2*a*inn+a*a*nn
  ix=rng.integers(0,len(R),(50000,len(R)));g=b[ix].sum(1)/ms[ix].sum(1)
  out.append({'config':k,'mode':mode,'gain':float(b.sum()/m.sum()),'wins':int((m<b).sum()),'worst':float(np.max(m/b)),'alpha':a,'shrink_gain':float(b.sum()/ms.sum()),'shrink_wins':int((ms<b).sum()),'shrink_worst':float(np.max(ms/b)),'ci':[float(np.quantile(g,.025)),float(np.quantile(g,.975))]})
d=pd.DataFrame(out).sort_values('shrink_gain',ascending=False);d.to_csv(P/'ALIGNED_AGGREGATE.csv',index=False);(P/'ALIGNED_AGGREGATE.json').write_text(json.dumps(d.to_dict('records'),indent=2));print(d.head(30).to_string(index=False))
