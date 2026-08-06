#!/usr/bin/env python3
import argparse,json,glob
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);args=ap.parse_args();P=Path('/mnt/data/work/new_opportunities/selected')
rows=[json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(P/f'{args.stage}_network_*.json')))]
rng=np.random.default_rng(20260730);out=[]
for mode in ('cf4','full'):
 for k in rows[0]['candidates']:
  b=np.array([r['base_unbiased_mse'] for r in rows]);m=np.array([r['candidates'][k][mode]['unbiased_mse'] for r in rows]);inn=np.array([r['candidates'][k][mode]['inner'] for r in rows]);nn=np.array([r['candidates'][k][mode]['norm_sq'] for r in rows])
  # candidate alphas 0.25..2 selected on aggregate screen only
  grid=np.arange(0,2.01,.25);loss=np.array([(b+2*a*inn+a*a*nn).sum() for a in grid]);a=float(grid[loss.argmin()]);ms=b+2*a*inn+a*a*nn
  # exact alpha for diagnostic
  ao=float(np.clip(-inn.sum()/max(nn.sum(),1e-30),0,2));mo=b+2*ao*inn+ao*ao*nn
  def ci(x):
   ix=rng.integers(0,len(x),(20000,len(x)));g=b[ix].sum(1)/x[ix].sum(1);return [float(np.quantile(g,.025)),float(np.quantile(g,.975))]
  out.append({'candidate':k,'mode':mode,'networks':len(rows),'unshrunk_gain':float(b.sum()/m.sum()),'unshrunk_ci':ci(m),'unshrunk_wins':int((m<b).sum()),'unshrunk_worst':float(np.max(m/b)),
   'frozen_grid_alpha':a,'grid_shrink_gain':float(b.sum()/ms.sum()),'grid_shrink_ci':ci(ms),'grid_shrink_wins':int((ms<b).sum()),'grid_shrink_worst':float(np.max(ms/b)),
   'diagnostic_opt_alpha':ao,'diagnostic_gain':float(b.sum()/mo.sum()),'diagnostic_worst':float(np.max(mo/b))})
d=pd.DataFrame(out).sort_values('grid_shrink_gain',ascending=False);d.to_csv(P/f'{args.stage}_AGGREGATE.csv',index=False);(P/f'{args.stage}_AGGREGATE.json').write_text(json.dumps({'stage':args.stage,'network_ids':[r['network_id'] for r in rows],'rows':d.to_dict('records')},indent=2))
print(d.to_string(index=False))
