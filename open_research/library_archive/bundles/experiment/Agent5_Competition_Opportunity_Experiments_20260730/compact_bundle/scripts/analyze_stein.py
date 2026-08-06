import glob,json,math
import numpy as np,pandas as pd
from pathlib import Path
files=sorted(glob.glob('/mnt/data/work/new_opportunities/results/stein_network_71*.json'))
rows=[]
for f in files:
 d=json.load(open(f));
 for name,c in d['candidates'].items():
  rows.append({'network_id':d['network_id'],'candidate':name,'base_obs':d['base_observed_mse'],'base_unb':d['base_unbiased_mse'],'noise':d['reference_noise_mse'],**c})
df=pd.DataFrame(rows)

def boot_ratio(sub,ckey,bkey='base_unb',B=20000):
 rng=np.random.default_rng(20260730); a=sub[bkey].to_numpy(); c=sub[ckey].to_numpy(); n=len(a)
 vals=np.empty(B)
 for i in range(B):
  ix=rng.integers(0,n,n); vals[i]=a[ix].sum()/c[ix].sum()
 return np.quantile(vals,[.025,.975]).tolist()
out=[]
for name,g in df.groupby('candidate'):
 for mode,key in [('full','full_unbiased_mse'),('cf4','cf4_unbiased_mse'),('oracle','oracle_scalar_unbiased_mse')]:
  b=g.base_unb.sum(); c=g[key].sum(); gain=b/c
  ratios=g[key]/g.base_unb
  out.append({'candidate':name,'mode':mode,'gain_base_over_candidate':gain,'ci95':boot_ratio(g,key),'wins':int((g[key]<g.base_unb).sum()),'median_candidate_over_base':float(np.median(ratios)),'worst_candidate_over_base':float(np.max(ratios)),'mean_cosine':float(g.correction_cosine.mean()),'median_negative_mass':float(g.negative_mass.median()),'max_negative_mass':float(g.negative_mass.max()),'features':int(g.n_features.iloc[0])})
res=pd.DataFrame(out).sort_values(['mode','gain_base_over_candidate'],ascending=[True,False])
print(res.to_string(index=False))
Path('/mnt/data/work/new_opportunities/results/STEIN_SCREEN_SUMMARY.json').write_text(json.dumps({'n_networks':len(files),'rows':out},indent=2))
res.to_csv('/mnt/data/work/new_opportunities/results/STEIN_SCREEN_SUMMARY.csv',index=False)
