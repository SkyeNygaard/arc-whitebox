import glob,re,json,sys,numpy as np
sys.path.insert(0,'/mnt/data/competition_relevance_20260730');from nonlinear_basis_aggregation import methods
ROOT='/mnt/data/competition_relevance_20260730/nonlinear_basis_aggregation/basis_cases'
B={}
for f in glob.glob(ROOT+'/*.npz'):
 m=re.search(r'n(\d+)_r(\d+)',f);B[(int(m.group(1)),int(m.group(2)))]=np.load(f)['B']
a=.7773602544607717;rows=[]
for n in range(7408,7412):
 means={r:B[n,r].mean(0) for r in [3,11,97]}
 for r in [3,11,97]:
  ref=np.mean([means[q] for q in [3,11,97] if q!=r],0);base,mm=methods(B[n,r]);cand=base+a*(mm['geomed']-base);bm=np.mean((base-ref)**2);cm=np.mean((cand-ref)**2);rows.append({'network':n,'rot':r,'ratio':float(cm/bm),'base_mse':float(bm),'candidate_mse':float(cm)})
r=np.array([x['ratio'] for x in rows]);out={'frozen_alpha':a,'networks':[7408,7409,7410,7411],'n':len(rows),'mean_ratio':float(r.mean()),'median_ratio':float(np.median(r)),'wins':int(np.sum(r<1)),'worst':float(r.max()),'rows':rows}
open('/mnt/data/competition_relevance_20260730/nonlinear_basis_aggregation/HOLDOUT_GEOMED.json','w').write(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
