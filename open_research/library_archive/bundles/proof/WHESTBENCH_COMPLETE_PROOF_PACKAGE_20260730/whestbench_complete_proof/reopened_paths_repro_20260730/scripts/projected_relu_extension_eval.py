import sys,json,math,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/whest_reopened')
import reopened_path_experiments as r
OUT=r.OUT
seeds=list(range(3020,3068));key='nonlinear_k4_m64_b0_0.001';alpha=-2.0
arrs={s:dict(np.load(OUT/f'network_{s}.npz')) for s in seeds}
br=[];cr=[];bc=[];cc=[];rat=[]
for s in seeds:
 a=arrs[s];p=a['base']+alpha*(a[key]-a['base'])
 braw=r.raw_mse(a['base'],a['ref']);craw=r.raw_mse(p,a['ref'])
 bx=r.cross_mse(a['base'],a['ref_a'],a['ref_b']);cx=r.cross_mse(p,a['ref_a'],a['ref_b'])
 br.append(braw);cr.append(craw);bc.append(bx);cc.append(cx);rat.append(craw/braw)
br=np.array(br);cr=np.array(cr);bc=np.array(bc);cc=np.array(cc);rat=np.array(rat)
rng=np.random.default_rng(2026073002);B=200000
ix=rng.integers(0,len(seeds),size=(B,len(seeds)),endpoint=False)
boot_raw=cr[ix].sum(1)/br[ix].sum(1)
# cross denominator can be nonpositive in rare resamples; retain positive aggregate draws only.
bden=bc[ix].sum(1);boot_cross=cc[ix].sum(1)/bden;boot_cross=boot_cross[np.isfinite(boot_cross)&(bden>0)]
# difficulty strata fixed by baseline raw quartiles.
qs=np.quantile(br,[.25,.5,.75]);strata=[]
lo=-np.inf
for i,hi in enumerate(list(qs)+[np.inf]):
 m=(br>lo)&(br<=hi);strata.append({'stratum':i+1,'n':int(m.sum()),'baseline_range':[float(lo),float(hi)],'pooled_ratio':float(cr[m].sum()/br[m].sum()),'mean_ratio':float(rat[m].mean()),'wins':int(np.sum(rat[m]<1))});lo=hi
# Optimized analytical FLOP audit (multiply-add counted as 2 FLOPs, matching common dense accounting).
D=r.D;N=r.N;n_grad=768;depth=32;K=64;k=4
flops={}
flops['gradient_forward']=2*n_grad*D*D*depth
flops['gradient_backward']=2*n_grad*D*D*depth # 31 hidden backward + input gradient = 32 products
flops['project_input']=2*N*D*k
flops['feature_map']=2*N*k*K
flops['fold_sufficient_HtH']=2*N*K*K
flops['fold_sufficient_HtF']=2*N*K*D
flops['small_solves_upper']=6*(2/3*K**3+2*K*K*D)
flops['weight_svd_upper']=4*D**3
incremental=sum(flops.values())
baseline_B=175.62
ratio_compute=(baseline_B+incremental/1e9)/baseline_B
pooled=float(cr.sum()/br.sum());cross=float(cc.sum()/bc.sum())
result={'candidate':key,'alpha':alpha,'n':len(seeds),'pooled_raw_ratio':pooled,'pooled_cross_ratio':cross,'mean_raw_ratio':float(rat.mean()),'wins':int(np.sum(rat<1)),'p90':float(np.quantile(rat,.9)),'worst':float(rat.max()),
'bootstrap_raw_95':[float(np.quantile(boot_raw,.025)),float(np.quantile(boot_raw,.975))],
'bootstrap_raw_probability_below_1':float(np.mean(boot_raw<1)),'bootstrap_raw_probability_below_0.95':float(np.mean(boot_raw<.95)),
'bootstrap_cross_95':[float(np.quantile(boot_cross,.025)),float(np.quantile(boot_cross,.975))],
'difficulty_strata':strata,'per_seed_ratio':{str(s):float(x) for s,x in zip(seeds,rat)},
'flop_audit':{'components':flops,'incremental_B':incremental/1e9,'baseline_effective_B':baseline_B,'compute_ratio':ratio_compute,'projected_adjusted_ratio':pooled*ratio_compute,
'notes':'Optimized implementation accumulates per-fold sufficient statistics once; excludes dispatch/memory overhead and therefore is favorable.'},
'gate_passed':bool(pooled<.95 and cross<.95 and np.quantile(boot_raw,.975)<1 and pooled*ratio_compute<1)}
(OUT/'PROJECTED_RELU_EXTENSION_RESULTS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
