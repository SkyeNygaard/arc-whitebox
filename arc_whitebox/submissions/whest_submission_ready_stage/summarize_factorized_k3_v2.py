#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args();r=json.loads(a.result.read_text())
print('split:',r['split']);print('upstream MSE:',r['upstream']['summary']['final_mean_mse'])
best=None
for k,e in r['hybrid'].items():
 s=e['summary'];print(f"{k}: gain={s['gain_vs_upstream']:.3f}x win={100*s['fraction_mlps_improved']:.1f}% median={s['median_mlp_gain']:.3f} worst={s['worst_mlp_gain']:.3f} guard={100*s['fraction_guard_activated']:.1f}% MSE={s['final_mean_mse']:.6g}")
 if best is None or s['final_mean_mse']<best[1]['final_mean_mse']:best=(k,s)
print('best:',best[0],json.dumps(best[1],indent=2))
if best[1]['gain_vs_upstream']>=1.25 and best[1]['fraction_mlps_improved']>=.75:print('VERDICT: SUBMISSION PORT IS JUSTIFIED')
elif best[1]['gain_vs_upstream']>=1.05:print('VERDICT: BORDERLINE — optimize/calibrate')
else:print('VERDICT: STOP THIS BRANCH')
