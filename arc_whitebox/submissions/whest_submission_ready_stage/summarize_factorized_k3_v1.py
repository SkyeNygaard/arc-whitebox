#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args()
r=json.loads(a.result.read_text())
print('split:',r['split'])
print('upstream factorized-K3 MSE:',r['upstream']['summary']['final_mean_mse'])
best=None
for alpha,entry in r['hybrid'].items():
 s=entry['summary']
 print(f"alpha={alpha}: gain={s['gain_vs_upstream']:.3f}x, improved={100*s['fraction_mlps_improved']:.1f}%, MSE={s['final_mean_mse']:.6g}")
 if best is None or s['final_mean_mse']<best[1]['final_mean_mse']:best=(alpha,s)
print('best:',best[0],best[1])
if best[1]['gain_vs_upstream']>=1.25 and best[1]['fraction_mlps_improved']>=0.75:
 print('VERDICT: CONTINUE TO FULL TEST / FLOPS')
elif best[1]['gain_vs_upstream']>=1.05:
 print('VERDICT: BORDERLINE')
else:
 print('VERDICT: STOP — predicted K3 features do not preserve the oracle gain')
