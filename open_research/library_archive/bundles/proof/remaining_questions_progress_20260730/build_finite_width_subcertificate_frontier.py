#!/usr/bin/env python3
import json
from decimal import Decimal,getcontext
from pathlib import Path
getcontext().prec=100
ROOT=Path(__file__).resolve().parent
cert=json.load(open(ROOT/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json'))
kup=Decimal(cert['certified_result']['kerdock_mse_upper_bound']);rows=cert['components']
front=[]
for M in range(2,321,2):
 sel=[r for r in rows if 2*(int(r['s'])+1)<=M]
 obj=sum((Decimal(r['y']) for r in sel),Decimal(0));frac=obj/kup
 front.append({'maximum_kernel_harmonic_degree':M,'component_count':len(sel),'mse_lower_bound':str(obj),'fraction_of_infinite_width_kerdock_upper':str(frac)})
thresholds={}
for t in ['0.50','0.60','0.70','0.80','0.85','0.90','0.92','0.93','0.935','0.937']:
 td=Decimal(t); hit=next((x for x in front if Decimal(x['fraction_of_infinite_width_kerdock_upper'])>=td),None);thresholds[t]=hit
# Conditional finite-width ratios for example alpha,beta.
scenarios=[]
for M in [26,40,60,80,100,128,140,164,194,214,240,280]:
 x=next(z for z in front if z['maximum_kernel_harmonic_degree']>=M)
 f=Decimal(x['fraction_of_infinite_width_kerdock_upper'])
 for alpha,beta in [('1.00','1.12'),('0.95','1.12'),('0.90','1.12'),('0.95','1.05')]:
  scenarios.append({'cutoff':x['maximum_kernel_harmonic_degree'],'subcertificate_fraction':str(f),'alpha':alpha,'beta':beta,'finite_width_ratio_floor':str(f*Decimal(alpha)/Decimal(beta))})
out={'title':'Exact subcertificate frontier for finite-width transfer','definition':'At cutoff M, retain only released comparison components whose squared feature kernel has maximum harmonic degree <= M. This is an exact valid subcertificate without reoptimization.',
 'frontier':front,'thresholds':thresholds,'illustrative_transfer_scenarios':scenarios,
 'key_conclusion':'A 90% infinite-width signed floor already has a valid subcertificate using kernel coefficients only through degree 128; 93% requires degree 194. Full 93.7046% uses degree 280.'}
(ROOT/'FINITE_WIDTH_SUBCERTIFICATE_FRONTIER.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'thresholds':thresholds,'key_conclusion':out['key_conclusion']},indent=2))
