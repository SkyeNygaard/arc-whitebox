#!/usr/bin/env python3
import json,math
from decimal import Decimal,getcontext
from pathlib import Path
getcontext().prec=80
D=256;N=66048;ROOT=Path(__file__).resolve().parent

def hdim(l):
 if l==0:return 1
 if l==1:return D
 return math.comb(D+l-1,l)-math.comb(D+l-3,l-2)
cert=json.load(open(ROOT/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json'))
rows=[];minmargin=Decimal(1)
for x in cert['components']:
 s=int(x['s']);r=Decimal(x['r']);ds=Decimal(hdim(s));dt=Decimal(hdim(s+1));T=ds+r*dt
 qs=Decimal(N)/T;qt=Decimal(N)*r/T
 assert 0<=qs<=1 and 0<=qt<=1
 # The desired diagonal has ds copies qs and dt copies qt, sums to N.
 assert abs(ds*qs+dt*qt-Decimal(N))<Decimal('1e-60')
 margin=min(Decimal(1)-qs,Decimal(1)-qt)
 minmargin=min(minmargin,margin)
 bound=T*T/Decimal(N)-ds-r*r*dt
 assert bound>0
 rows.append({'s':s,'r':str(r),'q_s':str(qs),'q_splus1':str(qt),'rank_floor':str(bound)})
out={'verified':True,'component_count':len(rows),'minimum_diagonal_room_below_one':str(minmargin),
 'schur_horn_condition':'Each desired diagonal lies in [0,1] and sums to integer N, hence is majorized by (1,...,1,0,...,0) with N ones.',
 'conclusion':'A rank-N orthogonal projection with these block-diagonal traces exists. Scaling it by T/N attains equality in the abstract block-trace/rank relaxation.',
 'rows':rows}
(ROOT/'BLOCKTRACE_RELAXATION_SHARPNESS_CHECK.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
