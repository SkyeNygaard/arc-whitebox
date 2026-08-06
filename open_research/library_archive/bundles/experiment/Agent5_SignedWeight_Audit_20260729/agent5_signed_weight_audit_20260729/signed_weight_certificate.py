#!/usr/bin/env python3
"""Directed-interval certificate for M=sup_{[-1,1]}(K_32-h).

The interval primitives are vendored verbatim from the verified v5 T22 package.
The proof is deliberately simpler than a full stationary-point search:
  * exact Bernstein coefficients prove h is increasing on [-1,1];
  * monotonicity of K bounds q on [-1, 37/50];
  * convexity of K and an interval derivative bound prove q is increasing on
    [37/50,1].
Therefore the global maximum is q(1)=1-h(1), an exact rational.
"""
from __future__ import annotations
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor_proof'))
from formal_proof_common import Problem  # type: ignore
from formal_interval_certificate import ( # type: ignore
    bernstein_on_minus1_1,
    polynomial_bernstein_range_on_interval,
    polyval_fraction,
    I,
)

CUT=Fraction(37,50)
PREC=90
p=Problem(ROOT/'vendor_proof',prec=PREC,d=256,depth=32)

# Exact monotonicity certificate for h.
hp_bern=bernstein_on_minus1_1(p.hp)
assert min(hp_bern)>0

q1=Fraction(1)-polyval_fraction(p.h,Fraction(1))
q1_lo=p.dr.Dlo(q1); q1_hi=p.dr.Dhi(q1)

# Left region: K(t)<=K(CUT), h(t)>=h(-1).
Kcut,_=p.state(CUT)
hminus1=p.pval(p.hI,Fraction(-1))
left_q_interval=p.dr.sub(Kcut,hminus1)
left_q_upper=left_q_interval.hi
assert left_q_upper<q1_lo

# Right region: K' is increasing because K_0 is linear and kappa is increasing
# and convex; composition preserves convexity. Bound h' by exact Bernstein
# coefficients on [CUT,1], and K' below by K'(CUT).
hp_lo_frac,hp_hi_frac=polynomial_bernstein_range_on_interval(p.hp,CUT,Fraction(1))
_,Kp_cut=p.state(CUT)
hp_range=I(p.dr.Dlo(hp_lo_frac),p.dr.Dhi(hp_hi_frac))
qprime_interval=p.dr.sub(Kp_cut,hp_range)
qprime_lower=qprime_interval.lo
assert qprime_lower>0

result={
  'title':'Directed-interval certificate for M = sup(K_32-h)',
  'dimension':256,
  'depth':32,
  'cut':str(CUT),
  'q1_exact':str(q1),
  'M_exact':str(q1),
  'M_interval':{'lower':str(q1_lo),'upper':str(q1_hi)},
  'hprime_bernstein_min_exact':str(min(hp_bern)),
  'hprime_bernstein_min_decimal':str(p.dr.Dlo(min(hp_bern))),
  'left_region':{
    'interval':[-1,str(CUT)],
    'method':'K(t)<=K(cut), h(t)>=h(-1)',
    'K_cut_interval':{'lower':str(Kcut.lo),'upper':str(Kcut.hi)},
    'h_minus1_interval':{'lower':str(hminus1.lo),'upper':str(hminus1.hi)},
    'q_upper':str(left_q_upper),
    'strictly_below_M_by_at_least':str(p.dr.sub(I(q1_lo,q1_hi),left_q_interval).lo),
  },
  'right_region':{
    'interval':[str(CUT),1],
    'method':"q'=K'-h'>0; K' increasing by convexity",
    'hprime_upper_exact':str(hp_hi_frac),
    'hprime_upper_decimal':str(p.dr.Dhi(hp_hi_frac)),
    'Kprime_cut_interval':{'lower':str(Kp_cut.lo),'upper':str(Kp_cut.hi)},
    'qprime_lower':str(qprime_lower),
  },
  'conclusion':'M=q(1)=1-h(1) exactly',
  'passed':True,
}
(ROOT/'M_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
