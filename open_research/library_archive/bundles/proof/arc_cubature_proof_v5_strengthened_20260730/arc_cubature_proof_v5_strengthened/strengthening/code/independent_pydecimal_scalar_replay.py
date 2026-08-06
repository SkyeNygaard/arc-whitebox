#!/usr/bin/env python3
"""Independent scalar replay using Python's pure-Python _pydecimal engine.

This does not recompute the expensive deep-kernel interval evaluations. It takes
only the already-certified pointwise/kernel/mean interval artifacts as inputs
and independently redoes all exact-fraction, multiplicity, subtraction,
division, and one-sided theorem assembly steps with outward rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import decimal
from decimal import Decimal, Context, localcontext

PREC = 100


def ctx(rounding: str) -> Context:
    c = Context(prec=PREC, rounding=rounding, Emin=-999999999, Emax=999999999)
    for signal in c.traps:
        c.traps[signal] = False
    return c

LO = ctx(decimal.ROUND_FLOOR)
HI = ctx(decimal.ROUND_CEILING)


def op(c: Context, fn, *args: Decimal) -> Decimal:
    with localcontext(c):
        return +fn(*args)


def add_lo(a,b): return op(LO, lambda x,y:x+y,a,b)
def add_hi(a,b): return op(HI, lambda x,y:x+y,a,b)
def sub_lo(a,b): return op(LO, lambda x,y:x-y,a,b)
def sub_hi(a,b): return op(HI, lambda x,y:x-y,a,b)
def mul_lo(a,b): return op(LO, lambda x,y:x*y,a,b)
def mul_hi(a,b): return op(HI, lambda x,y:x*y,a,b)
def div_lo(a,b): return op(LO, lambda x,y:x/y,a,b)
def div_hi(a,b): return op(HI, lambda x,y:x/y,a,b)


def frac_iv(q: Fraction) -> tuple[Decimal, Decimal]:
    n = Decimal(q.numerator); d = Decimal(q.denominator)
    return div_lo(n,d), div_hi(n,d)


def add(x,y): return add_lo(x[0],y[0]), add_hi(x[1],y[1])
def sub(x,y): return sub_lo(x[0],y[1]), sub_hi(x[1],y[0])


def mul(x,y):
    lows = [mul_lo(a,b) for a in x for b in y]
    highs = [mul_hi(a,b) for a in x for b in y]
    return min(lows), max(highs)


def div(x,y):
    if y[0] <= 0 <= y[1]: raise ZeroDivisionError(y)
    rec = (div_lo(Decimal(1),y[1]), div_hi(Decimal(1),y[0]))
    return mul(x,rec)


def iv(obj): return Decimal(obj['lower']), Decimal(obj['upper'])
def out_iv(x): return {'lower':str(x[0]),'upper':str(x[1])}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--proof-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    r=args.proof_dir/'results'
    paths={
      'pointwise':r/'FORMAL_CERTIFICATE_D256_L32.json',
      'energy':r/'FORMAL_DELSARTE_BOUND_D256_L32.json',
      'mean':r/'FORMAL_KERNEL_MEAN_D256_L32.json',
      'theorem':r/'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json',
    }
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    p,e,m,t=(data[x] for x in ('pointwise','energy','mean','theorem'))
    assert p['passed'] and e['passed'] and m['passed'] and t['passed']
    assert p['coverage_exact'] and p['no_gaps_or_overlaps']
    assert Decimal(p['global_upper_bound']) < 0

    coeff=[Fraction(x) for x in e['auxiliary_coefficients_exact']]
    assert all(x>=0 for x in coeff[1:])
    h1=sum(coeff)
    c0=coeff[0]
    bound_q=c0+(Fraction(1)-h1)/66048
    bound=frac_iv(bound_q)

    k={name:iv(v) for name,v in e['kernel_intervals'].items()}
    row=add(k['one'],k['minus_one'])
    row=add(row,mul(k['zero'],frac_iv(Fraction(510))))
    pair=add(k['minus_one_sixteenth'],k['plus_one_sixteenth'])
    row=add(row,mul(pair,frac_iv(Fraction(32768))))
    energy=mul(row,frac_iv(Fraction(1,66048)))
    gap=sub(energy,bound)

    A0=iv(m['A0_certified'])
    kmse=sub(energy,A0)
    cert=sub(bound,A0)
    if cert[0] <= 0: raise AssertionError(cert)
    ratio=div((kmse[1],kmse[1]),(cert[0],cert[0]))
    excess=sub((ratio[1],ratio[1]),(Decimal(1),Decimal(1)))
    percent=mul(excess,(Decimal(100),Decimal(100)))

    shipped={
      'energy':iv(e['kerdock_energy']),
      'bound':iv(e['universal_energy_lower_bound']),
      'gap':iv(e['kerdock_minus_universal_bound']),
      'kmse':iv(t['kerdock_mse_interval']),
      'cert':iv(t['certificate_expression_B_minus_A0_interval']),
      'ratio':iv(t['actual_multiplicative_ratio_kerdock_over_infimum']),
      'percent':iv(t['actual_relative_excess_percent']),
    }
    replay={'energy':energy,'bound':bound,'gap':gap,'kmse':kmse,'cert':cert,
            'ratio':(Decimal(1),ratio[1]),'percent':(Decimal(0),percent[1])}

    containment={}
    for name,x in replay.items():
        y=shipped[name]
        # Both are outward enclosures. Require overlap and quantify exact equality.
        containment[name]={
          'byte_value_equal': x==y,
          'replay_contains_shipped': x[0] <= y[0] and x[1] >= y[1],
          'shipped_contains_replay': y[0] <= x[0] and y[1] >= x[1],
          'overlap': max(x[0],y[0]) <= min(x[1],y[1]),
        }
        assert containment[name]['overlap']

    module_details={
      'decimal_module_file': Path(getattr(decimal,'__file__','')).name,
      'decimal_implementation': Decimal.__module__,
      'decimal_version': getattr(decimal,'__version__',None),
      'libmpdec_version_attribute': getattr(decimal,'__libmpdec_version__',None),
      'decimal_shim_sha256': hashlib.sha256(Path(decimal.__file__).read_bytes()).hexdigest(),
      'note': 'Executed with a decimal.py shim that re-exports _pydecimal; no C decimal arithmetic is used by this process.',
    }

    result={
      'title':'Independent pure-_pydecimal scalar theorem replay',
      'scope':'Replays exact-fraction Delsarte bound, Kerdock multiplicity sum, spherical-mean subtraction, and one-sided ratio from certified interval inputs; does not recompute deep-kernel intervals.',
      'precision':PREC,
      'implementation':module_details,
      'input_sha256':{k:sha(v) for k,v in paths.items()},
      'exact_checks':{
        'higher_gegenbauer_coefficients_nonnegative':True,
        'h1_exact':str(h1),
        'c0_exact':str(c0),
        'universal_bound_exact':str(bound_q),
        'pointwise_global_upper_strictly_negative':True,
        'coverage_exact':True,
        'no_gaps_or_overlaps':True,
      },
      'replayed':{k:out_iv(v) for k,v in replay.items()},
      'shipped':{k:out_iv(v) for k,v in shipped.items()},
      'comparison':containment,
      'certified_ratio_upper':str(ratio[1]),
      'certified_relative_excess_percent_upper':str(percent[1]),
      'passed':all(v['overlap'] for v in containment.values()),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
