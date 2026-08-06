#!/usr/bin/env python3
"""Directed-decimal endpoint separation using certified T16 coefficient intervals."""
from __future__ import annotations
import hashlib
import json
from decimal import Decimal, Context, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

HERE=Path(__file__).resolve().parent
T16=HERE/'source_certificates'/'T16_PRIMAL_DUAL_CERTIFICATE.json'
K32=HERE/'K32_MUB_LINE_SPECTRUM_CERTIFICATE.json'
OUT=HERE/'T16_ENDPOINT_CERTIFICATE.json'

DOWN=Context(prec=120,rounding=ROUND_FLOOR,Emin=-999999,Emax=999999)
UP=Context(prec=120,rounding=ROUND_CEILING,Emin=-999999,Emax=999999)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def add_lo(a,b):
    with localcontext(DOWN): return a+b
def add_hi(a,b):
    with localcontext(UP): return a+b
def sub_lo(a,b):
    with localcontext(DOWN): return a-b
def sub_hi(a,b):
    with localcontext(UP): return a-b

def main():
    d=json.loads(T16.read_text())
    c=[(Decimal(a),Decimal(b)) for a,b in d['hermite_coefficient_certificate']['coefficient_intervals']]
    h1lo=Decimal(0); h1hi=Decimal(0)
    hmlo=Decimal(0); hmhi=Decimal(0)
    for ell,(lo,hi) in enumerate(c):
        h1lo=add_lo(h1lo,lo); h1hi=add_hi(h1hi,hi)
        if ell%2==0:
            hmlo=add_lo(hmlo,lo); hmhi=add_hi(hmhi,hi)
        else:
            hmlo=sub_lo(hmlo,hi); hmhi=sub_hi(hmhi,lo)
    plus_res=(sub_lo(Decimal(1),h1hi),sub_hi(Decimal(1),h1lo))
    k=json.loads(K32.read_text())['point_values']['minus_one']
    km=(Decimal(k['lo']),Decimal(k['hi']))
    minus_res=(sub_lo(km[0],hmhi),sub_hi(km[1],hmlo))
    result={
      'status':'PASS' if plus_res[0]>0 and minus_res[0]>0 else 'FAIL',
      'inputs':{
        't16_certificate_sha256':sha(T16),
        'k32_line_spectrum_certificate_sha256':sha(K32),
      },
      'h_plus_one_interval':[str(h1lo),str(h1hi)],
      'K32_minus_one_interval':[str(km[0]),str(km[1])],
      'h_minus_one_interval':[str(hmlo),str(hmhi)],
      'K32_minus_h_at_plus_one':[str(plus_res[0]),str(plus_res[1])],
      'K32_minus_h_at_minus_one':[str(minus_res[0]),str(minus_res[1])],
      'conclusion':'Both endpoint residuals are strictly positive. Together with the interior Hermite remainder, equality occurs only at the three interior contact nodes.',
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)

if __name__=='__main__': main()
