#!/usr/bin/env python3
"""Derive the complete h-K sign diagram from certificate fields.

Unlike the original verifier, this script contains no hard-coded endpoint-sign
or max/min pattern. It infers all signs from certified intervals and exact box
ordering, then checks that curvature monotonicity propagates them across every
connecting region.
"""
from __future__ import annotations
import argparse,json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


def strict_sign(lo,hi):
    lo,hi=Decimal(str(lo)),Decimal(str(hi))
    if lo>0:return 'positive'
    if hi<0:return 'negative'
    raise AssertionError((lo,hi))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--certificate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cert=json.loads(a.certificate.read_text()); b=cert['base_certificate']; C=b['critical_boxes']; J=b['inflection_boxes']
    assert cert['coverage_exact'] and cert['no_gaps_or_overlaps'] and cert['passed']
    assert len(C)==len(J)+1

    special=[]
    for i in range(len(J)): special += [('critical',i,C[i]),('inflection',i,J[i])]
    special += [('critical',len(C)-1,C[-1])]
    for left,right in zip(special,special[1:]): assert Fraction(left[2]['right'])<Fraction(right[2]['left'])

    critical=[]
    for i,c in enumerate(C):
        ls=strict_sign(*c['left_gp']); rs=strict_sign(*c['right_gp'])
        if (ls,rs)==('positive','negative'):
            kind='maximum'; assert Decimal(c['gpp_upper'])<0
        elif (ls,rs)==('negative','positive'):
            kind='minimum'; assert Decimal(c['gpp_lower'])>0
        else: raise AssertionError((i,ls,rs))
        critical.append({'index':i,'left_gprime':ls,'right_gprime':rs,'derived_kind':kind,'g_upper':c['g_upper']})

    inflection=[]
    for i,j in enumerate(J):
        s=strict_sign(j['gp_lower'],j['gp_upper']); assert s==j['gp_sign']
        inflection.append({'index':i,'gprime_sign':s})

    rows=cert['curvature_intervals']; regions=[]
    # Exact geometric boundaries are the consecutive special-box gaps, plus C_last -> right-tail cutoff.
    gaps=[]
    for x,y in zip(special,special[1:]): gaps.append((Fraction(x[2]['right']),Fraction(y[2]['left'])))
    gaps.append((Fraction(C[-1]['right']),Fraction(b['right_tail']['left'])))
    for rid,(expected_left,expected_right) in enumerate(gaps):
        rr=sorted([r for r in rows if r['region']==rid],key=lambda r:Fraction(r['left']))
        assert rr and Fraction(rr[0]['left'])==expected_left and Fraction(rr[-1]['right'])==expected_right
        for x,y in zip(rr,rr[1:]): assert Fraction(x['right'])==Fraction(y['left'])
        signs={r['sign'] for r in rr}; assert len(signs)==1
        curvature=next(iter(signs))
        for r in rr:
            numeric=strict_sign(r['lower'],r['upper']); assert numeric==curvature

        if rid < len(gaps)-1:
            left_obj=special[rid][2]; right_obj=special[rid+1][2]
            left_sign = strict_sign(*left_obj['right_gp']) if special[rid][0]=='critical' else strict_sign(left_obj['gp_lower'],left_obj['gp_upper'])
            right_sign = strict_sign(*right_obj['left_gp']) if special[rid+1][0]=='critical' else strict_sign(right_obj['gp_lower'],right_obj['gp_upper'])
        else:
            left_sign=strict_sign(*C[-1]['right_gp'])
            # g'=h'-K'; strict upper/lower comparison establishes negativity.
            assert Decimal(b['right_tail']['hprime_upper']) < Decimal(b['right_tail']['Kprime_lower'])
            right_sign='negative'
        assert left_sign==right_sign
        # If g' is monotone under the certified curvature and has the same strict sign at both ends, it has that sign throughout.
        if curvature=='negative':
            assert left_sign=='negative' or right_sign=='positive'
        elif curvature=='positive':
            assert left_sign=='positive' or right_sign=='negative'
        else: raise AssertionError(curvature)
        regions.append({'region':rid,'left':str(expected_left),'right':str(expected_right),'subintervals':len(rr),'curvature':curvature,'derived_gprime_sign':left_sign})

    assert Decimal(b['left_tail']['gp_lower'])>0
    for e in b['endpoint_boxes']: assert Decimal(e['g_upper'])<0
    maxima=[x for x in critical if x['derived_kind']=='maximum']
    minima=[x for x in critical if x['derived_kind']=='minimum']
    assert all(Decimal(x['g_upper'])<0 for x in maxima)
    global_bound=max([Decimal(x['g_upper']) for x in maxima]+[Decimal(x['g_upper']) for x in b['endpoint_boxes']])
    assert global_bound<0 and global_bound==Decimal(b['global_candidate_upper_bound'])

    out={'title':'Derived sign-logic verification with no hard-coded sign pattern','special_box_order_verified':True,'critical_boxes':critical,'inflection_boxes':inflection,'connecting_regions':regions,'derived_maxima':[x['index'] for x in maxima],'derived_minima':[x['index'] for x in minima],'global_candidate_upper_bound':str(global_bound),'conclusion':'All possible interior maxima and both endpoints are strictly negative; monotonicity covers all intervening regions, so h-K<0 on [-1,1].','passed':True}
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
