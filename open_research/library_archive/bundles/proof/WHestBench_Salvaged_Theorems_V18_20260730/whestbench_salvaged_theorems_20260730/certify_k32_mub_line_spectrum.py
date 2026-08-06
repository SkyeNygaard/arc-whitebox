#!/usr/bin/env python3
"""Directed-rounding certificate for the K32 full-MUB-line Gram spectrum."""
from __future__ import annotations
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from formal_interval_certificate import Directed, I, pi_bounds, deep_kernel_fraction

OUT=HERE/'K32_MUB_LINE_SPECTRUM_CERTIFICATE.json'

def main():
    dr=Directed(prec=130)
    plo,phi=pi_bounds(190)
    pi=I(dr.Dlo(plo),dr.Dhi(phi))
    depth=32
    d=256
    M=129
    N=M*d
    points={
        'minus_one':Fraction(-1),
        'zero':Fraction(0),
        'plus_one_over_16':Fraction(1,16),
        'minus_one_over_16':Fraction(-1,16),
        'plus_one':Fraction(1),
    }
    vals={k:deep_kernel_fraction(x,depth,dr,pi) for k,x in points.items()}
    A=dr.div(dr.add(vals['plus_one'],vals['minus_one']),dr.integer(2))
    O=vals['zero']
    C=dr.div(dr.add(vals['plus_one_over_16'],vals['minus_one_over_16']),dr.integer(2))
    within=dr.sub(A,O)
    cross=dr.sub(O,C)
    between=dr.add(within,dr.mul(dr.integer(d),cross))
    global_eig=dr.add(dr.add(A,dr.mul(dr.integer(d-1),O)),dr.mul(dr.integer(N-d),C))
    uniform_energy=dr.div(global_eig,dr.integer(N))
    transverse_min=I(min(within.lo,between.lo),min(within.hi,between.hi))

    def enc(x:I):
        return {'lo':str(x.lo),'hi':str(x.hi),'width':str(x.hi-x.lo)}
    result={
        'status':'PASS' if within.lo>0 and between.lo>0 and global_eig.lo>0 else 'FAIL',
        'kernel_definition':'K_0(t)=t; K_{r+1}=kappa(K_r); depth=32',
        'dimension':d,
        'number_of_real_MUBs':M,
        'number_of_lines':N,
        'point_values':{k:enc(v) for k,v in vals.items()},
        'line_association_values':{'A':enc(A),'O':enc(O),'C':enc(C)},
        'eigenvalues':{
            'within_basis_zero_sum':enc(within),
            'between_basis_zero_sum':enc(between),
            'global':enc(global_eig),
            'uniform_line_energy':enc(uniform_energy),
            'zero_sum_stability_modulus_min':enc(transverse_min),
        },
        'multiplicities':{
            'within_basis_zero_sum':M*(d-1),
            'between_basis_zero_sum':M-1,
            'global':1,
        },
        'interpretation':'The K32 symmetrized line Gram matrix is positive definite. Uniform mass-one line weights and the free-mass scaled-uniform line weights are unique; fixed-mass excess risk is at least the certified transverse modulus times squared Euclidean weight distance.',
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)

if __name__=='__main__':
    main()
