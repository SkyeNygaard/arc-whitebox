from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path
from formal_interval_certificate import Directed,I,pi_bounds,auxiliary_monomials,deep_kernel_and_prime_fraction

D=256
DEPTH=32
N=66048
A=Fraction(1,16)


def scale(dr:Directed, x:I, q:Fraction)->I:
    return dr.mul(x, dr.frac_interval(q))


def main(prec:int=70):
    base=Path(__file__).resolve().parent
    dr=Directed(prec)
    pl,ph=pi_bounds(prec+35)
    pi=I(dr.Dlo(pl),dr.Dhi(ph))
    coeff,_=auxiliary_monomials(str(base/'auxiliary_coefficients_d256_L32_deg5.json'),D)
    assert all(c>=0 for c in coeff[1:])
    c0=coeff[0]
    h1=sum(coeff)  # normalized Gegenbauer basis has G_l(1)=1
    assert h1 <= 1

    def K(x:Fraction)->I:
        return deep_kernel_and_prime_fraction(x,DEPTH,dr,pi)[0]

    vals={
        'minus_one':K(Fraction(-1)),
        'minus_one_sixteenth':K(-A),
        'zero':K(Fraction(0)),
        'plus_one_sixteenth':K(A),
        'one':K(Fraction(1)),
    }

    # Every row of the 66,048-point real-MUB/Kerdock Gram matrix contains:
    # 1 self, 1 antipode, 510 orthogonal points, and 32,768 points at each
    # of +/-1/16.  Divide the row sum by N to obtain the uniform energy.
    row=dr.add(vals['one'], vals['minus_one'])
    row=dr.add(row, scale(dr,vals['zero'],Fraction(510)))
    row=dr.add(row, scale(dr,dr.add(vals['minus_one_sixteenth'],vals['plus_one_sixteenth']),Fraction(32768)))
    energy=scale(dr,row,Fraction(1,N))

    # Delsarte bound for any probability rule supported on at most N nodes
    # with nonnegative weights:
    # E_K >= c0 + (K(1)-h(1))*sum_i w_i^2
    #     >= c0 + (1-h(1))/N.
    bound_frac=c0 + (Fraction(1)-h1)/N
    bound=dr.frac_interval(bound_frac)
    gap=dr.sub(energy,bound)
    assert gap.lo >= 0

    out={
      'dimension':D,'depth':DEPTH,'node_budget':N,'precision':prec,
      'assumptions':['weights are nonnegative','weights sum to one','support size is at most N','kernel is K_32 as explicitly defined in the theorem package'],
      'auxiliary_coefficients_exact':[str(c) for c in coeff],
      'higher_coefficients_nonnegative':all(c>=0 for c in coeff[1:]),
      'c0_exact':str(c0),'h1_exact':str(h1),
      'kernel_intervals':{k:{'lower':str(v.lo),'upper':str(v.hi)} for k,v in vals.items()},
      'kerdock_energy':{'lower':str(energy.lo),'upper':str(energy.hi)},
      'universal_energy_lower_bound_exact':str(bound_frac),
      'universal_energy_lower_bound':{'lower':str(bound.lo),'upper':str(bound.hi)},
      'kerdock_minus_universal_bound':{'lower':str(gap.lo),'upper':str(gap.hi),'interpretation':'Interval for E_Kerdock minus the certificate lower bound; it is an upper bound on actual Kerdock suboptimality, not the unknown true gap.'},
      'theorem':'For every nonnegative weighted cubature rule using at most 66,048 nodes, its kernel energy is at least the stated universal bound. The Kerdock energy minus that bound has the stated interval. After subtracting the common spherical mean, the upper endpoint bounds Kerdock suboptimality in MSE.',
      'passed':True,
    }
    path=base/'results/FORMAL_DELSARTE_BOUND_D256_L32.json'
    path.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    main()
