from __future__ import annotations
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from formal_proof_common import Problem,F,GP_RADII,GPP_CENTERS,GPP_RADII,RIGHT_CUT
from formal_interval_certificate import bernstein_on_minus1_1


def build(base:Path,prec=55):
    P=Problem(base,prec)
    dr=P.dr
    bern=bernstein_on_minus1_1(P.hp)
    assert all(x>0 for x in bern)
    assert all(x>=0 for x in P.coeff[1:])

    critical=[]
    for idx,(c,r) in enumerate(zip(P.gp_centers,GP_RADII)):
        a,b=c-r,c+r
        gpp=P.gpp_range(a,b)
        gc=P.gval(c);gpc=P.gp_range(c,c)
        if idx in (0,2,4):
            assert gpp.hi<0
            m=gpp.hi.copy_negate();A=max(gpc.lo.copy_abs(),gpc.hi.copy_abs())
            with localcontext(dr.up): corr=A*A/(Decimal(2)*m)
            
            with localcontext(dr.up): ub=gc.hi+corr
            kind='strictly_concave_maximum_box'
        else:
            assert gpp.lo>0
            ub=max(P.gval(a).hi,P.gval(b).hi);kind='strictly_convex_minimum_box'
        assert ub<0
        lgp=P.gp_range(a,a);rgp=P.gp_range(b,b)
        if idx in (0,2,4): assert lgp.lo>0 and rgp.hi<0
        else: assert lgp.hi<0 and rgp.lo>0
        critical.append({'index':idx,'center':str(c),'radius':str(r),'left':str(a),'right':str(b),
          'kind':kind,'gpp_lower':str(gpp.lo),'gpp_upper':str(gpp.hi),'g_upper':str(ub),
          'center_g':[str(gc.lo),str(gc.hi)],'center_gp':[str(gpc.lo),str(gpc.hi)],
          'left_gp':[str(lgp.lo),str(lgp.hi)],'right_gp':[str(rgp.lo),str(rgp.hi)]})

    inflection=[];expected=['negative','positive','negative','positive']
    for idx,(c,r,sgn) in enumerate(zip(GPP_CENTERS,GPP_RADII,expected)):
        a,b=c-r,c+r;gpc=P.gp_range(c,c)
        # Bound |g''| over 12 pieces, then apply the mean-value theorem.
        vals=[];parts=12
        for j in range(parts):
            u=a+(b-a)*j/parts;v=a+(b-a)*(j+1)/parts;vals.append(P.gpp_range(u,v))
        M=max(max(x.lo.copy_abs(),x.hi.copy_abs()) for x in vals)
        with localcontext(dr.up): delta=M*dr.Dhi(r)
        
        with localcontext(dr.down): Rlo=gpc.lo-delta
        with localcontext(dr.up): Rhi=gpc.hi+delta
        R=(Rlo,Rhi)
        if sgn=='negative': assert R[1]<0
        else: assert R[0]>0
        inflection.append({'index':idx,'center':str(c),'radius':str(r),'left':str(a),'right':str(b),
          'gp_sign':sgn,'gp_lower':str(R[0]),'gp_upper':str(R[1]),
          'center_gp':[str(gpc.lo),str(gpc.hi)],'gpp_abs_bound':str(M)})

    er=F('1e-7');left_endpoint=(F('-1'),F('-1')+er);right_endpoint=(F('1')-er,F('1'))
    endpoint=[]
    for name,(a,b) in [('left',left_endpoint),('right',right_endpoint)]:
        Ka,_=P.state(a)
        with localcontext(dr.up): ub=P.pval(P.hI,b).hi-Ka.lo
        assert ub<0
        endpoint.append({'name':name,'left':str(a),'right':str(b),'g_upper':str(ub)})
    left_tail=(left_endpoint[1],F(critical[0]['left']));left_gp=P.gp_range(*left_tail);assert left_gp.lo>0
    Htail=P.prange(P.hpI,RIGHT_CUT,F('1'));_,Kpc=P.state(RIGHT_CUT);assert Htail.hi<Kpc.lo

    max_ubs=[Decimal(critical[i]['g_upper']) for i in (0,2,4)]+[Decimal(x['g_upper']) for x in endpoint]
    return {
      'dimension':P.d,'depth':P.depth,'precision':prec,
      'pi_lower':str(P.pi.lo),'pi_upper':str(P.pi.hi),
      'gegenbauer_coefficients':[str(x) for x in P.coeff],
      'gegenbauer_coefficients_nonnegative':all(x>=0 for x in P.coeff[1:]),
      'hprime_bernstein_coefficients':[str(x) for x in bern],
      'h_strictly_increasing':True,
      'critical_boxes':critical,'inflection_boxes':inflection,'endpoint_boxes':endpoint,
      'left_tail':{'left':str(left_tail[0]),'right':str(left_tail[1]),'gp_lower':str(left_gp.lo),'gp_upper':str(left_gp.hi)},
      'right_tail':{'left':str(RIGHT_CUT),'right':'1','hprime_upper':str(Htail.hi),'Kprime_lower':str(Kpc.lo)},
      'global_candidate_upper_bound':str(max(max_ubs)),'passed':True,
    }

if __name__=='__main__':
 base=Path(__file__).resolve().parent;out=build(base);p=base/'results/formal_base_certificate_d256_L32.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('critical_boxes','inflection_boxes')},indent=2))
