from __future__ import annotations
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from formal_proof_common import Problem

RANGES=[(0,100),(100,200),(200,300),(300,400),(400,500),(500,550),(550,551),(551,560),(560,565),(565,570),(570,575),(575,585),(585,600),(600,605),(605,606),(606,607),(607,613),(613,627),(627,727),(727,827),(827,927),(927,1027),(1027,1079)]

def main():
 base=Path(__file__).resolve().parent;res=base/'results';P=Problem(base);mesh=P.mesh();by_index={}
 sources=[]
 for a,b in RANGES:
  p=res/f'formal_gpp_chunk_{a:04d}_{b:04d}.json';d=json.loads(p.read_text());assert d['passed'] and d['start']==a and d['end']==b
  sources.append(p.name)
  for row in d['rows']:by_index.setdefault(row['index'],[]).append(row)
 assert set(by_index)==set(range(len(mesh))), (set(range(len(mesh)))-set(by_index))
 assembled=[];min_margin=None;min_row=None
 for idx,(a,b,sgn,rid,j,parts) in enumerate(mesh):
  rows=sorted(by_index[idx],key=lambda x:Fraction(x['left']))
  assert Fraction(rows[0]['left'])==a and Fraction(rows[-1]['right'])==b
  for x,y in zip(rows[:-1],rows[1:]):assert Fraction(x['right'])==Fraction(y['left'])
  for r in rows:
   assert r['sign']==sgn
   lo=Decimal(r['lower']);hi=Decimal(r['upper'])
   margin=hi.copy_negate() if sgn=='negative' else lo
   assert margin>0
   if min_margin is None or margin<min_margin:min_margin=margin;min_row=r
  assembled.extend(rows)
 basecert=json.loads((res/'formal_base_certificate_d256_L32.json').read_text());assert basecert['passed']
 out={
  'theorem_target':'h(t) <= K_32(t) for every t in [-1,1]',
  'dimension':256,'depth':32,'precision':55,
  'base_certificate':basecert,
  'mesh_original_intervals':len(mesh),'mesh_certified_subintervals':len(assembled),
  'mesh_sources':sources,
  'minimum_curvature_sign_margin':str(min_margin),'minimum_margin_row':min_row,
  'coverage_exact':True,'no_gaps_or_overlaps':True,
  'logical_conclusion':[
    'g=h-K has positive derivative on the left tail and negative derivative on the right tail.',
    'The certified sign pattern of g second derivative and the four inflection boxes forces exactly five derivative sign transitions.',
    'The three possible local maxima lie inside critical boxes 0, 2, and 4.',
    'Strong-concavity bounds make g strictly negative in all three maximum boxes.',
    'The two minimum boxes and both endpoint boxes are also strictly negative.',
    'Therefore h(t) < K_32(t) for every t in [-1,1].'
  ],
  'global_upper_bound':basecert['global_candidate_upper_bound'],'passed':True,
  'curvature_intervals':assembled,
 }
 (res/'FORMAL_CERTIFICATE_D256_L32.json').write_text(json.dumps(out,indent=2))
 summary={k:v for k,v in out.items() if k not in ('curvature_intervals','base_certificate')}
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
