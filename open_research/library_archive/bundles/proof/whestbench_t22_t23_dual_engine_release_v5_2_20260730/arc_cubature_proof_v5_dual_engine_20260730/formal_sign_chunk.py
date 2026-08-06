from __future__ import annotations
import argparse,json
from pathlib import Path
from formal_proof_common import Problem

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=100);p.add_argument('--prec',type=int,default=55);a=p.parse_args()
 base=Path(__file__).resolve().parent;P=Problem(base,a.prec);mesh=P.mesh();end=min(len(mesh),a.start+a.count);rows=[]
 for idx in range(a.start,end):
  l,r,sgn,rid,j,parts=mesh[idx]
  stack=[(l,r,0)]
  while stack:
   u,v,lev=stack.pop();R=P.gpp_range(u,v);good=R.hi<0 if sgn=='negative' else R.lo>0
   if good:
    rows.append({'index':idx,'left':str(u),'right':str(v),'sign':sgn,'region':rid,'piece':j,'pieces':parts,'level':lev,'lower':str(R.lo),'upper':str(R.hi)})
   else:
    if lev>=12:raise RuntimeError((idx,float(u),float(v),sgn,R))
    m=(u+v)/2;stack.append((u,m,lev+1));stack.append((m,v,lev+1))
 out={'start':a.start,'end':end,'total':len(mesh),'precision':a.prec,'rows':rows,'passed':True}
 path=base/'results'/f'formal_gpp_chunk_{a.start:04d}_{end:04d}.json';path.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
