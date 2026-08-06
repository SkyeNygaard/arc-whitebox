#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

def rows(path:Path):
    out=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        out.append(line.split('\t'))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mesh',type=Path,required=True)
    ap.add_argument('--certified',type=Path,required=True)
    ap.add_argument('--global-boxes',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    mesh=[]
    region_rows=defaultdict(list)
    for r in rows(a.mesh):
        idx=int(r[0]); left=Fraction(r[1]); right=Fraction(r[2]); sign=r[3]; region=int(r[4]); piece=int(r[5]); pieces=int(r[6])
        assert idx==len(mesh) and left<right and sign in {'positive','negative'}
        mesh.append((left,right,sign,region,piece,pieces))
        region_rows[region].append((piece,pieces,left,right,sign))
    assert len(mesh)==1079
    for region,rr in region_rows.items():
        rr.sort()
        pieces=rr[0][1]
        assert len(rr)==pieces and [x[0] for x in rr]==list(range(pieces))
        assert all(x[1]==pieces for x in rr)
        assert all(rr[i][3]==rr[i+1][2] for i in range(len(rr)-1))
        assert len({x[4] for x in rr})==1

    cert=defaultdict(list)
    for r in rows(a.certified):
        idx=int(r[0]); left=Fraction(r[1]); right=Fraction(r[2]); sign=r[3]; region=int(r[4]); piece=int(r[5]); pieces=int(r[6]); level=int(r[7])
        assert 0<=idx<len(mesh) and left<right and level>=0
        cert[idx].append((left,right,sign,region,piece,pieces,level))
    assert sum(map(len,cert.values()))==1421 and set(cert)==set(range(1079))
    max_level=0
    for idx,(ml,mr,ms,mreg,mpiece,mpieces) in enumerate(mesh):
        cr=sorted(cert[idx], key=lambda x:x[0])
        assert cr[0][0]==ml and cr[-1][1]==mr
        assert all(cr[i][1]==cr[i+1][0] for i in range(len(cr)-1))
        for left,right,sign,region,piece,pieces,level in cr:
            assert sign==ms and region==mreg and piece==mpiece and pieces==mpieces
            assert right-left == (mr-ml)/2**level
            offset=(left-ml)/(mr-ml)*2**level
            assert offset.denominator==1
            max_level=max(max_level,level)

    boxes={}
    for r in rows(a.global_boxes):
        kind,name,left,right,sign,*rest=r
        boxes[(kind,name)]={'left':Fraction(left),'right':Fraction(right),'sign':sign,'rest':rest}
        assert Fraction(left)<Fraction(right)
    # Exact full-domain chain. Endpoint-right is a redundant separately checked box.
    chain=[boxes[('END','left')],boxes[('TAIL','left_tail')],boxes[('CRIT','0')]]
    next_objects=[('INFL','0'),('CRIT','1'),('INFL','1'),('CRIT','2'),('INFL','2'),('CRIT','3'),('INFL','3'),('CRIT','4')]
    for region in range(9):
        rr=sorted(region_rows[region])
        chain.append({'left':rr[0][2],'right':rr[-1][3]})
        if region<8:
            chain.append(boxes[next_objects[region]])
    chain.append(boxes[('TAIL','right_tail')])
    assert chain[0]['left']==Fraction(-1) and chain[-1]['right']==Fraction(1)
    assert all(chain[i]['right']==chain[i+1]['left'] for i in range(len(chain)-1)), [(i,chain[i]['right'],chain[i+1]['left']) for i in range(len(chain)-1) if chain[i]['right']!=chain[i+1]['left']]
    er=boxes[('END','right')]
    rt=boxes[('TAIL','right_tail')]
    assert rt['left']<=er['left'] and er['right']==rt['right']==Fraction(1)

    out={
        'mesh_original_intervals':len(mesh),
        'certified_subintervals':sum(map(len,cert.values())),
        'split_count':sum(map(len,cert.values()))-len(mesh),
        'maximum_depth':max_level,
        'regions':len(region_rows),
        'exact_subinterval_coverage':True,
        'dyadic_subdivision_verified':True,
        'full_domain_coverage_minus1_to1':True,
        'no_gaps_or_overlaps':True,
        'passed':True,
    }
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
