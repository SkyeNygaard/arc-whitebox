#!/usr/bin/env python3
from __future__ import annotations
import math, json
from dataclasses import dataclass
from functools import reduce
from operator import mul

def prod(x): return reduce(mul,x,1)

@dataclass
class Counter:
    costs: dict
    calls: dict
    def __init__(self): self.costs={}; self.calls={}
    def add(self,op,cost): self.costs[op]=self.costs.get(op,0)+int(cost); self.calls[op]=self.calls.get(op,0)+1
    @property
    def total(self): return sum(self.costs.values())

class T:
    def __init__(self,shape,c): self.shape=tuple(shape); self.c=c
    def __getitem__(self,key):
        if not isinstance(key,tuple): key=(key,)
        # expand ellipsis
        if Ellipsis in key:
            i=key.index(Ellipsis); missing=len(self.shape)-(len(key)-1); key=key[:i]+(slice(None),)*missing+key[i+1:]
        key=key+(slice(None),)*(len(self.shape)-len(key))
        out=[]
        for dim,k in zip(self.shape,key):
            if isinstance(k,int): continue
            if isinstance(k,slice):
                start,stop,step=k.indices(dim)
                out.append(max(0,(stop-start+(step-1))//step))
            else: raise TypeError(k)
        return T(out,self.c)
    def _binary(self,o,op):
        assert self.shape==o.shape,(self.shape,o.shape)
        self.c.add(op,prod(self.shape)); return T(self.shape,self.c)
    def __add__(self,o): return self._binary(o,'add')
    def __sub__(self,o): return self._binary(o,'subtract')
    def __matmul__(self,o):
        a=self.shape; b=o.shape
        assert a[-1]==b[-2],(a,b)
        # leading shapes should already match/broadcast; implement standard right alignment.
        la=list(a[:-2]); lb=list(b[:-2]); n=max(len(la),len(lb)); la=[1]*(n-len(la))+la; lb=[1]*(n-len(lb))+lb
        lead=[]
        for x,y in zip(la,lb):
            assert x==y or x==1 or y==1,(a,b)
            lead.append(max(x,y))
        m,k=a[-2],a[-1]; nn=b[-1]
        out=tuple(lead)+(m,nn)
        self.c.add('matmul',prod(lead)*m*nn*(2*k-1))
        return T(out,self.c)

class XP:
    def __init__(self,c): self.c=c
    def stack(self,arrs,axis):
        arrs=tuple(arrs); base=list(arrs[0].shape); nd=len(base)+1; axis=axis if axis>=0 else nd+axis
        for a in arrs: assert a.shape==arrs[0].shape
        out=base[:axis]+[len(arrs)]+base[axis:]
        self.c.add('stack',prod(out)); return T(out,self.c)
    def block(self,grid):
        # 2D nested list, concatenate cols on last axis then rows on second-last.
        rows=[]
        for row in grid:
            lead=row[0].shape[:-2]; r=row[0].shape[-2]; cols=sum(x.shape[-1] for x in row)
            assert all(x.shape[:-2]==lead and x.shape[-2]==r for x in row)
            rows.append(lead+(r,cols))
        lead=rows[0][:-2]; cols=rows[0][-1]; rr=sum(x[-2] for x in rows)
        assert all(x[:-2]==lead and x[-1]==cols for x in rows)
        out=lead+(rr,cols); self.c.add('block',prod(out)); return T(out,self.c)

def encode(left,right,xp):
    hr=left.shape[-2]//2; hi=left.shape[-1]//2; ho=right.shape[-1]//2
    a11=left[..., :hr,:hi]; a12=left[..., :hr,hi:]; a21=left[...,hr:,:hi]; a22=left[...,hr:,hi:]
    b11=right[..., :hi,:ho]; b12=right[..., :hi,ho:]; b21=right[...,hi:,:ho]; b22=right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    return xp.stack((a11,a12,s4,a22,s1,s2,s3),axis=-3), xp.stack((b11,b21,b22,t4,t1,t2,t3),axis=-3)

def decode(p,xp):
    p1,p2,p3,p4,p5,p6,p7=(p[...,i,:,:] for i in range(7))
    u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5
    return xp.block([[u1,u4+p3],[u3-p4,u3+p5]])

def level1(left,right,xp):
    hr=left.shape[-2]//2; hi=left.shape[-1]//2; ho=right.shape[-1]//2
    a11=left[..., :hr,:hi]; a12=left[..., :hr,hi:]; a21=left[...,hr:,:hi]; a22=left[...,hr:,hi:]
    b11=right[..., :hi,:ho]; b12=right[..., :hi,ho:]; b21=right[...,hi:,:ho]; b22=right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    p1=a11@b11; p2=a12@b21; p3=s4@b22; p4=a22@t4; p5=s1@t1; p6=s2@t2; p7=s3@t3
    u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5
    return u1,u4+p3,u3-p4,u3+p5

def depth2(left,right,xp):
    hr=left.shape[-2]//2; hi=left.shape[-1]//2; ho=right.shape[-1]//2
    a11=left[..., :hr,:hi]; a12=left[..., :hr,hi:]; a21=left[...,hr:,:hi]; a22=left[...,hr:,hi:]
    b11=right[..., :hi,:ho]; b12=right[..., :hi,ho:]; b21=right[...,hi:,:ho]; b22=right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    ps=(level1(a11,b11,xp),level1(a12,b21,xp),level1(s4,b22,xp),level1(a22,t4,xp),level1(s1,t1,xp),level1(s2,t2,xp),level1(s3,t3,xp))
    dec=[]
    for q in range(4):
        p1,p2,p3,p4,p5,p6,p7=(x[q] for x in ps)
        u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5
        dec.append((u1,u4+p3,u3-p4,u3+p5))
    return tuple(tuple(dec[leaf][root] for leaf in range(4)) for root in range(4))

def count_matmul(rows):
    c=Counter(); xp=XP(c); left=T((rows,256),c); right=T((256,256),c)
    for _ in range(3): left,right=encode(left,right,xp)
    tree=depth2(left,right,xp)
    p=xp.block([[tree[0][0],tree[0][1],tree[1][0],tree[1][1]],[tree[0][2],tree[0][3],tree[1][2],tree[1][3]],[tree[2][0],tree[2][1],tree[3][0],tree[3][1]],[tree[2][2],tree[2][3],tree[3][2],tree[3][3]]])
    for _ in range(3): p=decode(p,xp)
    assert p.shape==(rows,256),p.shape
    return c

def count_chunked(total_rows,chunk):
    costs={}; calls={}; ncall=0
    rem=total_rows
    while rem:
        r=min(chunk,rem); assert r%32==0
        c=count_matmul(r); ncall+=1
        for k,v in c.costs.items(): costs[k]=costs.get(k,0)+v
        for k,v in c.calls.items(): calls[k]=calls.get(k,0)+v
        rem-=r
    elems=total_rows*256
    # ReLU, cast, block sums, sum accumulation, final divide.
    costs['maximum']=elems
    costs['astype']=elems
    costs['sum']=total_rows*256 - ncall*256
    costs['accumulate_add']=ncall*256
    costs['divide']=256
    return {'chunk_rows':chunk,'calls':ncall,'operation_flops':costs,'operation_calls':calls,'matmul_kernel_flops':sum(v for k,v in costs.items() if k not in ['maximum','astype','sum','accumulate_add','divide']),'total_flops':sum(costs.values())}

if __name__=='__main__':
    base=count_matmul(66048)
    print('base',base.total,base.costs,base.calls)
    expected=5481223424
    assert base.total==expected,(base.total,expected)
    rows=[count_chunked(66048,c) for c in [512,1024,2048,4096,8192,16384,33024,66048]]
    print(json.dumps(rows,indent=2))
