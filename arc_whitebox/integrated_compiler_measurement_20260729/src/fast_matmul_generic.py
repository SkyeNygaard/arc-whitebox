"""Functionally identical depth-5 partial-tree Winograd with pluggable backend."""
from __future__ import annotations
from typing import Any


def _encode(left: Any, right: Any, xp: Any) -> tuple[Any, Any]:
    hr, hi, ho = left.shape[-2]//2, left.shape[-1]//2, right.shape[-1]//2
    a11,a12,a21,a22 = left[...,:hr,:hi],left[...,:hr,hi:],left[...,hr:,:hi],left[...,hr:,hi:]
    b11,b12,b21,b22 = right[...,:hi,:ho],right[...,:hi,ho:],right[...,hi:,:ho],right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    return xp.stack((a11,a12,s4,a22,s1,s2,s3),axis=-3), xp.stack((b11,b21,b22,t4,t1,t2,t3),axis=-3)


def _decode(products: Any, xp: Any) -> Any:
    p1,p2,p3,p4,p5,p6,p7 = (products[...,i,:,:] for i in range(7))
    u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5
    return xp.block([[u1,u4+p3],[u3-p4,u3+p5]])


def _level_one(left: Any, right: Any, xp: Any) -> tuple[Any,Any,Any,Any]:
    hr,hi,ho=left.shape[-2]//2,left.shape[-1]//2,right.shape[-1]//2
    a11,a12,a21,a22=left[...,:hr,:hi],left[...,:hr,hi:],left[...,hr:,:hi],left[...,hr:,hi:]
    b11,b12,b21,b22=right[...,:hi,:ho],right[...,:hi,ho:],right[...,hi:,:ho],right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    p1=a11@b11; p2=a12@b21; p3=s4@b22; p4=a22@t4; p5=s1@t1; p6=s2@t2; p7=s3@t3
    u1=p1+p2; u2=p1+p6; u3=u2+p7; u4=u2+p5
    return u1,u4+p3,u3-p4,u3+p5


def _depth_two(left: Any, right: Any, xp: Any) -> tuple[tuple[Any,...],...]:
    hr,hi,ho=left.shape[-2]//2,left.shape[-1]//2,right.shape[-1]//2
    a11,a12,a21,a22=left[...,:hr,:hi],left[...,:hr,hi:],left[...,hr:,:hi],left[...,hr:,hi:]
    b11,b12,b21,b22=right[...,:hi,:ho],right[...,:hi,ho:],right[...,hi:,:ho],right[...,hi:,ho:]
    s1=a21+a22; s2=s1-a11; s3=a11-a21; s4=a12-s2
    t1=b12-b11; t2=b22-t1; t3=b22-b12; t4=t2-b21
    products=(_level_one(a11,b11,xp),_level_one(a12,b21,xp),_level_one(s4,b22,xp),_level_one(a22,t4,xp),_level_one(s1,t1,xp),_level_one(s2,t2,xp),_level_one(s3,t3,xp))
    decoded=[]
    for q in range(4):
        p1,p2,p3,p4,p5,p6,p7=(p[q] for p in products)
        u1=p1+p2;u2=p1+p6;u3=u2+p7;u4=u2+p5
        decoded.append((u1,u4+p3,u3-p4,u3+p5))
    return tuple(tuple(decoded[leaf][root] for leaf in range(4)) for root in range(4))


def winograd_hybrid_p3_d5_partial_tree(left: Any, right: Any, xp: Any) -> Any:
    if left.shape[-2] % 32 or left.shape[-1] % 32 or right.shape[-1] % 32:
        raise ValueError("depth-5 partial-tree multiplication requires dimensions divisible by 32")
    a,b=left,right
    for _ in range(3): a,b=_encode(a,b,xp)
    tree=_depth_two(a,b,xp)
    products=xp.block([
        [tree[0][0],tree[0][1],tree[1][0],tree[1][1]],
        [tree[0][2],tree[0][3],tree[1][2],tree[1][3]],
        [tree[2][0],tree[2][1],tree[3][0],tree[3][1]],
        [tree[2][2],tree[2][3],tree[3][2],tree[3][3]],
    ])
    for _ in range(3): products=_decode(products,xp)
    return products
