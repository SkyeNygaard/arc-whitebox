from __future__ import annotations
from typing import Any
import math

WIDTH=256
KERDOCK_BASES=128
INV_SQRT_2PI=1.0/math.sqrt(2.0*math.pi)

def mean_gaussian_radius(width: int=WIDTH) -> float:
    return math.sqrt(2.0)*math.exp(math.lgamma((width+1.0)/2.0)-math.lgamma(width/2.0))

def fwht_axis_one(values: Any, xp: Any) -> Any:
    span=1
    while span<WIDTH:
        grouped=values.reshape((KERDOCK_BASES,WIDTH//(2*span),2,span,WIDTH))
        left,right=grouped[:,:,0,:,:],grouped[:,:,1,:,:]
        values=xp.stack((left+right,left-right),axis=2).reshape((KERDOCK_BASES,WIDTH,WIDTH))
        span*=2
    return values

def first_layer_design(first_weight: Any,rotation: Any,chirps: Any,xp: Any) -> Any:
    effective=rotation@first_weight
    radius=mean_gaussian_radius(WIDTH)
    weighted=chirps[:,:,None]*effective[None,:,:]
    pre=fwht_axis_one(weighted,xp)*(radius/math.sqrt(WIDTH))
    rows=xp.stack((pre,-pre),axis=2).reshape((-1,WIDTH))
    coord=xp.stack((radius*effective,-radius*effective),axis=1).reshape((-1,WIDTH))
    return xp.maximum(xp.concatenate((rows,coord),axis=0),0.0)
