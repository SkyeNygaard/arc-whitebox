"""Chunked pairwise CoefNet correction using only flopscope.numpy.

This is the submission-ready ML portion. The factorized-K3 port should provide
mean, covariance and the directed k21 slice for one pre-ReLU layer.
"""
from __future__ import annotations
import flopscope.numpy as fnp


def make_upper_pairs(width: int):
    # Setup-time/static Python construction; convert once to flopscope arrays.
    ii=[];jj=[]
    for i in range(width):
        for j in range(i+1,width):
            ii.append(i);jj.append(j)
    return fnp.asarray(ii,dtype=fnp.int32),fnp.asarray(jj,dtype=fnp.int32)


def correction_from_k21(model, mean, covariance, k21, layer: int, depth: int,
                         pair_i, pair_j, alpha: float, x1_scale: float = 1.0,
                         x1a_scale: float = 1.0, chunk_size: int = 4096,
                         x_clip: float = 20.0, residual_clip: float = 0.5):
    width=mean.shape[0]
    variance=fnp.maximum(fnp.diag(covariance),1e-12)
    sigma=fnp.sqrt(variance);a=mean/sigma
    denom=fnp.maximum(sigma[pair_i]**3+sigma[pair_j]**3,1e-12)
    x1=(k21[pair_i,pair_j]+k21[pair_j,pair_i])/denom*x1_scale
    x1a=(k21[pair_i,pair_j]-k21[pair_j,pair_i])/denom*x1a_scale
    x1=fnp.clip(x1,-x_clip,x_clip);x1a=fnp.clip(x1a,-x_clip,x_clip)
    values=[]
    count=pair_i.shape[0]
    for start in range(0,count,chunk_size):
        stop=min(start+chunk_size,count);i=pair_i[start:stop];j=pair_j[start:stop]
        d=a[i]-a[j]
        features=fnp.stack((fnp.full(d.shape,(layer+1)/depth,dtype=mean.dtype),
                            a[i]+a[j],a[i]*a[j],fnp.abs(d),
                            fnp.clip(covariance[i,j]/fnp.maximum(sigma[i]*sigma[j],1e-12),-1.0,1.0)),axis=1)
        coef=model(features)
        norm=fnp.clip(coef[:,0]*x1[start:stop]+d*coef[:,1]*x1a[start:stop],-residual_clip,residual_clip)
        values.append(alpha*norm*sigma[i]*sigma[j])
    values=fnp.concatenate(values,axis=0)
    out=fnp.zeros((width,width),dtype=mean.dtype)
    out[pair_i,pair_j]=values;out[pair_j,pair_i]=values
    return out
