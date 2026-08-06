"""Pure NumPy runtime for the compact x1/x1a CoefNet."""
from __future__ import annotations
from pathlib import Path
import numpy as np
class NumpyCoefNet:
 def __init__(self,path):
  d=np.load(path);self.mean=d['mean'];self.std=d['std'];self.W=[];self.b=[];i=0
  while f'W{i}' in d:self.W.append(d[f'W{i}']);self.b.append(d[f'b{i}']);i+=1
  if len(self.W)!=3:raise ValueError('expected 3 linear layers')
 @staticmethod
 def silu(x):return x/(1+np.exp(-np.clip(x,-40,40)))
 def predict_invariant(self,base,d,x1,x1a):
  h=(np.asarray(base,np.float32)-self.mean)/self.std
  h=self.silu(h@self.W[0].T+self.b[0]);h=self.silu(h@self.W[1].T+self.b[1]);c=h@self.W[2].T+self.b[2]
  return c[:,0]*x1+d*c[:,1]*x1a
 def predict(self,X):
  X=np.asarray(X,np.float32);l,ai,aj,rho,x1,x1a=X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5];d=ai-aj
  base=np.column_stack([l,ai+aj,ai*aj,np.abs(d),rho]);return self.predict_invariant(base,d,x1,x1a)
