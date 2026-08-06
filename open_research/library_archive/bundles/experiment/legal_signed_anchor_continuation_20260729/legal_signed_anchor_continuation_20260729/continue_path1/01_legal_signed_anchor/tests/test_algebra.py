import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reanchored_pilot_defect import lower_anchor_selected, D

def test_selected_matches_matrix():
    rng=np.random.default_rng(1); m=rng.normal(size=D); mu=rng.normal(size=D)
    A=rng.normal(size=(D,D)); M=A@A.T/D
    idx=np.array([1,5,9]); v=rng.normal(size=(3,D)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    d=mu-m
    full=(np.diag(M)[:,None]*d[None,:]+2*d[:,None]*M+2*(m*m-mu*mu)[:,None]*mu[None,:])/(D+1)
    got=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*v,axis=1),idx,v)
    exp=np.sum(full[idx]*v,axis=1)
    assert np.allclose(got,exp,rtol=1e-12,atol=1e-12)
