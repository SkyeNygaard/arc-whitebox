#!/usr/bin/env python3
"""Offline converter from the training NPZ to a pickle-free flops.Module file."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import flopscope.numpy as fnp
from coefnet_flops import CoefNet

p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--dtype',choices=['float32','float64'],default='float32');a=p.parse_args()
with np.load(a.input) as d:
    hidden=int(d['W0'].shape[0]); model=CoefNet(n_in=int(d['W0'].shape[1]),hidden=hidden,n_out=int(d['W2'].shape[0]))
    dtype=np.float32 if a.dtype=='float32' else np.float64
    for name in ('mean','std','W0','b0','W1','b1','W2','b2'):
        setattr(model,name,fnp.asarray(np.asarray(d[name],dtype=dtype)))
a.output.parent.mkdir(parents=True,exist_ok=True);model.save(a.output)
print({'output':str(a.output),'hidden':hidden,'dtype':a.dtype,'bytes':a.output.stat().st_size})
