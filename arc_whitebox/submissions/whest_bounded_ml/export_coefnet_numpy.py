#!/usr/bin/env python3
"""Export a trained torch CoefNet checkpoint to a portable NumPy .npz."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch

def main():
 ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
 d=torch.load(a.checkpoint,map_location='cpu',weights_only=False);s=d['state_dict']
 # Supports checkpoints whose module is named body or b.
 prefix='body' if any(k.startswith('body.') for k in s) else 'b'
 ids=sorted({int(k.split('.')[1]) for k in s if k.startswith(prefix+'.') and k.endswith('.weight')})
 out={'mean':np.asarray(d['mean'],np.float32),'std':np.asarray(d['std'],np.float32),'hidden':np.int64(d.get('hidden',d.get('width',s[f'{prefix}.0.weight'].shape[0])))}
 for oi,idx in enumerate(ids):
  out[f'W{oi}']=s[f'{prefix}.{idx}.weight'].detach().numpy().astype(np.float32)
  out[f'b{oi}']=s[f'{prefix}.{idx}.bias'].detach().numpy().astype(np.float32)
 np.savez_compressed(a.output,**out);print(a.output,a.output.stat().st_size)
if __name__=='__main__':main()
