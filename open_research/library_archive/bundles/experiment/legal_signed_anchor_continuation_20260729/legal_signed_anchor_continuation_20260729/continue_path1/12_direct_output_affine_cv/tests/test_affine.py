import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import numpy as np, torch
from direct_output_cv import capture_baseline_and_affine

def test_affine_mean_identity_small():
 torch.manual_seed(0);d=4;z=torch.randn(10,d);x=torch.cat([z,-z],0);ws=[torch.randn(d,d) for _ in range(3)]
 base,A,c,layers=capture_baseline_and_affine(ws,x.numpy().astype(np.float32))
 assert np.max(np.abs(base-c))<1e-6
 assert max(q['mean_identity_error'] for q in layers)<1e-6
