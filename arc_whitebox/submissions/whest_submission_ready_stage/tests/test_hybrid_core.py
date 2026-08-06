import os
from pathlib import Path
import sys, numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_k3_coefnet_v2 import HybridConfig,HybridK3CoefNetV2

class H:
 def __init__(self,core):self.core=core;self.r=0
 def to_tensor(self):return self.core
class K3:
 def __init__(self,k):self.k=k
 def get_dslice(self,p):assert tuple(p)==(2,1);return self.k

def main():
 model=Path(os.environ.get('WHEST_COEFNET','/mnt/data/whest_ml/coefnet_d256_w64.npz'))
 if not model.exists():raise SystemExit('missing model')
 n=16;g=torch.Generator().manual_seed(4);A=torch.randn(n,n,generator=g,dtype=torch.float64)/4;cov=A@A.T+torch.eye(n,dtype=torch.float64)*.2;mu=torch.randn(n,generator=g,dtype=torch.float64)*.2
 k=torch.randn(n,n,generator=g,dtype=torch.float64)*.01
 pre={1:H(mu.clone()),2:H(cov.clone()),3:K3(k)}
 post={1:H(torch.zeros(n,dtype=torch.float64)),2:H(torch.eye(n,dtype=torch.float64))}
 W=torch.randn(n,n,generator=g,dtype=torch.float64)/np.sqrt(n)
 cfg=HybridConfig(alpha=.5,beta=.3,gamma=.7,residual_clip=.5)
 h=HybridK3CoefNetV2(model,cfg,depth=32,dtype=torch.float64,quadrature_nodes=24)
 gm,gc=h.gaussian_relu_moments(mu,cov)
 assert torch.isfinite(gm).all() and torch.isfinite(gc).all()
 assert (gc.diagonal()>0).all()
 h.apply_(pre,post,5,next_weights=W)
 assert torch.isfinite(post[1].core).all() and torch.isfinite(post[2].core).all()
 assert torch.allclose(post[2].core,post[2].core.T,atol=1e-12)
 print({'minvar':float(post[2].core.diagonal().min()),**h.last_diagnostics})
if __name__=='__main__':main()
