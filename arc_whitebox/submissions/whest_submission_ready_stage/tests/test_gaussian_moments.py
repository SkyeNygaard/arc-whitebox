import os
from pathlib import Path
import sys, math
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_k3_coefnet_v2 import HybridConfig,HybridK3CoefNetV2


def main():
    model=Path(os.environ.get('WHEST_COEFNET','/mnt/data/whest_ml/coefnet_d256_w64.npz'))
    h=HybridK3CoefNetV2(model,HybridConfig(),depth=32,dtype=torch.float64,quadrature_nodes=16)
    rhos=np.linspace(-.999,.999,1001)
    n=len(rhos)+1
    # Test 2x2 one rho at a time to include near-singular cases.
    worst=0.0
    for rho in rhos:
        cov=torch.tensor([[1.,rho],[rho,1.]],dtype=torch.float64)
        mu=torch.zeros(2,dtype=torch.float64)
        pm,pc=h.gaussian_relu_moments(mu,cov)
        second=float(pc[0,1]+pm[0]*pm[1])
        exact=(math.sqrt(max(0.,1-rho*rho))+(math.pi-math.acos(rho))*rho)/(2*math.pi)
        worst=max(worst,abs(second-exact))
    assert worst < 2e-7, worst
    print({'worst_zero_mean_second_moment_abs_error':worst})
if __name__=='__main__':main()
