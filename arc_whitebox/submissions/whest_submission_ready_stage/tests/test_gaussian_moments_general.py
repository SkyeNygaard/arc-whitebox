import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_k3_coefnet_v2 import HybridConfig,HybridK3CoefNetV2


def main():
    model=Path(os.environ.get('WHEST_COEFNET','/mnt/data/whest_ml/coefnet_d256_w64.npz'))
    h16=HybridK3CoefNetV2(model,HybridConfig(),quadrature_nodes=20,dtype=torch.float64)
    h128=HybridK3CoefNetV2(model,HybridConfig(),quadrature_nodes=128,dtype=torch.float64)
    rng=np.random.default_rng(20260729);worst=0.;rms=[]
    for _ in range(2000):
        s=np.exp(rng.uniform(-1,1,size=2));rho=rng.uniform(-.999,.999);mu=rng.normal(0,1,size=2)*s
        cov=np.array([[s[0]**2,rho*s[0]*s[1]],[rho*s[0]*s[1],s[1]**2]])
        mt=torch.tensor(mu,dtype=torch.float64);ct=torch.tensor(cov,dtype=torch.float64)
        m1,c1=h16.gaussian_relu_moments(mt,ct);m2,c2=h128.gaussian_relu_moments(mt,ct)
        err=max(float(torch.max(torch.abs(m1-m2))),float(torch.max(torch.abs(c1-c2))))
        worst=max(worst,err);rms.append(err*err)
    rms=float(np.sqrt(np.mean(rms)))
    assert worst < 2e-7,(worst,rms)
    print({'random_cases':2000,'worst_abs_20_vs_128_nodes':worst,'rms_case_max_error':rms})
if __name__=='__main__':main()
