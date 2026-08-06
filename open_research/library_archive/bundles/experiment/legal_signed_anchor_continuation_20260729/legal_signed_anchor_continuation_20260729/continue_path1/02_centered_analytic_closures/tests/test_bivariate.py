import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from analytic_defect_closures import bvn_cdf_varying, relu_bivariate


def test_independent_cdf():
    a=np.array([[0.3,-0.7],[1.2,0.0]])
    b=np.array([[0.2,0.4],[-0.5,0.8]])
    rho=np.zeros_like(a)
    from scipy.special import ndtr
    assert np.max(np.abs(bvn_cdf_varying(a,b,rho)-ndtr(a)*ndtr(b))) < 1e-12


def test_relu_bivariate_mc():
    rng=np.random.default_rng(123)
    mu=np.array([0.25,-0.35,0.8])
    A=np.array([[1.0,.25,-.1],[.25,.7,.15],[-.1,.15,1.3]])
    m,c=relu_bivariate(mu,A)
    z=rng.multivariate_normal(mu,A,size=800000)
    z=np.maximum(z,0)
    mm=z.mean(0); cc=np.cov(z,rowvar=False,bias=True)
    assert np.max(np.abs(m-mm)) < 4e-3
    assert np.max(np.abs(c-cc)) < 5e-3
