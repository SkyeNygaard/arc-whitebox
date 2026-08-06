import numpy as np
from direct_pair_estimator import (
    PairContractions, accumulate_selected_pair_moments, lower_anchor_defect,
    final_output_correction, output_metric,
)

def test_direct_contractions():
    rng=np.random.default_rng(7); n,d,p=503,20,9
    h=rng.normal(size=(n,d)); idx=rng.choice(d,p,replace=False); v=rng.normal(size=(p,d)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    scale=1.37
    got=accumulate_selected_pair_moments(h,idx,v,second_moment_scale=scale,block_rows=73)
    M=scale*(h.T@h/n)
    np.testing.assert_allclose(got.marginal_second,np.diag(M)[idx],rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(got.row_direction,np.einsum('pd,pd->p',M[idx],v),rtol=2e-14,atol=2e-14)

def test_lower_identity():
    rng=np.random.default_rng(11); d,p=23,10
    A=rng.normal(size=(3000,d)); M=A.T@A/len(A); mu=rng.normal(size=d); m=rng.normal(size=d)
    idx=rng.choice(d,p,replace=False); v=rng.normal(size=(p,d)); v/=np.linalg.norm(v,axis=1,keepdims=True)
    pair=PairContractions(np.diag(M)[idx],np.einsum('pd,pd->p',M[idx],v))
    got=lower_anchor_defect(mu,m,idx,v,pair)
    delta=mu-m; q=np.diag(M)
    L=(q[:,None]*delta[None,:]+2*delta[:,None]*M+2*(m*m-mu*mu)[:,None]*mu[None,:])/(d+1)
    want=np.einsum('pd,pd->p',L[idx],v)
    np.testing.assert_allclose(got,want,rtol=3e-14,atol=3e-14)

def test_output_quadratic():
    rng=np.random.default_rng(13); e=rng.normal(size=17); c=rng.normal(size=17)
    q=output_metric(e,c)
    np.testing.assert_allclose(q['corrected_error_norm_squared'],np.dot(e+c,e+c),rtol=1e-14)
    beta=rng.normal(size=(5,17)); a=rng.normal(size=5)
    np.testing.assert_allclose(final_output_correction(a,beta),a@beta)

if __name__=='__main__':
    test_direct_contractions(); test_lower_identity(); test_output_quadratic(); print('all tests passed')
