D=256; L=32; G=2; N=1024; NF=129*512

def test_nominal_compute_band():
    core=G*L*(2*N*D*D)+L*(4*D**3)
    low=core+L*2*(2*NF*D)
    high=core+L*2*(3*NF*D)
    assert 12.8e9 < low < 13.0e9
    assert 13.9e9 < high < 14.1e9
