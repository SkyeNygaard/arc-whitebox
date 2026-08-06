import numpy as np
from l08_abstention_gate import layer8_basis_fold_relative_dispersion, should_apply_k32, N_BASES, ROWS_PER_BASIS, WIDTH

def main():
    x=np.ones((N_BASES*ROWS_PER_BASIS,WIDTH),dtype=np.float32)
    assert abs(layer8_basis_fold_relative_dispersion(x))<1e-12
    assert should_apply_k32(x,1e-6)
    y=x.copy(); y[:ROWS_PER_BASIS]*=2
    assert layer8_basis_fold_relative_dispersion(y)>0
    try: layer8_basis_fold_relative_dispersion(np.ones((3,4)))
    except ValueError: pass
    else: raise AssertionError('shape check missing')
    print('gate tests pass')
if __name__=='__main__':main()
