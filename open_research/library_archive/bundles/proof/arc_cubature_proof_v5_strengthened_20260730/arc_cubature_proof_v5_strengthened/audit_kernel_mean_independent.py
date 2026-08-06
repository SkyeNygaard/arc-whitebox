from __future__ import annotations

"""Independent non-rigorous high-precision audit of the formal kernel mean.

This intentionally uses mpmath quadrature and a different representation:
t=sin(theta), so the spherical density becomes C*cos(theta)^254.
It is an implementation audit, not part of the formal proof.
"""

import json
from pathlib import Path
import mpmath as mp


def main():
    mp.mp.dps = 90
    base = Path(__file__).resolve().parent
    formal = json.loads((base/'results/FORMAL_KERNEL_MEAN_D256_L32.json').read_text())

    def kappa(x):
        return (mp.sqrt(1-x*x)+(mp.pi-mp.acos(x))*x)/mp.pi

    def kernel(x):
        for _ in range(32):
            x = kappa(x)
        return x

    C = mp.gamma(128)/(mp.sqrt(mp.pi)*mp.gamma(mp.mpf(255)/2))
    f = lambda th: kernel(mp.sin(th))*mp.cos(th)**254
    cuts = [-mp.pi/2, -mp.mpf('0.8'), -mp.mpf('0.4'), mp.mpf('0'),
            mp.mpf('0.4'), mp.mpf('0.8'), mp.pi/2]
    value = C*mp.quad(f, cuts)
    lo = mp.mpf(formal['A0_certified']['lower'])
    hi = mp.mpf(formal['A0_certified']['upper'])
    out = {
        'passed': bool(lo <= value <= hi),
        'method': '90-digit mpmath quadrature after t=sin(theta)',
        'value': mp.nstr(value, 90),
        'formal_lower': str(formal['A0_certified']['lower']),
        'formal_upper': str(formal['A0_certified']['upper']),
        'distance_from_lower': mp.nstr(value-lo, 50),
        'distance_from_upper': mp.nstr(hi-value, 50),
        'note': 'Independent audit only; not used as proof.'
    }
    assert out['passed']
    (base/'results/INDEPENDENT_KERNEL_MEAN_AUDIT.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
