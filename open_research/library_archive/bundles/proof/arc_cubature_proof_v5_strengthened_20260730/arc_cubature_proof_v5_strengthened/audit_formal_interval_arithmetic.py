from __future__ import annotations

"""Independent high-precision point audit of the formal interval primitives.

This file is deliberately independent of the discovery/optimization code.  It
uses direct mpmath formulas for the ReLU kernel and its first two derivatives.
The audit is not part of the proof; it is a clean implementation cross-check.
"""

import json
import random
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import mpmath as mp

from formal_interval_certificate import (
    Directed, I, pi_bounds, asin_scalar, kappa_pair_scalar,
    deep_kernel_and_prime_fraction,
)
from formal_meanvalue_certificate import deep_second_interval

mp.mp.dps = 100


def kappa_mp(x: mp.mpf) -> mp.mpf:
    return (mp.sqrt(1 - x*x) + (mp.pi - mp.acos(x))*x) / mp.pi


def kappa_prime_mp(x: mp.mpf) -> mp.mpf:
    return (mp.pi - mp.acos(x)) / mp.pi


def deep_value_derivatives_mp(x: mp.mpf, depth: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    f = mp.mpf(x)
    p = mp.mpf(1)
    q = mp.mpf(0)
    for _ in range(depth):
        kp = kappa_prime_mp(f)
        kpp = 1 / (mp.pi * mp.sqrt(1 - f*f)) if abs(f) < 1 else mp.inf
        q = kpp*p*p + kp*q
        p = kp*p
        f = kappa_mp(f)
    return f, p, q


def main() -> None:
    rng = random.Random(20260728)
    dr = Directed(55)
    pl, ph = pi_bounds(85)
    pi = I(dr.Dlo(pl), dr.Dhi(ph))
    tests = {'pi': 0, 'asin': 0, 'kappa': 0, 'deep': 0, 'second_interval_samples': 0}

    assert mp.mpf(str(pi.lo)) < mp.pi < mp.mpf(str(pi.hi))
    tests['pi'] = 1

    for _ in range(120):
        x = Decimal(str(rng.uniform(-0.999999, 0.999999)))
        A = asin_scalar(x, dr, pi)
        truth = mp.asin(mp.mpf(str(x)))
        assert mp.mpf(str(A.lo)) <= truth <= mp.mpf(str(A.hi))
        tests['asin'] += 1

        K, Kp = kappa_pair_scalar(x, dr, pi)
        xm = mp.mpf(str(x))
        kt = kappa_mp(xm)
        kpt = kappa_prime_mp(xm)
        assert mp.mpf(str(K.lo)) <= kt <= mp.mpf(str(K.hi))
        assert mp.mpf(str(Kp.lo)) <= kpt <= mp.mpf(str(Kp.hi))
        tests['kappa'] += 1

    for _ in range(60):
        f = Fraction(rng.randint(-999999, 999999), 1000000)
        K, Kp = deep_kernel_and_prime_fraction(f, 32, dr, pi)
        truth, p, _ = deep_value_derivatives_mp(mp.mpf(f.numerator)/f.denominator, 32)
        assert mp.mpf(str(K.lo)) <= truth <= mp.mpf(str(K.hi))
        assert mp.mpf(str(Kp.lo)) <= p <= mp.mpf(str(Kp.hi))
        tests['deep'] += 1

    for _ in range(30):
        a = rng.uniform(-0.95, 0.70)
        width = rng.uniform(1e-5, 0.02)
        b = min(0.74, a + width)
        fa, fb = Fraction(str(a)), Fraction(str(b))
        Q = deep_second_interval(fa, fb, 32, dr, pi)
        for j in range(11):
            x = mp.mpf(str(a + (b-a)*j/10))
            q = deep_value_derivatives_mp(x, 32)[2]
            assert mp.mpf(str(Q.lo)) <= q <= mp.mpf(str(Q.hi))
            tests['second_interval_samples'] += 1

    out = {
        'passed': True,
        'tests': tests,
        'note': 'Independent high-precision point checks; this audits implementation but is not itself the proof.',
    }
    base = Path(__file__).resolve().parent
    (base/'results/FORMAL_INTERVAL_AUDIT.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
