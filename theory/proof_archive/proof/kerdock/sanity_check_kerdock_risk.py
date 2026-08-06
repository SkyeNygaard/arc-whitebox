#!/usr/bin/env python3
"""Independent, non-directed numerical sanity check for complete Kerdock risk.

This is not part of the computer-assisted proof.  It evaluates the depth-32
normalized ReLU kernel directly with mpmath, uses the exact pair-count spectrum
of a complete real-MUB antipodal design, and independently integrates the
uniform-sphere kernel mean in one dimension.
"""
from pathlib import Path
import json
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'KERDOCK_RISK_SANITY_CHECK.json'

mp.mp.dps = 70
d = 256
B = 129
N = 2*d*B


def kappa(t):
    # Clamp only against harmless evaluation drift at the exact endpoints.
    t = min(mp.mpf(1), max(mp.mpf(-1), mp.mpf(t)))
    return (mp.sqrt(max(mp.mpf('0'), 1-t*t)) + (mp.pi-mp.acos(t))*t) / mp.pi


def K32(t):
    y = mp.mpf(t)
    for _ in range(32):
        y = kappa(y)
    return y

s = 1/mp.sqrt(d)
counts = {
    't=1': 1,
    't=-1': 1,
    't=0': 2*(d-1),
    't=+1/sqrt(d)': (B-1)*d,
    't=-1/sqrt(d)': (B-1)*d,
}
assert sum(counts.values()) == N
energy = (
    K32(1) + K32(-1) + 2*(d-1)*K32(0)
    + (B-1)*d*(K32(s)+K32(-s))
) / N

# If t = cos(theta), the inner product of two independent uniform sphere
# points has normalized density c*sin(theta)^(d-2) on theta in [0, pi].
c = mp.gamma(mp.mpf(d)/2) / (mp.sqrt(mp.pi)*mp.gamma(mp.mpf(d-1)/2))
mean = c * mp.quad(lambda th: K32(mp.cos(th))*mp.sin(th)**(d-2),
                   [0, mp.pi/2, mp.pi])
risk = energy - mean

cert_lo = mp.mpf('2.4336603575430029389091338017406054668573276382630724978671590845071104120856063e-7')
cert_hi = mp.mpf('2.4336603575430052276094665026697645914811206370055599695108464279151347033914533e-7')
# This ordinary high-precision quadrature is not directed.  We therefore test
# agreement with the certified interval only to a deliberately loose tolerance.
distance = mp.mpf('0') if cert_lo <= risk <= cert_hi else min(abs(risk-cert_lo), abs(risk-cert_hi))
assert distance < mp.mpf('1e-18'), (risk, cert_lo, cert_hi, distance)

record = {
    'status': 'PASS',
    'evidence_class': 'independent high-precision numerical sanity check; not a proof and not directed interval arithmetic',
    'dimension': d,
    'depth': 32,
    'bases': B,
    'node_count': N,
    'per_node_pair_counts': counts,
    'kernel_energy': mp.nstr(energy, 65),
    'uniform_kernel_mean': mp.nstr(mean, 65),
    'kerdock_risk': mp.nstr(risk, 65),
    'certified_interval': [mp.nstr(cert_lo,65), mp.nstr(cert_hi,65)],
    'distance_to_certified_interval': mp.nstr(distance,20),
    'conclusion': 'Direct pair-spectrum evaluation and independent one-dimensional integration agree with the certified Kerdock-risk interval to much better than 1e-18 absolute.'
}
OUT.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
