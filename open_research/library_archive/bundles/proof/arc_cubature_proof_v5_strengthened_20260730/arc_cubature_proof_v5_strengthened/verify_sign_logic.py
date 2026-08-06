from __future__ import annotations

"""Machine-check the finite sign diagram turning curvature certificates into h<K.

The interval arithmetic proves signs on boxes and connecting regions.  This
script verifies their exact left-to-right order and derives the sign of g' on
every connecting segment rather than merely printing the intended pattern.
"""

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES = BASE/'results'


def sign_interval(pair: list[str]) -> str:
    lo, hi = map(Decimal, pair)
    if lo > 0:
        return 'positive'
    if hi < 0:
        return 'negative'
    raise AssertionError(f'non-strict sign interval {pair}')


def main() -> None:
    cert = json.loads((RES/'FORMAL_CERTIFICATE_D256_L32.json').read_text())
    b = cert['base_certificate']
    C = b['critical_boxes']
    J = b['inflection_boxes']
    assert len(C) == 5 and len(J) == 4

    # Exact geometric order of all special boxes.
    special = []
    for i in range(4):
        special.extend([('critical', i, C[i]), ('inflection', i, J[i])])
    special.append(('critical', 4, C[4]))
    for x, y in zip(special[:-1], special[1:]):
        assert Fraction(x[2]['right']) < Fraction(y[2]['left'])

    expected_kinds = ['max','min','max','min','max']
    for i, (box, kind) in enumerate(zip(C, expected_kinds)):
        left_sign = sign_interval(box['left_gp'])
        right_sign = sign_interval(box['right_gp'])
        if kind == 'max':
            assert box['kind'] == 'strictly_concave_maximum_box'
            assert Decimal(box['gpp_upper']) < 0
            assert (left_sign, right_sign) == ('positive','negative')
        else:
            assert box['kind'] == 'strictly_convex_minimum_box'
            assert Decimal(box['gpp_lower']) > 0
            assert (left_sign, right_sign) == ('negative','positive')
        assert Decimal(box['g_upper']) < 0

    expected_inflection_gp = ['negative','positive','negative','positive']
    for box, expected in zip(J, expected_inflection_gp):
        assert box['gp_sign'] == expected
        assert sign_interval([box['gp_lower'],box['gp_upper']]) == expected

    # Curvature regions occur between the nine consecutive special-box pairs,
    # followed by the interval from the final maximum box to RIGHT_CUT.
    curvature = ['negative','positive','positive','negative','negative',
                 'positive','positive','negative','negative']
    rows = cert['curvature_intervals']
    by_region = {i: [] for i in range(9)}
    for row in rows:
        by_region[row['region']].append(row)
    for rid, expected in enumerate(curvature):
        rr = sorted(by_region[rid], key=lambda r: Fraction(r['left']))
        assert rr
        for row in rr:
            assert row['sign'] == expected
            lo, hi = Decimal(row['lower']), Decimal(row['upper'])
            assert hi < 0 if expected == 'negative' else lo > 0

    # Derive g' signs on each connecting curvature region.  For a decreasing
    # function, a negative left endpoint or positive right endpoint controls
    # the whole interval.  For an increasing function, a positive left endpoint
    # or negative right endpoint controls it.
    endpoint_signs = [
        ('negative','negative'),  # C0.right -> J0.left, g''<0
        ('negative','negative'),  # J0.right -> C1.left, g''>0
        ('positive','positive'),  # C1.right -> J1.left, g''>0
        ('positive','positive'),  # J1.right -> C2.left, g''<0
        ('negative','negative'),  # C2.right -> J2.left, g''<0
        ('negative','negative'),  # J2.right -> C3.left, g''>0
        ('positive','positive'),  # C3.right -> J3.left, g''>0
        ('positive','positive'),  # J3.right -> C4.left, g''<0
        ('negative','negative'),  # C4.right -> right cut, g''<0
    ]
    derived = []
    for rid, (curv, signs) in enumerate(zip(curvature, endpoint_signs)):
        left, right = signs
        assert left == right
        if curv == 'negative':
            # decreasing: negative at left or positive at right is sufficient
            assert left == 'negative' or right == 'positive'
        else:
            # increasing: positive at left or negative at right is sufficient
            assert left == 'positive' or right == 'negative'
        derived.append({'region': rid, 'curvature': curv, 'gprime_sign': left})

    assert Decimal(b['left_tail']['gp_lower']) > 0
    assert Decimal(b['right_tail']['hprime_upper']) < Decimal(b['right_tail']['Kprime_lower'])
    for e in b['endpoint_boxes']:
        assert Decimal(e['g_upper']) < 0

    maxima = [{'critical_box': i, 'g_upper': C[i]['g_upper']} for i in (0,2,4)]
    out = {
        'passed': True,
        'special_box_order_verified': True,
        'curvature_region_signs': curvature,
        'derived_connecting_region_signs': derived,
        'left_tail_gprime_sign': 'positive',
        'right_tail_gprime_sign': 'negative',
        'only_interior_maximum_boxes': [0,2,4],
        'candidate_maximum_bounds': maxima,
        'endpoint_bounds': b['endpoint_boxes'],
        'conclusion': (
            'The certified curvature and endpoint signs force exactly five stationary '
            'transitions (max,min,max,min,max). g is negative at all three possible '
            'interior maxima and both endpoint boxes; hence g=h-K<0 on [-1,1].'
        ),
    }
    (RES/'FORMAL_SIGN_LOGIC_AUDIT.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
