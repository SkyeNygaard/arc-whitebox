from __future__ import annotations

"""Generate the exact Gram multiplicities used for an antipodal maximal real-MUB union.

The near-optimality theorem is conditional only on the stated MUB incidence
property.  Existence in dimension 256 is the classical Kerdock construction.
"""

import json
from pathlib import Path


def main(d: int = 256) -> None:
    if d != 256:
        raise ValueError('this proof package is specialized to d=256')
    bases = d // 2 + 1
    counts = {
        'inner_product_1': 1,
        'inner_product_minus_1': 1,
        'inner_product_0': 2*(d-1),
        'inner_product_plus_1_over_16': (bases-1)*d,
        'inner_product_minus_1_over_16': (bases-1)*d,
    }
    total = 2*d*bases
    assert sum(counts.values()) == total == 66048
    out = {
        'dimension': d,
        'number_of_bases': bases,
        'antipodal_nodes_per_basis': 2*d,
        'total_nodes': total,
        'abstract_input': (
            'An antipodal union of 129 pairwise mutually unbiased orthonormal bases in R^256. '
            'The proof of near-optimality uses only this incidence property; existence is supplied '
            'by the classical real Kerdock construction.'
        ),
        'per_fixed_node': counts,
        'derivation': [
            'A maximal real MUB family in dimension d=256 has d/2+1=129 bases.',
            'Including both signs gives 2d=512 nodes per basis and N=2d(d/2+1)=d(d+2)=66048.',
            'Within the fixed node\'s own basis: one self, one antipode, and 2(d-1)=510 signed orthogonal nodes.',
            'There are d/2=128 other bases. In each, the d antipodal pairs contribute exactly d positive and d negative inner products of magnitude 1/sqrt(d)=1/16.',
            'Thus each sign occurs 128d=32768 times.',
        ],
        'count_sum_check': sum(counts.values()),
        'passed': True,
    }
    base = Path(__file__).resolve().parent
    (base/'results/KERDOCK_MULTIPLICITY_PROOF.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
