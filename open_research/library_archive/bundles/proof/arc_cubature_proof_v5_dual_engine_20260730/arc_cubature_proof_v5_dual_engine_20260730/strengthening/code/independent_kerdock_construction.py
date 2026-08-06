#!/usr/bin/env python3
"""Independent exact certificate for the dimension-256 real Kerdock MUB incidence.

The certificate core uses only Python integers/lists/hashlib/json. NumPy is used
only in an optional final comparison with the archived production chirp asset;
none of the incidence checks depend on NumPy or the production implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

D = 256
FIELD = 128
MOD_REDUCTION = 0x03  # x^7 == x+1 for x^7+x+1


def gf_mul(a: int, b: int) -> int:
    out = 0
    x = a
    y = b
    while y:
        if y & 1:
            out ^= x
        y >>= 1
        carry = x & 0x40
        x = (x << 1) & 0x7F
        if carry:
            x ^= MOD_REDUCTION
    return out


def gf_square(a: int) -> int:
    return gf_mul(a, a)


def gf_pow(a: int, n: int) -> int:
    out = 1
    x = a
    while n:
        if n & 1:
            out = gf_mul(out, x)
        x = gf_square(x)
        n >>= 1
    return out


def gf_trace(a: int) -> int:
    acc = a
    x = a
    for _ in range(6):
        x = gf_square(x)
        acc ^= x
    if acc not in (0, 1):
        raise AssertionError((a, acc))
    return acc


def chirp(u: int) -> list[int]:
    values: list[int] = []
    for coordinate in range(D):
        x = coordinate & 0x7F
        xn = coordinate >> 7
        ux = gf_mul(u, x)
        q = gf_pow(ux, 3) ^ gf_pow(ux, 5) ^ gf_pow(ux, 9)
        bit = gf_trace(q) ^ (xn & gf_trace(ux))
        values.append(1 if bit == 0 else -1)
    return values


def fwht(values: list[int]) -> list[int]:
    out = values.copy()
    half = 1
    while half < len(out):
        block = 2 * half
        for start in range(0, len(out), block):
            for j in range(start, start + half):
                left = out[j]
                right = out[j + half]
                out[j] = left + right
                out[j + half] = left - right
        half *= 2
    return out


def hash_sign_rows(rows: list[list[int]]) -> str:
    # Canonical one-byte encoding: +1 -> 1, -1 -> 0.
    h = hashlib.sha256()
    for row in rows:
        h.update(bytes(1 if x == 1 else 0 for x in row))
    return h.hexdigest()


def exact_walsh_orthogonality() -> dict[str, int]:
    # H[a,x]=(-1)^parity(a&x). Check every ordered row pair by integer sum.
    max_off = 0
    min_diag = D
    max_diag = D
    for a in range(D):
        for b in range(a, D):
            dot = 0
            for x in range(D):
                dot += 1 if ((a ^ b) & x).bit_count() % 2 == 0 else -1
            if a == b:
                min_diag = min(min_diag, dot)
                max_diag = max(max_diag, dot)
                if dot != D:
                    raise AssertionError((a, dot))
            else:
                max_off = max(max_off, abs(dot))
                if dot != 0:
                    raise AssertionError((a, b, dot))
    return {"min_diagonal": min_diag, "max_diagonal": max_diag,
            "max_absolute_off_diagonal": max_off}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    nonzero = list(range(1, FIELD))
    inverse_failures: list[int] = []
    permutation_failures: list[int] = []
    for a in nonzero:
        inv = gf_pow(a, FIELD - 2)
        if gf_mul(a, inv) != 1:
            inverse_failures.append(a)
        if sorted(gf_mul(a, b) for b in nonzero) != nonzero:
            permutation_failures.append(a)

    rows = [chirp(u) for u in range(FIELD)]
    unique = len({tuple(row) for row in rows})
    spectrum_values: set[int] = set()
    bad_pairs: list[dict[str, object]] = []
    pair_count = 0
    for i in range(FIELD):
        for j in range(i + 1, FIELD):
            pair_count += 1
            product = [a * b for a, b in zip(rows[i], rows[j], strict=True)]
            spectrum = fwht(product)
            abs_values = sorted({abs(x) for x in spectrum})
            spectrum_values.update(abs_values)
            if abs_values != [16]:
                bad_pairs.append({"left": i, "right": j,
                                  "absolute_spectrum_values": abs_values})

    walsh = exact_walsh_orthogonality()

    asset = None
    if args.asset:
        import numpy as np
        z = np.load(args.asset)
        archived = z["chirps"]
        reconstructed = np.asarray(rows, dtype=np.float32)
        asset = {
            "file": args.asset.name,
            "archive_sha256": hashlib.sha256(args.asset.read_bytes()).hexdigest(),
            "shape": list(archived.shape),
            "dtype": str(archived.dtype),
            "raw_array_sha256": hashlib.sha256(archived.tobytes()).hexdigest(),
            "exact_match_to_independent_reconstruction": bool(
                archived.shape == reconstructed.shape
                and np.array_equal(archived, reconstructed)
            ),
        }

    bases = FIELD + 1
    lines = bases * D
    points = 2 * lines
    fixed_node_multiplicities = {
        "plus_one": 1,
        "minus_one": 1,
        "zero": 2 * (D - 1),
        "plus_one_over_16": FIELD * D,
        "minus_one_over_16": FIELD * D,
    }
    assert sum(fixed_node_multiplicities.values()) == points

    out = {
        "title": "Independent exact Kerdock/MUB incidence certificate",
        "implementation": {
            "certificate_core": "Python integer/list arithmetic only",
            "production_code_imported": False,
            "numpy_role": "optional archived-asset comparison only",
        },
        "field": {
            "description": "GF(2)[x]/(x^7+x+1)",
            "elements": FIELD,
            "inverse_failures": inverse_failures,
            "multiplication_permutation_failures": permutation_failures,
        },
        "walsh_matrix": walsh,
        "chirps": {
            "count": len(rows),
            "unique": unique,
            "canonical_sign_bit_sha256": hash_sign_rows(rows),
        },
        "mutual_unbiasedness": {
            "checked_unordered_chirp_basis_pairs": pair_count,
            "expected_pairs": FIELD * (FIELD - 1) // 2,
            "bad_pairs": bad_pairs,
            "pair_product_walsh_absolute_values": sorted(spectrum_values),
            "cross_basis_normalized_absolute_inner_product": "1/16",
            "coordinate_basis_cross_inner_product_absolute": "1/16",
        },
        "incidence": {
            "dimension": D,
            "signed_walsh_bases": FIELD,
            "coordinate_bases": 1,
            "total_bases": bases,
            "unoriented_lines": lines,
            "antipodal_points": points,
            "fixed_node_inner_product_multiplicities": fixed_node_multiplicities,
        },
        "archived_asset_comparison": asset,
        "passed": (
            not inverse_failures
            and not permutation_failures
            and walsh["max_absolute_off_diagonal"] == 0
            and unique == FIELD
            and pair_count == FIELD * (FIELD - 1) // 2
            and not bad_pairs
            and spectrum_values == {16}
            and points == 66048
            and (asset is None or asset["exact_match_to_independent_reconstruction"])
        ),
    }
    if not out["passed"]:
        raise SystemExit(json.dumps(out, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
