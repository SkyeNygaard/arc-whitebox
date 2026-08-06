# Exact baseline package missing

The accessible archive does not contain the exact final package tied to the reported exposed Mini-100 result.

## Expected package names referenced in historical notes

- `whestbench_final_129basis_20260730.tar.gz`
- `whestbench_final_129basis_depth4_20260730.tar.gz`

## Expected contents

Historical notes say the package contained five files, including:

- `estimator.py`
- `fast_matmul.py`
- `kerdock_mub5_seed3.npz`
- `.whestignore`
- package metadata/README

## Hashes that are safe to state

The Kerdock asset appeared repeatedly with SHA-256:

`58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad`

## Hashes that must not be substituted

The accessible `production_partial_tree_source` package has a mismatched estimator manifest. Its `fast_matmul.py` hash and estimator hash do not authenticate the exact final 129-basis package.

## Authentication requirements

A recovered archive should be accepted only after:

1. archive SHA-256 matches an independently preserved record or multiple historical copies;
2. package validates under the stated `whestbench` and FlopScope versions;
3. the 100-network exposed run reproduces the reported aggregate and per-network JSON;
4. node asset hash and package file hashes are recorded;
5. output is deterministic across fresh subprocesses;
6. package identity is distinct from the broken root estimator.
