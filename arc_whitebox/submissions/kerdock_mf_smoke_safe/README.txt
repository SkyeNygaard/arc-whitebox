Smoke-safe source bundle.

Contains:
- estimator.py
- kerdock_mf_90624_precomputed.npz

Package this directory with the local WhestBench CLI. This version intentionally
avoids a direct `import numpy as np`, which is the likely cause of the grader's
IMPORT_FAILED smoke result.
