# WHestBench non-overlap continuation v25

Run the fast theorem verifier:

```bash
python code/verify_posterior_score_decomposition.py
```

Run the heavier exact-MUB two-plane diagnostic:

```bash
python code/run_plane_capture_diagnostic.py
```

The second command reconstructs all 129 real mutually unbiased bases from the vendored Kerdock chirp asset and may take several minutes.
