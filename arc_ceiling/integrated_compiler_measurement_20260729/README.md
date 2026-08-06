# Integrated Kerdock compiler measurement package

Status: **externally blocked in this sandbox**. The package contains a tested NumPy compiler core and an official-style subprocess runner, but the official Mini-100 parquet data and FlopScope/WHestBench runtime were unavailable.

## Structural validation

```bash
python -m pytest -q tests
python src/smoke_measurement.py
```

## Decisive paired run

```bash
python src/run_paired.py \
  --data /absolute/path/to/official_phase1_mini/data \
  --asset assets/kerdock_mub5_seed3.npz \
  --indices $(seq 0 99) \
  --outdir results/official_paired
```

Read `report.md`, `LOCAL_HANDOFF.md`, and `MISSING_ASSETS.json` before running.
