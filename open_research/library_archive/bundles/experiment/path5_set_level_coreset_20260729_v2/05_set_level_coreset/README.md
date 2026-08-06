# Path 5 experiment bundle

Primary report: `PATH5_SET_LEVEL_CORESET_REPORT.md`  
Machine-readable summary: `PATH5_SET_LEVEL_CORESET_RESULTS.json`

The bundle is an offline exact-geometry research artifact. It is not a production estimator. Oracle calibration is used only to label fixed supports under the preregistered positive/bounded-weight constraints.

Key reproduction entry points:

- `code/oracle_library_sweep.py`: full fixed-library support sweep.
- `code/generate_top8_rank_dataset.py`: grouped candidate-level dataset generation.
- `code/top8_sketch_selector.py`: fixed eight-support direct-sketch selector.
- `code/train_top8_ranker_light.py`: grouped learned ranker screen.
- `code/aggregate_path5_results.py`: canonical result aggregation.
