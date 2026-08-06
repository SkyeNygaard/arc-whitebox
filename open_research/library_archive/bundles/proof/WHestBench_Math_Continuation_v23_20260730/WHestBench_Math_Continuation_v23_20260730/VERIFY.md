# Verification

Run each command from the unpacked package root. The two degree-320 exact replays are deliberately separate fresh processes because each constructs a large rational Gegenbauer basis.

```bash
python verify.py inertia_exact_replay
python verify.py v21_degree280
python verify.py shared_profile_and_sturm
python verify.py gaussian_suffix_nonexpansivity
python verify.py gaussian_crossing_formula
python verify.py finite_width_monotonicity_no_go
```

All six commands were run successfully from a clean unpacked copy for this release. `combined_verification_v23.json` records the release check. This verifies exact rational witness arithmetic against the stored directed kernel interval endpoints; it does not independently regenerate those endpoints.
