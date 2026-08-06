WHestBench Prompt 2 finite-width certificate reproduction

The frozen scripts use /mnt/data as their artifact root. Extract all files in this
archive directly into /mnt/data, preserving the flat archive layout, then run:

  cd /mnt/data
  python verify_prompt2_markov_29state_degree28.py

Required runtime: Python 3.11+ and NumPy. The proof-critical arithmetic in the
final verifier is fractions.Fraction plus directed decimal flooring; NumPy is
used by imported assembler diagnostics, not by the final domination check.

Expected normalized floor lower endpoint:
  1.92502416170783524100205877712113374289784088032e-8
