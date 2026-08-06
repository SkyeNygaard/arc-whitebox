#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cc -O3 mpfr_kernel_jet_511.c -l:libmpfr.so.6 -l:libgmp.so.10 -lm -o /tmp/mpfr_kernel_jet_511
/tmp/mpfr_kernel_jet_511 > /tmp/MPFR_KERNEL_JET_511.regenerated.json
cmp MPFR_KERNEL_JET_511.json /tmp/MPFR_KERNEL_JET_511.regenerated.json
python verify_prompt1_degree47_dual.py
