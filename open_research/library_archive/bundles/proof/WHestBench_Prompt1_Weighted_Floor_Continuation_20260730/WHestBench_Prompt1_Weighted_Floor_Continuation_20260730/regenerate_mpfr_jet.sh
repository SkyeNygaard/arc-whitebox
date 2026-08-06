#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cc -O3 mpfr_kernel_jet_511.c -l:libmpfr.so.6 -l:libgmp.so.10 -lm -o /tmp/mpfr_kernel_jet_511_prompt1_cont
/tmp/mpfr_kernel_jet_511_prompt1_cont > /tmp/MPFR_KERNEL_JET_511.prompt1_cont.json
cmp MPFR_KERNEL_JET_511.json /tmp/MPFR_KERNEL_JET_511.prompt1_cont.json
echo "MPFR JET BYTE-FOR-BYTE MATCH"
