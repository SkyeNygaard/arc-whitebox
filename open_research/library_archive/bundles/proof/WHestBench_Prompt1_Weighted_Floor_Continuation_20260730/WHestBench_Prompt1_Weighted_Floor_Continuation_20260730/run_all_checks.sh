#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./regenerate_mpfr_jet.sh
./run_checks.sh
