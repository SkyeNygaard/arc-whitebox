#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/unblocked_research/run_official_on_skyes_mac.sh" "${1:-arc_whitebox}" "$HERE/T0_official_grader_instrumentation_20260729"
