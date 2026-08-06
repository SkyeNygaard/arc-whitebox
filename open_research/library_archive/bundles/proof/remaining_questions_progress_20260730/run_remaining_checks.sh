#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python verify_signed_near_optimality_certificate_blocktrace_order320.py
python verify_blocktrace_relaxation_sharpness.py
python analyze_target_free_support_capacity.py >/dev/null
python analyze_oracle_coefficient_drift.py >/dev/null
python build_finite_width_subcertificate_frontier.py >/dev/null
python summarize_comparison_cone_searches.py >/dev/null
