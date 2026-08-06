#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_decisive_contraction.sh \
#     runs/pilot100/higher_moments_x1_results.json \
#     runs/pilot100/higher_moments_x1_coefnet.npz \
#     data/higher \
#     data/official_weights \
#     runs/pilot100/next_variance_eval.json

RESULTS_JSON=${1:?results json required}
MODEL=${2:?portable npz model required}
MOMENTS_DIR=${3:?higher-moment directory required}
WEIGHTS_DIR=${4:?official weight directory required}
OUTPUT=${5:-runs/pilot100/next_variance_eval.json}

mkdir -p "$(dirname "$OUTPUT")"

python eval_next_variance_x1.py \
  "$MODEL" \
  --results-json "$RESULTS_JSON" \
  --data-dir "$MOMENTS_DIR" \
  --weights-dir "$WEIGHTS_DIR" \
  --layers 4,8,12,16,20,24,28,30 \
  --diagonal-modes oracle,gaussian \
  --alpha auto \
  --alpha-metric relative_variance \
  --out "$OUTPUT"

python summarize_next_variance.py "$OUTPUT"
