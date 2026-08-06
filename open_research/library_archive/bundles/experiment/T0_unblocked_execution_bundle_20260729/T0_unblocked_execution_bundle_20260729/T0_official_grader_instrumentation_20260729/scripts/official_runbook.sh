#!/usr/bin/env bash
set -euo pipefail
: "${WHEST:=whest}"
: "${DATASET:?Set DATASET to the official Phase-1 Mini dataset directory}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/official_results"; WORK="$OUT/unpacked"
mkdir -p "$OUT" "$WORK"
packages=(production_baseline A42 A43 A43_delta64 A43_basis096 A43_basis064 A43_basis032)
for name in "${packages[@]}"; do
  "$WHEST" validate-package "$ROOT/packages/${name}.tar.gz"
  rm -rf "$WORK/$name"; mkdir -p "$WORK/$name"
  tar -xzf "$ROOT/packages/${name}.tar.gz" -C "$WORK/$name"
  "$WHEST" run --estimator "$WORK/$name/estimator.py" --dataset "$DATASET" --split mini --runner subprocess > "$OUT/${name}.json"
done
