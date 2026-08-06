#!/usr/bin/env python3
"""Download a selected subset of the 1,000 per-MLP higher-moment files."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "keenanpepper/arc-whestbench-higher-moments-2026"

def parse_indices(spec: str) -> list[int]:
    out: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = map(int, piece.split("-", 1))
            out.update(range(start, end + 1))
        else:
            out.add(int(piece))
    if any(index < 0 or index > 999 for index in out):
        raise ValueError("indices must be in 0..999")
    return sorted(out)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", help="for example 0-699,700-849,850-999")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(parse_indices(args.indices), 1):
        name = f"full/mlp_{index:05d}.npz"
        cached = Path(hf_hub_download(REPO, name, repo_type="dataset",
                                     revision=args.revision))
        destination = args.output / cached.name
        if not destination.exists():
            shutil.copy2(cached, destination)
        print(f"[{position}] {destination}", flush=True)

if __name__ == "__main__":
    main()
