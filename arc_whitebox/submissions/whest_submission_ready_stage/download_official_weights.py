#!/usr/bin/env python3
"""Download only the official WhestBench weight rows needed for an experiment.

For a pilot over global indices 0..99 this normally downloads only the first few
Parquet shards, not the full 1,000-network dataset. Output is one
``mlp_XXXXX.npy`` file per requested global index, each shape (32,256,256).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_indices(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def indices_from_results(path: Path, splits: list[str]) -> list[int]:
    result = json.loads(path.read_text())
    out: set[int] = set()
    for split in splits:
        key = f"{split}_ids"
        if key not in result:
            raise KeyError(f"{path} has no {key!r}")
        out.update(map(int, result[key]))
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--repo", default="aicrowd/arc-whestbench-public-2026")
    parser.add_argument("--revision", default="v1-phase1")
    parser.add_argument("--max-shards", type=int, default=28)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies first: pip install pyarrow huggingface_hub"
        ) from exc

    wanted: set[int] = set(parse_indices(args.indices))
    if args.results_json:
        wanted.update(indices_from_results(
            args.results_json,
            [s.strip() for s in args.splits.split(",") if s.strip()],
        ))
    if not wanted:
        raise SystemExit("No indices requested")

    args.output.mkdir(parents=True, exist_ok=True)
    remaining = set(wanted)
    global_start = 0
    manifest: dict[str, object] = {
        "repo": args.repo,
        "revision": args.revision,
        "requested": sorted(wanted),
        "files": {},
        "shards": [],
    }

    for shard in range(args.max_shards):
        if not remaining:
            break
        filename = f"data/full-{shard:05d}-of-00028.parquet"
        path = Path(hf_hub_download(
            args.repo, filename, revision=args.revision, repo_type="dataset"
        ))
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        shard_end = global_start + rows
        needed_here = sorted(i for i in remaining if global_start <= i < shard_end)
        manifest["shards"].append({
            "shard": shard,
            "global_start": global_start,
            "rows": rows,
            "selected": needed_here,
            "cache_path": str(path),
        })
        if needed_here:
            table = pq.read_table(path, columns=["weights"])
            column = table.column("weights")
            for global_index in needed_here:
                local_index = global_index - global_start
                weights = np.asarray(column[local_index].as_py(), dtype=np.float32)
                if weights.shape != (32, 256, 256):
                    raise ValueError(
                        f"global {global_index}: unexpected weights shape {weights.shape}"
                    )
                destination = args.output / f"mlp_{global_index:05d}.npy"
                np.save(destination, weights)
                manifest["files"][str(global_index)] = str(destination)
                remaining.remove(global_index)
                print(json.dumps({
                    "global_index": global_index,
                    "path": str(destination),
                    "shard": shard,
                }), flush=True)
        global_start = shard_end

    if remaining:
        raise RuntimeError(f"Could not find requested indices: {sorted(remaining)}")
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "downloaded": len(wanted),
        "output": str(args.output),
        "manifest": str(manifest_path),
    }))


if __name__ == "__main__":
    main()
