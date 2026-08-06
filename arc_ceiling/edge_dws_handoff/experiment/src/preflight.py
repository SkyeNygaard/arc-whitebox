from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    args = p.parse_args()
    b = load_bundle(args.data, args.manifest, args.splits)
    out = {
        "status": "ready",
        "examples": b.n,
        "label_dim": b.label_dim,
        "split_examples": {k: len(v) for k, v in b.splits.items()},
        "split_base_networks": {k: len({str(x) for x in b.arrays["base_network_id"][v]}) for k, v in b.splits.items()},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
