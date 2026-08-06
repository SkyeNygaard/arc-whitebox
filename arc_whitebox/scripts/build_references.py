"""Build cached Monte-Carlo references. Run in the background; it is the long pole."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from whest.nets import load_or_build_reference  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--samples", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    for s in args.seeds:
        t = time.time()
        mlp, ref = load_or_build_reference(
            args.width, args.depth, s, args.samples, ROOT, chunk=args.chunk
        )
        d = ref["Y_a"][-1] - ref["Y_b"][-1]
        noise = float((d * d).mean() / 4)
        print(
            f"[{args.width}x{args.depth} seed={s} m={args.samples:,}] "
            f"{time.time()-t:6.1f}s  ref-noise-var={noise:.3e}  "
            f"Ybar={ref['Y'][-1].mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
