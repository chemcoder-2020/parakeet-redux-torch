#!/usr/bin/env python3
"""Download the moondream/parakeet-redux model files from HuggingFace.

    python scripts/download_model.py [--dir weights]
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/moondream/parakeet-redux/resolve/main/"
FILES = ["config.json", "ternary.json", "tokenizer.json", "model.safetensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="weights")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out_dir / name
        if dst.exists():
            print(f"skip  {name} (exists, {dst.stat().st_size:,} bytes)")
            continue
        print(f"fetch {name} ...")
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "parakeet-redux-torch"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"  -> {dst} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
