#!/usr/bin/env python3
"""Reference transcription with the official moondream package (Photon).

Used to cross-check `parakeet_redux_torch` outputs against the runtime that
ships with the model. Run inside a venv with `pip install --upgrade "moondream>=2.4.1"`.
"""
from __future__ import annotations

import argparse
import json
import time

import moondream as md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="moondream/parakeet-redux")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timestamps", default="word", choices=["none", "segment", "word"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    with md.photon(args.model, device=args.device) as speech:
        result = speech.transcribe(audio=args.audio, timestamps=args.timestamps)
    elapsed = time.perf_counter() - t0

    out = {"text": result["text"], "wall_seconds": round(elapsed, 3)}
    if "segments" in result and result["segments"]:
        out["segments"] = result["segments"]
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
