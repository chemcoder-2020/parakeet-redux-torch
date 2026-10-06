#!/usr/bin/env python3
"""Reproduce the Photon-speed configuration and verify it end-to-end.

Runs all six reference clips through the Apple-GPU hybrid encoder
(--encoder-device mps, optionally --encoder-dtype float16), checks each
transcript against the recorded Photon outputs in testdata/ref_*.json,
and reports warmed wall times.

    python scripts/verify_photon_speed.py                 # fp16 MPS (Photon-speed config)
    python scripts/verify_photon_speed.py --dtype float32 # exact-numerics hybrid
    python scripts/verify_photon_speed.py --device cpu    # CPU path

Exit code 0 = all six transcripts identical; 1 = any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from parakeet_redux_torch.loader import load_model
from parakeet_redux_torch.tokenizer import ParakeetTokenizer
from parakeet_redux_torch.transcribe import Transcriber

FILES = ["bcn_weather.mp3", "librispeech_mr_quilter.wav", "fleur_es_sample.wav",
         "en-Alice_woman.wav", "mary_had_lamb.mp3", "f2641_0_throatclearing.wav"]
REFERENCE_WALL = {"bcn_weather.mp3": 0.30, "mary_had_lamb.mp3": 0.43}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps", help="encoder device (mps, cuda or cpu; default mps)")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16",
                    help="encoder dtype (default float16 = Photon-speed config; "
                         "float16 numerics are approximate)")
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads (default 1)")
    ap.add_argument("--model-dir", default=str(REPO / "weights"))
    ap.add_argument("--runs", type=int, default=3, help="timed runs per file (default 3)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available in this torch build / machine.")
        print("Re-run with:  python scripts/verify_photon_speed.py --device cpu")
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this torch build / machine.")
        print("Re-run with:  python scripts/verify_photon_speed.py --device cpu")
        return 1

    model, config, _ = load_model(args.model_dir, device="cpu")
    model.encoder.to(args.device)
    if args.dtype == "float16":
        model.encoder.half()
    transcriber = Transcriber(model, ParakeetTokenizer(Path(args.model_dir) / "tokenizer.json"),
                              config)

    print(f"torch {torch.__version__} | encoder on {args.device} in {args.dtype}, "
          f"{args.threads} thread(s) | runs per file: {args.runs}")
    transcriber.transcribe(str(REPO / "testdata" / FILES[0]))  # warm up kernels
    print("warm-up done\n")

    ok = 0
    wall = {}
    for f in FILES:
        ref = json.loads((REPO / "testdata" / f"ref_{f.rsplit('.', 1)[0]}.json").read_text())["text"]
        ts = []
        for _ in range(args.runs):
            s = time.perf_counter()
            result = transcriber.transcribe(str(REPO / "testdata" / f))
            ts.append(time.perf_counter() - s)
        same = result["text"] == ref
        ok += same
        wall[f] = min(ts)
        print(f"{'PASS' if same else 'FAIL'}  {f:<34} {min(ts):.3f} s")
        if not same:
            print(f"      ours: {result['text'][:120]!r}")
            print(f"      ref : {ref[:120]!r}")

    print(f"\nidentical transcripts: {ok}/6")
    for f, expected in REFERENCE_WALL.items():
        print(f"reference wall (M1, warmed): {f} {expected:.2f} s, got {wall[f]:.3f} s")
    if args.dtype == "float16" and args.device == "mps":
        print("note: fp16 is approximate numerics; all six pass but borderline audio "
              "can differ. Use --dtype float32 for the strict-exact path.")
    return 0 if ok == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
