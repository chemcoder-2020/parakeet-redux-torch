#!/usr/bin/env python3
"""Verify transcript parity and reproduction timings for a given encoder config.

Defaults to the CPU path (exact fp32 — works on macOS, Linux and Windows).
On an Apple-silicon Mac, --device mps adds the Metal-GPU configurations
(optionally with --dtype float16; fp16 numerics are approximate).

Runs all six reference clips, checks every transcript against the recorded
Photon outputs in testdata/ref_*.json, and reports warmed wall times.

    python scripts/verify_photon_speed.py                 # CPU, exact (default)
    python scripts/verify_photon_speed.py --threads 4     # CPU, other thread count
    python scripts/verify_photon_speed.py --device mps    # Apple GPU, fp16 encoder
    python scripts/verify_photon_speed.py --device mps --dtype float32   # Apple GPU, exact

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
# Warmed M1 reference walls per configuration (see the README "Speed" section).
REFERENCE_WALL = {
    ("cpu", "float32"): {"bcn_weather.mp3": 0.63, "mary_had_lamb.mp3": 0.86},
    ("mps", "float32"): {"bcn_weather.mp3": 0.37, "mary_had_lamb.mp3": 0.52},
    ("mps", "float16"): {"bcn_weather.mp3": 0.30, "mary_had_lamb.mp3": 0.43},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", help="encoder device: cpu (default), mps, cuda")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto",
                    help="encoder dtype (default: float32 on CPU, float16 on mps — "
                         "float16 numerics are approximate)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads (default 1; best on Apple Silicon — "
                         "sweep 1/2/4/8 on other CPUs)")
    ap.add_argument("--model-dir", default=str(REPO / "weights"))
    ap.add_argument("--runs", type=int, default=3, help="timed runs per file (default 3)")
    args = ap.parse_args()

    dtype = args.dtype
    if dtype == "auto":
        dtype = "float16" if args.device == "mps" else "float32"
    if dtype == "float16" and args.device == "cpu":
        print("float16 on CPU is slower and not supported here — use --device mps "
              "(or leave --dtype on auto).")
        return 1

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
    if dtype == "float16":
        model.encoder.half()
    transcriber = Transcriber(model, ParakeetTokenizer(Path(args.model_dir) / "tokenizer.json"),
                              config)

    print(f"torch {torch.__version__} | encoder on {args.device} in {dtype}, "
          f"{args.threads} thread(s) | runs per file: {args.runs}")
    transcriber.transcribe(str(REPO / "testdata" / FILES[0]))  # warm up
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
    ref_wall = REFERENCE_WALL.get((args.device, dtype))
    if ref_wall:
        for f, expected in ref_wall.items():
            print(f"reference wall (M1, warmed, {args.device}/{dtype}): "
                  f"{f} {expected:.2f} s, got {wall[f]:.3f} s")
    if dtype == "float16" and args.device == "mps":
        print("note: fp16 is approximate numerics; all six pass but borderline audio "
              "can differ. Use --dtype float32 for the strict-exact path.")
    return 0 if ok == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
