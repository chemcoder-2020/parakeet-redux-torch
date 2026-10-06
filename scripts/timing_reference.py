#!/usr/bin/env python3
"""Warmed-up timing of the official moondream runtime (Photon) for comparison.

Run inside the reference venv:  .venv-ref/bin/python scripts/timing_reference.py
"""
from __future__ import annotations

import time

import moondream as md

FILES = ["testdata/bcn_weather.mp3", "testdata/mary_had_lamb.mp3"]


def main() -> None:
    with md.photon("moondream/parakeet-redux", device="cpu") as speech:
        # warm-up (load kernels / compile)
        speech.transcribe(audio=FILES[0], timestamps="none")
        for f in FILES:
            t0 = time.perf_counter()
            r = speech.transcribe(audio=f, timestamps="none")
            dt = time.perf_counter() - t0
            print(f"{f}: {dt:.3f}s -> {r['text'][:60]!r}")


if __name__ == "__main__":
    main()
