#!/usr/bin/env python3
"""Run the pure-PyTorch implementation on the test files, print a comparison
table against the recorded moondream (Photon) reference outputs, and save our
results to testdata/ours_<name>.json.

Usage:  .venv/bin/python scripts/compare_with_reference.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parakeet_redux_torch.audio import load_audio
from parakeet_redux_torch.loader import load_model
from parakeet_redux_torch.tokenizer import ParakeetTokenizer
from parakeet_redux_torch.transcribe import Transcriber

ROOT = Path(__file__).resolve().parent.parent
TESTDATA = ROOT / "testdata"

FILES = [
    ("bcn_weather", "bcn_weather.mp3"),
    ("librispeech_mr_quilter", "librispeech_mr_quilter.wav"),
    ("fleur_es_sample", "fleur_es_sample.wav"),
    ("en-Alice_woman", "en-Alice_woman.wav"),
    ("mary_had_lamb", "mary_had_lamb.mp3"),
    ("f2641_0_throatclearing", "f2641_0_throatclearing.wav"),
]


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).split()


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def main() -> None:
    t0 = time.perf_counter()
    model, config, _ = load_model(ROOT / "weights")
    print(f"[load] {time.perf_counter() - t0:.1f}s\n")
    tokenizer = ParakeetTokenizer(ROOT / "weights/tokenizer.json")
    tr = Transcriber(model, tokenizer, config)

    print(f"{'file':<26} {'audio':>6} {'wall':>6} {'xRT':>5} {'WER':>6}  match")
    print("-" * 78)
    total_edits, total_words = 0, 0
    for name, filename in FILES:
        path = TESTDATA / filename
        if not path.exists():
            print(f"{name:<26} MISSING {filename}")
            continue
        wav = load_audio(path)
        seconds = wav.shape[0] / 16000
        t0 = time.perf_counter()
        res = tr.transcribe_waveform(wav, with_words=True)
        wall = time.perf_counter() - t0

        (TESTDATA / f"ours_{name}.json").write_text(json.dumps({
            "text": res["text"], "words": res["words"],
            "wall_seconds": round(wall, 3), "audio_seconds": round(seconds, 3),
            "mel_frames": res["mel_frames"], "encoder_frames": res["encoder_frames"],
        }, indent=2, ensure_ascii=False))

        ref_file = TESTDATA / f"ref_{name}.json"
        row = f"{name:<26} {seconds:6.2f} {wall:6.2f} {seconds / wall:5.2f}"
        if ref_file.exists():
            ref = json.loads(ref_file.read_text())
            r, h = norm_words(ref["text"]), norm_words(res["text"])
            edits = edit_distance(r, h)
            total_edits += edits
            total_words += len(r)
            wer = edits / max(1, len(r))
            exact = "IDENTICAL" if ref["text"].strip() == res["text"].strip() else f"differs"
            row += f" {wer:6.1%}  {exact}"
            print(row)
            if wer > 0 or ref["text"].strip() != res["text"].strip():
                print(f"   ref:  {ref['text']}")
                print(f"   ours: {res['text']}")
        else:
            print(row + "      (no reference file)")
            print(f"   ours: {res['text']}")

    if total_words:
        print("-" * 78)
        print(f"aggregate word-level WER over {total_words} reference words: "
              f"{total_edits / total_words:.2%}")


if __name__ == "__main__":
    main()
