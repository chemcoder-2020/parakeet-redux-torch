"""Command-line transcription:

    python -m parakeet_redux_torch.cli audio.wav --model-dir weights
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audio import load_audio
from .loader import load_model
from .tokenizer import ParakeetTokenizer
from .transcribe import Transcriber


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="parakeet_redux_torch",
        description="Transcribe audio with moondream/parakeet-redux using only PyTorch "
                    "(no moondream package / Photon runtime).")
    ap.add_argument("audio", help="audio file (wav/mp3/flac/m4a/...)")
    ap.add_argument("--model-dir", default="weights",
                    help="directory containing model.safetensors + config.json + ternary.json + tokenizer.json")
    ap.add_argument("--device", default="cpu", help="torch device (cpu, mps, cuda)")
    ap.add_argument("--words", action="store_true", help="include word-level timestamps")
    ap.add_argument("--json", action="store_true", help="print full result as JSON")
    ap.add_argument("--verify-ternary", action="store_true",
                    help="verify ternary unpacking against recorded zero-fractions")
    args = ap.parse_args(argv)

    model, config, _ = load_model(args.model_dir, device=args.device,
                                  verify_ternary=args.verify_ternary)
    tokenizer = ParakeetTokenizer(Path(args.model_dir) / "tokenizer.json")
    transcriber = Transcriber(model, tokenizer, config)

    waveform = load_audio(args.audio)
    audio_seconds = waveform.shape[0] / 16000.0

    t0 = time.perf_counter()
    result = transcriber.transcribe_waveform(waveform, with_words=args.words)
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps({
            "text": result["text"],
            "words": result.get("words"),
            "audio_seconds": round(audio_seconds, 3),
            "wall_seconds": round(elapsed, 3),
            "rtf": round(audio_seconds / elapsed, 2) if elapsed > 0 else None,
            "mel_frames": result["mel_frames"],
            "encoder_frames": result["encoder_frames"],
        }, indent=2, ensure_ascii=False))
    else:
        print(result["text"])
        print(f"\n[audio {audio_seconds:.2f}s | wall {elapsed:.2f}s | "
              f"{audio_seconds / elapsed:.2f}x realtime]", file=__import__("sys").stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
