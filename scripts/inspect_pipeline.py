#!/usr/bin/env python3
"""Pipeline inspection: mel -> encoder -> greedy TDT decode, with stage prints."""
from __future__ import annotations

import json
import sys
import time

import torch

from parakeet_redux_torch.audio import load_audio
from parakeet_redux_torch.decode import greedy_tdt_decode, tokens_to_words
from parakeet_redux_torch.loader import load_model
from parakeet_redux_torch.mel import LogMelFrontend
from parakeet_redux_torch.model import BLANK_ID
from parakeet_redux_torch.tokenizer import ParakeetTokenizer


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "testdata/bcn_weather.mp3"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "weights"

    t0 = time.perf_counter()
    model, config, _ = load_model(model_dir, verify_ternary=False)
    print(f"[load] {time.perf_counter() - t0:.1f}s")

    tokenizer = ParakeetTokenizer(f"{model_dir}/tokenizer.json")
    frontend = LogMelFrontend()

    wav = load_audio(audio_path)
    print(f"[audio] {wav.shape[0]} samples = {wav.shape[0] / 16000:.2f}s")

    t0 = time.perf_counter()
    mel = frontend(wav)
    print(f"[mel] {tuple(mel.shape)} in {time.perf_counter() - t0:.1f}s "
          f"(expected frames {wav.shape[0] // 160}), mean {mel.mean():.4f} std {mel.std():.4f}")

    t0 = time.perf_counter()
    with torch.no_grad():
        enc = model.encoder(mel.unsqueeze(0))[0]
    print(f"[encoder] {tuple(enc.shape)} in {time.perf_counter() - t0:.1f}s, "
          f"norm {enc.norm(dim=-1).mean():.3f}")

    t0 = time.perf_counter()
    tokens = greedy_tdt_decode(model, enc, durations=tuple(config["durations"]),
                               max_symbols=config.get("max_symbols_per_step", 10))
    print(f"[decode] {len(tokens)} tokens in {time.perf_counter() - t0:.1f}s")

    text = tokenizer.decode([t[0] for t in tokens])
    print(f"[text] {text}")
    print(f"[first tokens] {tokens[:8]}")
    words = tokens_to_words(tokens, tokenizer)
    print(f"[first words] {json.dumps(words[:8], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
