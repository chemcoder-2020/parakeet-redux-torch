#!/usr/bin/env python3
"""Independently verify the in-repo mel frontend against librosa.

librosa is what NeMo uses (and the HuggingFace `ParakeetFeatureExtractor` calls
it directly for the filterbank). This script compares:
  1. our computed slaney mel filterbank vs librosa.filters.mel
  2. the full log-mel frontend vs a line-by-line reimplementation of the HF
     extractor's numerics on the same waveform

Run inside an env that has librosa (dev/verification only - the package itself
does not need it):  .venv-ref/bin/python scripts/verify_mel_vs_librosa.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parakeet_redux_torch.mel import (LOG_ZERO_GUARD, LogMelFrontend,
                                      mel_filterbank)

import librosa  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def hf_style_features(waveform: torch.Tensor, mel_filters: torch.Tensor,
                      n_fft: int = 512, win_length: int = 400, hop_length: int = 160,
                      preemphasis: float = 0.97) -> torch.Tensor:
    """Line-by-line port of ParakeetFeatureExtractor._torch_extract_fbank_features +
    normalization, for the single-utterance case (no padding)."""
    x = waveform.to(torch.float32)
    x = torch.cat([x[:1], x[1:] - preemphasis * x[:-1]], dim=0)
    window = torch.hann_window(win_length, periodic=False)
    stft = torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length,
                      window=window, return_complex=True, pad_mode="constant")
    magnitudes = torch.view_as_real(stft)
    magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)

    mel_spec = mel_filters @ magnitudes
    mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD)

    num_frames = x.shape[0] // hop_length
    mel_spec = mel_spec[:, :num_frames]
    mean = mel_spec.mean(dim=1, keepdim=True)
    std = mel_spec.std(dim=1, unbiased=True, keepdim=True)
    mel_spec = (mel_spec - mean) / (std + 1e-5)
    return mel_spec.transpose(0, 1).contiguous()


def main() -> None:
    print("== 1. filterbank ==")
    ref = np.asarray(librosa.filters.mel(sr=16000, n_fft=512, n_mels=128,
                                         fmin=0.0, fmax=8000, norm="slaney"))
    mine = mel_filterbank()
    print(f"shapes: mine {mine.shape}, librosa {ref.shape}, dtypes {mine.dtype}/{ref.dtype}")
    diff = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
    print(f"max abs diff: {diff.max():.3e}   mean abs diff: {diff.mean():.3e}")

    print("\n== 2. full frontend on test wav ==")
    wav_path = ROOT / "testdata/librispeech_mr_quilter.wav"
    with wave.open(str(wav_path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    x = torch.from_numpy(pcm)

    mine_mel = LogMelFrontend()(x)
    ref_mel = hf_style_features(x, torch.from_numpy(np.asarray(ref)).to(torch.float32))
    print(f"shapes: mine {tuple(mine_mel.shape)}, ref {tuple(ref_mel.shape)}")
    if mine_mel.shape == ref_mel.shape:
        d = (mine_mel - ref_mel).abs()
        print(f"max abs diff: {d.max():.3e}   mean abs diff: {d.mean():.3e}")
    else:
        print("SHAPE MISMATCH")


if __name__ == "__main__":
    main()
