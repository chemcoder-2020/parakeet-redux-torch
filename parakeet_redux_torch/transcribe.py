"""End-to-end transcription: waveform -> text (+ word timestamps)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .audio import load_audio
from .decode import greedy_tdt_decode, tokens_to_words
from .mel import LogMelFrontend
from .tokenizer import ParakeetTokenizer


class Transcriber:
    """Bundles model + frontend + tokenizer for repeated transcription."""

    def __init__(self, model, tokenizer: ParakeetTokenizer, config: dict | None = None,
                 max_encoder_frames: int = 5000):
        self.model = model
        self.tokenizer = tokenizer
        self.frontend = LogMelFrontend()
        self.durations = tuple((config or {}).get("durations", [0, 1, 2, 3, 4]))
        self.max_symbols = (config or {}).get("max_symbols_per_step", 10)
        self.seconds_per_frame = (config or {}).get("seconds_per_frame", 0.08)
        # max_position_embeddings in the encoder config; beyond this the model
        # was never trained (and the official runtime chunks at VAD pauses).
        self.max_encoder_frames = max_encoder_frames

    @torch.no_grad()
    def transcribe_waveform(self, waveform: torch.Tensor,
                            with_words: bool = False) -> dict:
        mel = self.frontend(waveform)                 # [Tm, 128]
        enc = self.model.encoder(mel.unsqueeze(0))[0]  # [Te, 1024]
        if self.max_encoder_frames is not None and enc.shape[0] > self.max_encoder_frames:
            raise ValueError(
                f"audio needs {enc.shape[0]} encoder frames, over the "
                f"{self.max_encoder_frames}-frame limit (about "
                f"{self.max_encoder_frames * self.seconds_per_frame:.0f}s per pass). "
                "The official runtime splits long recordings at VAD pauses; "
                "chunk the audio yourself and transcribe the pieces."
            )
        tokens = greedy_tdt_decode(self.model, enc, durations=self.durations,
                                   max_symbols=self.max_symbols)
        text = self.tokenizer.decode([t[0] for t in tokens])
        result = {
            "text": text,
            "tokens": tokens,
            "encoder_frames": int(enc.shape[0]),
            "mel_frames": int(mel.shape[0]),
        }
        if with_words:
            result["words"] = tokens_to_words(tokens, self.tokenizer)
        return result

    def transcribe(self, audio: str | Path | torch.Tensor, with_words: bool = False) -> dict:
        waveform = audio if isinstance(audio, torch.Tensor) else load_audio(audio)
        return self.transcribe_waveform(waveform, with_words=with_words)
