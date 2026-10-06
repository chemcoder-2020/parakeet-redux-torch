"""Audio loading to 16 kHz mono float32 (via ffmpeg; soundfile as fallback)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Decode an audio file (any format ffmpeg knows) to mono float32."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [ffmpeg, "-nostdin", "-threads", "0", "-i", str(path),
               "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
               "-ar", str(sample_rate), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            wav = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
            return torch.from_numpy(wav)

    # fallback: soundfile (only if sample rate already matches)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise RuntimeError(
                f"soundfile got {sr} Hz; need {sample_rate} Hz and ffmpeg is unavailable")
        return torch.from_numpy(data.mean(axis=1))
    except ImportError:
        raise RuntimeError("ffmpeg not found on PATH and soundfile is not installed")


def write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono float32 tensor to a 16-bit PCM wav file."""
    import wave
    x = np.clip(waveform.numpy(), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
