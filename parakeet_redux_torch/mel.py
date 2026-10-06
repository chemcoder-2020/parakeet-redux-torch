"""Log-mel frontend for Parakeet (16 kHz, 128 mels, per-feature normalized).

This mirrors NVIDIA NeMo's `AudioToMelSpectrogramPreprocessor` settings for
parakeet-tdt-0.6b-v3 (as ported by the HuggingFace `ParakeetFeatureExtractor`,
which is validated against NeMo):

  - hann window, 400 samples (symmetric), n_fft 512, hop 160
  - STFT with center padding (zeros) of n_fft // 2 on both sides
  - power spectrum, then librosa-style "slaney" mel filterbank (128 mels)
  - log(mel + 2**-24)
  - per-feature normalization: (x - mean) / (std + 1e-5), unbiased std
  - the number of frames kept is floor(num_samples / hop_length)
"""
from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
LOG_ZERO_GUARD = 2 ** -24
EPSILON = 1e-5


# ---------------------------------------------------------------------------
# librosa-compatible slaney mel filterbank (computed, not imported)
# ---------------------------------------------------------------------------

def _hz_to_mel_slaney(freqs: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3
    with np.errstate(divide="ignore"):  # log(0) is computed then discarded below
        mels = (freqs - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(freqs >= min_log_hz,
                        min_log_mel + np.log(freqs / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(mels >= min_log_mel,
                    min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def _mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    min_mel = _hz_to_mel_slaney(np.array([fmin], dtype=np.float64)).item()
    max_mel = _hz_to_mel_slaney(np.array([fmax], dtype=np.float64)).item()
    mels = np.linspace(min_mel, max_mel, n_mels, dtype=np.float64)
    return _mel_to_hz_slaney(mels)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                   n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-normalized mel filterbank, shape [n_mels, n_fft // 2 + 1]."""
    fmax = sr / 2.0
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, fmax, n_freqs, dtype=np.float64)
    mel_f = _mel_frequencies(n_mels + 2, 0.0, fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)  # [n_mels + 2, n_freqs]
    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    # slaney-style area normalization
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------------

class LogMelFrontend(torch.nn.Module):
    """Stateless 16 kHz log-mel frontend. Input: [num_samples] float32."""

    def __init__(self, sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 win_length: int = WIN_LENGTH, hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS, preemphasis: float = 0.97):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis = preemphasis
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False),
            persistent=False)
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(mel_filterbank(sr, n_fft, n_mels)),
            persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: float32 [num_samples] -> features [num_frames, n_mels]."""
        x = waveform.to(torch.float32)
        if x.dim() == 2:
            x = x.mean(0)

        if self.preemphasis is not None:
            x = torch.cat([x[:1], x[1:] - self.preemphasis * x[:-1]], dim=0)

        stft = torch.stft(
            x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, return_complex=True, pad_mode="constant",
        )
        magnitudes = stft.real.pow(2) + stft.imag.pow(2)          # [freqs, frames]
        mel_spec = self.mel_filters @ magnitudes                  # [n_mels, frames]
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD)

        # NeMo/HF keep floor(num_samples / hop_length) frames.
        num_frames = x.shape[0] // self.hop_length
        mel_spec = mel_spec[:, :num_frames]

        # per-feature normalization (unbiased std, as in NeMo)
        mean = mel_spec.mean(dim=1, keepdim=True)
        std = mel_spec.std(dim=1, unbiased=True, keepdim=True)
        mel_spec = (mel_spec - mean) / (std + EPSILON)

        return mel_spec.transpose(0, 1).contiguous()              # [frames, n_mels]


def subsampled_lengths(num_frames: torch.Tensor, subsampling_factor: int = 8,
                       subsampling_conv_kernel_size: int = 3,
                       subsampling_conv_stride: int = 2) -> torch.Tensor:
    """Encoder frame count after the conv subsampler (per NeMo's formula)."""
    import math
    num_layers = int(math.log2(subsampling_factor))
    add_pad = (subsampling_conv_kernel_size - 1) // 2 * 2 - subsampling_conv_kernel_size
    lengths = num_frames.to(torch.float64)
    for _ in range(num_layers):
        lengths = torch.floor((lengths + add_pad) / subsampling_conv_stride) + 1.0
    return lengths.to(torch.int64)
