"""Pure-PyTorch implementation of the Parakeet Redux model architecture
(FastConformer encoder + TDT decoder/joint), mirroring NVIDIA NeMo's
`parakeet-tdt-0.6b-v3` as implemented in the NeMo, parakeet-mlx and
HuggingFace transformers ports.

Every module's parameter names match the `moondream/parakeet-redux`
safetensors keys exactly (the repo uses the HuggingFace conversion naming),
so the packed weights can be loaded with a strict state_dict load.

No dependency on onnxruntime/nemo/mlx/moondream - only torch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK_ID = 8192
NUM_EXTRA_OUTPUTS = 5  # durations (0, 1, 2, 3, 4)
DURATIONS = (0, 1, 2, 3, 4)
SECONDS_PER_FRAME = 0.08


# ---------------------------------------------------------------------------
# Subsampling (8x striding depthwise-separable conv, NeMo "dw_striding")
# ---------------------------------------------------------------------------

class DwStridingSubsampling(nn.Module):
    def __init__(self, hidden_size: int = 1024, feat_in: int = 128,
                 conv_channels: int = 256, kernel_size: int = 3,
                 stride: int = 2, num_subsampling_layers: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.subsampling_factor = stride ** num_subsampling_layers

        final_freq_dim = feat_in
        for _ in range(num_subsampling_layers):
            final_freq_dim = math.floor((final_freq_dim + 2 * padding - kernel_size) / stride) + 1

        layers: list[nn.Module] = [
            nn.Conv2d(1, conv_channels, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        ]
        for _ in range(num_subsampling_layers - 1):
            layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size,
                                    stride=stride, padding=padding, groups=conv_channels))
            layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(conv_channels * final_freq_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F] -> [B, T // 8, hidden_size]."""
        x = x.unsqueeze(1)                    # [B, 1, T, F] (H=T, W=F)
        x = self.layers(x)                    # [B, C, T', F']
        x = x.permute(0, 2, 1, 3)             # [B, T', C, F']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T', C * F']
        return self.linear(x)


# ---------------------------------------------------------------------------
# Relative positional encoding (sinusoidal, Shaw-style, as in NeMo)
# ---------------------------------------------------------------------------

class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pos_emb of shape [1, 2T-1, D] for a sequence of length T."""
        t = x.shape[1]
        # float64 where available; MPS has no float64, and float32 there is
        # well within the transcript-relevant precision.
        dtype = torch.float32 if x.device.type == "mps" else torch.float64
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.d_model, 2, dtype=dtype,
                                                    device=x.device) / self.d_model))
        pos = torch.arange(t - 1, -t, -1, dtype=dtype, device=x.device)
        freqs = pos[:, None] * inv_freq[None, :]
        sin = freqs.sin()
        cos = freqs.cos()
        pos_emb = torch.stack([sin, cos], dim=-1).reshape(2 * t - 1, self.d_model)
        return pos_emb.to(dtype=x.dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# Conformer pieces
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int = 1024, d_ff: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int = 1024, kernel_size: int = 9):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size,
                                        padding=self.padding, groups=channels, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]."""
        x = x.transpose(1, 2)                 # [B, C, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


__all__ = []  # populated at the bottom


def _load_conformer_attention() -> type[nn.Module]:
    class RelPosAttention(nn.Module):
        """Multi-head attention with Shaw-style relative positional encoding."""

        def __init__(self, d_model: int = 1024, n_heads: int = 8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.scale = self.head_dim ** -0.5
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            self.relative_k_proj = nn.Linear(d_model, d_model, bias=False)
            self.bias_u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
            self.bias_v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        @staticmethod
        def _rel_shift(x: torch.Tensor) -> torch.Tensor:
            """[B, H, T, P] -> [B, H, T, P], the standard relative shift."""
            b, h, t, p = x.shape
            x = F.pad(x, (1, 0))              # pad last dim (left)
            x = x.view(b, h, p + 1, t)
            x = x[:, :, 1:, :]
            return x.reshape(b, h, t, p)

        def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
            b, t, _ = x.shape
            h, hd = self.n_heads, self.head_dim

            q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
            k = self.k_proj(x).view(b, t, h, hd).transpose(1, 2)
            v = self.v_proj(x).view(b, t, h, hd).transpose(1, 2)
            p = self.relative_k_proj(pos_emb).view(b, -1, h, hd).transpose(1, 2)

            q_u = q + self.bias_u.view(1, h, 1, hd)
            q_v = q + self.bias_v.view(1, h, 1, hd)

            # term (b)+(d): query-with-v against (projected) relative positions
            bd = q_v @ p.transpose(-1, -2)    # [B, H, T, P]
            bd = self._rel_shift(bd)[..., :t] * self.scale

            # terms (a)+(c): query-with-u against content keys
            ac = q_u @ k.transpose(-1, -2)    # [B, H, T, T]
            scores = ac * self.scale + bd

            attn = F.softmax(scores, dim=-1)
            out = attn @ v                    # [B, H, T, hd]
            out = out.transpose(1, 2).reshape(b, t, h * hd)
            return self.o_proj(out)

    return RelPosAttention


RelPosAttention = _load_conformer_attention()


class ConformerLayer(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 8, d_ff: int = 4096,
                 conv_kernel_size: int = 9):
        super().__init__()
        self.feed_forward1 = FeedForward(d_model, d_ff)
        self.self_attn = RelPosAttention(d_model, n_heads)
        self.conv = ConvolutionModule(d_model, conv_kernel_size)
        self.feed_forward2 = FeedForward(d_model, d_ff)
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, num_layers: int = 24, d_model: int = 1024, n_heads: int = 8,
                 d_ff: int = 4096, num_mel_bins: int = 128, conv_kernel_size: int = 9,
                 subsampling_conv_channels: int = 256, subsampling_factor: int = 8):
        super().__init__()
        self.pos_enc = RelPositionalEncoding(d_model)
        self.subsampling = DwStridingSubsampling(
            hidden_size=d_model, feat_in=num_mel_bins,
            conv_channels=subsampling_conv_channels,
            num_subsampling_layers=int(math.log2(subsampling_factor)))
        self.layers = nn.ModuleList(
            ConformerLayer(d_model, n_heads, d_ff, conv_kernel_size)
            for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T_mel, num_mel_bins] -> [B, T_enc, d_model]."""
        x = self.subsampling(x)
        pos_emb = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x


# ---------------------------------------------------------------------------
# TDT prediction network / joint network
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """LSTM prediction network ("decoder" in the checkpoint)."""

    def __init__(self, vocab_size: int = BLANK_ID + 1, hidden_size: int = 640,
                 num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.decoder_projector = nn.Linear(hidden_size, hidden_size)

    def step(self, token: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        """One decoding step. token: [B] long; state: (h, c) each [L, B, H].

        Returns (projected_output [B, H], new_state).
        """
        emb = self.embedding(token).unsqueeze(1)          # [B, 1, H]
        out, new_state = self.lstm(emb, state)
        return self.decoder_projector(out[:, 0]), new_state


class JointNetwork(nn.Module):
    """Only the terminal head lives here; the encoder/decoder projections are
    top-level modules (`encoder_projector`, `decoder.decoder_projector`) to
    mirror the checkpoint naming."""

    def __init__(self, hidden_size: int = 640, num_outputs: int = BLANK_ID + 1 + NUM_EXTRA_OUTPUTS):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_outputs)

    def forward(self, enc_proj: torch.Tensor, dec_proj: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(enc_proj + dec_proj))


# ---------------------------------------------------------------------------
# VAD head (experimental; segmenter used by Photon for long audio)
# ---------------------------------------------------------------------------

class VADHead(nn.Module):
    """Small voice-activity head over subsampler frames (80 ms each).

    NOTE: exported by the model repo but its activation functions are not
    documented; we use ReLU between convolutions and a sigmoid output, which
    matches the exported structure (proj 1024->128 k1, ctx 128->128 k5, out
    128->1 k1).
    """

    def __init__(self, d_model: int = 1024, hidden: int = 128, kernel: int = 5):
        super().__init__()
        self.proj = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.ctx = nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=kernel // 2)
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, subsampled: torch.Tensor) -> torch.Tensor:
        """subsampled: [B, T, d_model] -> per-frame probability [B, T]."""
        x = subsampled.transpose(1, 2)
        x = F.relu(self.proj(x))
        x = F.relu(self.ctx(x))
        x = torch.sigmoid(self.out(x))
        return x.squeeze(1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ParakeetRedux(nn.Module):
    def __init__(self, num_layers: int = 24, d_model: int = 1024, n_heads: int = 8,
                 d_ff: int = 4096, num_mel_bins: int = 128, conv_kernel_size: int = 9,
                 subsampling_conv_channels: int = 256, subsampling_factor: int = 8,
                 decoder_hidden_size: int = 640, num_decoder_layers: int = 2,
                 vocab_size: int = BLANK_ID + 1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, n_heads, d_ff, num_mel_bins,
                               conv_kernel_size, subsampling_conv_channels,
                               subsampling_factor)
        self.encoder_projector = nn.Linear(d_model, decoder_hidden_size)
        self.decoder = Predictor(vocab_size, decoder_hidden_size, num_decoder_layers)
        self.joint = JointNetwork(decoder_hidden_size)
        self.vad_head = VADHead(d_model)

    @property
    def d_model(self) -> int:
        return self.encoder.layers[0].norm_out.normalized_shape[0]


__all__ = [
    "BLANK_ID", "DURATIONS", "SECONDS_PER_FRAME",
    "DwStridingSubsampling", "RelPositionalEncoding", "FeedForward",
    "ConvolutionModule", "RelPosAttention", "ConformerLayer", "Encoder",
    "Predictor", "JointNetwork", "VADHead", "ParakeetRedux",
]
