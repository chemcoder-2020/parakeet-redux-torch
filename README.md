# parakeet-redux-torch

Run [**moondream/parakeet-redux**](https://huggingface.co/moondream/parakeet-redux) speech-to-text with **plain PyTorch** — no `moondream` package, no Photon runtime, no NeMo.

Parakeet Redux is Moondream's compressed variant of NVIDIA's Parakeet TDT 0.6B: a 604M-parameter FastConformer-TDT speech recognizer (25 languages, punctuation, word timestamps) whose encoder weights are **ternary-quantized** to `{-1, 0, +1}` and packed in a custom format (`thrush-ternary-v2`) that normally only Moondream's Photon engine understands.

This repo unpacks the ternary weights and runs the whole model — encoder, TDT decoder, tokenizer, log-mel frontend — on stock `torch.nn`/numpy ops, and **reproduces the official runtime's transcripts exactly** on every test file.

## Verified: identical output to the official runtime

Six public test clips were transcribed with both this port and the official package (`pip install "moondream>=2.4.1"`, Photon engine, CPU) on the same machine:

| file | audio | lang | word-level WER vs Photon | transcript |
| --- | --- | --- | --- | --- |
| `bcn_weather.mp3` | 11.0 s | en | 0.0% | identical |
| `librispeech_mr_quilter.wav` | 5.9 s | en | 0.0% | identical |
| `en-Alice_woman.wav` | 9.3 s | en | 0.0% | identical |
| `mary_had_lamb.mp3` | 16.0 s | en | 0.0% | identical |
| `fleur_es_sample.wav` | 7.7 s | es | 0.0% | identical |
| `f2641_0_throatclearing.wav` | 2.9 s | — | 0.0% | identical (both empty) |

**Aggregate: 0.00% WER over 115 reference words — 6/6 identical transcripts**, including a Spanish clip, a 48 kHz mp3, a 40 kbps mp3, and a non-speech edge case (both engines return empty text). Word timestamps match too: on `bcn_weather.mp3` every word start/end agrees with Photon to the millisecond. Raw outputs for every file are recorded in `testdata/` (`ours_*.json` vs `ref_*.json`) and re-comparable with `scripts/compare_with_reference.py`.

Component-level checks (rerunnable):

- **Ternary unpacking** — the decoded weights' zero-fraction matches the value recorded in the model repo for **all 264 quantized modules** to 1e-6 (`python -m pytest tests/ -q`, or `--verify-ternary` on the CLI).
- **Log-mel frontend vs librosa** — slaney mel filterbank agrees to 3.7e-09 (max abs, float32); the full frontend to 1.1e-06 on a 585-frame sample (`scripts/verify_mel_vs_librosa.py`).
- **7/7 tests pass**: unpack + end-to-end transcription tests.

### Speed

Apple M1 (16 GB), Python 3.11, torch 2.14.1, fp32, warmed up (wall time, min of repeated runs):

| file | audio | CPU (default) | CPU, `--threads 1` | GPU encoder (`--encoder-device mps`) | Photon CPU | Photon MPS |
| --- | --- | --- | --- | --- | --- | --- |
| `bcn_weather.mp3` | 11.0 s | 0.84 s (13× RT) | 0.63 s (17× RT) | **0.37 s (30× RT)** | 0.35 s (32× RT) | 0.29 s (39× RT) |
| `mary_had_lamb.mp3` | 16.0 s | 1.01 s (16× RT) | 0.81 s (20× RT) | **0.52 s (31× RT)** | 0.49 s (33× RT) | 0.43 s (37× RT) |

Every configuration above was verified to reproduce the same six identical transcripts as Photon.

Notes (all measured on this machine):

- **`--encoder-device mps`** runs the encoder on the Apple GPU and keeps greedy TDT decoding on the CPU (per-step dispatch dominates those small tensors on the GPU). Within ~7% of Photon's CPU wall time: `parakeet-redux-torch audio.wav --threads 1 --encoder-device mps` is the fastest stock-PyTorch configuration found.
- **`--threads 1`** — torch intra-op threading *hurts* this workload on Apple Silicon (many small ops; thread hand-off costs more than it saves). On the reference machine one thread beats four by ~25%.
- **Why Photon's CPU path is still faster:** its kernels compute directly from the 2-bit packed weights with NEON int8 dot-products (`gemm8`), quantize activations to int8, and fuse whole conformer blocks into single passes. This port materializes fp32 weights (~1.2 GB vs 160 MB packed) and runs stock fp32 `torch` ops. Options measured and not taken: dynamic int8 is unavailable in torch builds for macOS ARM (no functional quantized engine — `NoQEngine` at `linear_prepack`); `torch.compile` measured ~20% *slower* than eager here; replacing the slow per-group depthwise-conv fallback (~25k `slow_conv2d` dispatches) with a ~4× faster shifted multiply-add loop flipped borderline words on one test clip, so the exact path was kept.

## Quickstart

Requires Python ≥ 3.10, PyTorch, numpy, and `ffmpeg` on PATH (for audio decoding).

```bash
git clone https://github.com/chemcoder-2020/parakeet-redux-torch
cd parakeet-redux-torch
python -m venv .venv && source .venv/bin/activate
pip install -e .

# download the weights (~178 MB, CC-BY-4.0) into ./weights
python scripts/download_model.py

# CLI
parakeet-redux-torch testdata/bcn_weather.mp3 --words

# fastest stock-PyTorch configuration on Apple Silicon (see Speed section)
parakeet-redux-torch testdata/bcn_weather.mp3 --threads 1 --encoder-device mps
```

```python
from parakeet_redux_torch.loader import load_model
from parakeet_redux_torch.tokenizer import ParakeetTokenizer
from parakeet_redux_torch.transcribe import Transcriber

model, config, _ = load_model("weights")
transcriber = Transcriber(model, ParakeetTokenizer("weights/tokenizer.json"), config)

result = transcriber.transcribe("audio.wav", with_words=True)
print(result["text"])
print(result["words"][:3])   # [{'word': 'Yesterday', 'start': 0.64, 'end': 1.36}, ...]
```

## How it works

```
audio ──ffmpeg──> 16 kHz mono ──log-mel──> [T×128] ──FastConformer──> [T/8×1024] ──greedy TDT──> tokens ──BPE──> text
```

1. **Audio** (`audio.py`): ffmpeg → 16 kHz mono float32 (soundfile fallback for 16 kHz files).
2. **Frontend** (`mel.py`): pre-emphasis 0.97; STFT n_fft 512 / window 400 / hop 160 (non-periodic Hann); 128 slaney-normalized mel bins, 0–8 kHz; `log(mel + 2⁻²⁴)`; per-feature mean/var normalization (unbiased). Verified against librosa.
3. **Encoder** (`model.py`): 8× depthwise-striding subsampling → 24 Conformer blocks (relative-position MHSA with Transformer-XL-style `rel_shift`, macaron feed-forward, k=9 conv module, dropout off at eval) → 1024-dim frames.
4. **Decoding** (`decode.py`): greedy TDT, faithful to NeMo's `GreedyTDTInfer`: joint argmax is split into `[vocab+blank | 5 duration bins]`; the 2-layer LSTM predictor state updates only on non-blank; per-step skip logic and the `max_symbols_per_step=10` fallback behave like the reference. Token→word grouping uses the 80 ms encoder-frame grid.
5. **Tokenizer** (`tokenizer.py`): metaspace-BPE decode (~50 lines, no `tokenizers` dependency). One subtlety found during verification: this tokenizer lists the digits `0`–`9` as *added tokens*, so a naive "skip added tokens" decoder silently drops all digits; the decoder here skips only real control pieces (`<|…|>`, `<unk>`, `<pad>`, `<blank>`).

### The ternary format

The encoder ships as packed ternary (`ternary.json` → `"format": "thrush-ternary-v2"`): base-3 digits, 5 per byte, **least-significant digit first**, row-major, pad digits 0; weight rule `w[row, col] = scales[row, col // 128] * (code − 1)` with `code ∈ {0,1,2}` → weights in `{−1, 0, +1}` per 128-element group. 264 modules (604M params) are ternary; the remaining 23M (embeddings, LSTM, joint head, VAD head, subsampling convs) are dense. `ternary.json` also records a per-module `zero_fraction` — a free scale-independent checksum that this port uses to prove digit-order correctness (`--verify-ternary`).

## Repo layout

```
parakeet_redux_torch/   the package (loader, mel, model, decode, tokenizer, audio, transcribe, cli)
scripts/
  download_model.py          fetch weights from HuggingFace
  reference_moondream.py     reference transcription via the official moondream package
  compare_with_reference.py  run this port on all test files, diff vs references, save ours_*.json
  inspect_pipeline.py        stage-by-stage pipeline printout (mel/encoder/decode)
  verify_mel_vs_librosa.py   independent frontend verification vs librosa
  timing_reference.py        warmed-up Photon timing
tests/                  pytest suite (unpack checksums + end-to-end transcription)
testdata/               small public samples + recorded reference/our outputs (see testdata/README.md)
```

## Notes and limitations

- **Long recordings:** this port processes the whole file in one pass and accepts up to 5000 encoder frames (~6.7 min per pass — the checkpoint's `max_position_embeddings`). The official runtime instead splits long recordings at VAD-pause boundaries (the checkpoint's `vad_head` weights are loaded and available in the state dict, but chunking is not implemented here). Beyond 5000 frames the transcriber raises with a clear message instead of silently degrading.
- **Hardware:** verified on CPU (Apple M1) and on the Apple-Silicon GPU encoder path (`--encoder-device mps` — transcript parity checked on all six test files); `--device cuda` is wired through `load_model` but untested for numerical parity.
- **fp32 throughout**; no int8/fp16 paths, no torch.compile, no batch mode.
- Transcripts preserve the model's own casing/punctuation; `fleur_es_sample.wav` shows expected Spanish output.

## Attribution & licenses

- Model weights: [`moondream/parakeet-redux`](https://huggingface.co/moondream/parakeet-redux) by Moondream — **CC-BY-4.0**, based on NVIDIA's `parakeet-tdt-0.6b-v3` (CC-BY-4.0). Weights are *not* included in this repo; `scripts/download_model.py` fetches them.
- This code: Apache-2.0. Developed by clean-rooming against the HuggingFace Parakeet reference implementations and the publicly documented file formats; the `moondream` package is only used by verification scripts, never at runtime.
- Test audio from [`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples).
