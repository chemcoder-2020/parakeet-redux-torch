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

Apple M1 (16 GB), Python 3.11, torch 2.14.1, warmed up (wall time, min of repeated runs):

| file | audio | CPU (default) | CPU, `--threads 1` | GPU encoder (`--encoder-device mps`) | GPU enc., `--encoder-dtype float16` | Photon CPU | Photon MPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bcn_weather.mp3` | 11.0 s | 0.84 s (13× RT) | 0.63 s (17× RT) | 0.37 s (30× RT) | **0.30 s (37× RT)** | 0.35 s (32× RT) | 0.29 s (39× RT) |
| `mary_had_lamb.mp3` | 16.0 s | 1.01 s (16× RT) | 0.86 s (19× RT) | 0.52 s (31× RT) | **0.43 s (37× RT)** | 0.49 s (33× RT) | 0.43 s (37× RT) |

All fp32 configurations reproduce the same six identical transcripts as Photon; the fp16 encoder path (approximate numerics — see notes) matches them too on all six files.

Notes (all measured on this machine):

- **CPU is the default and works everywhere:** `--threads 1` → 0.63 s / 0.86 s (17–19× realtime, exact fp32). Thread sweep on `bcn_weather.mp3`: 1 → 0.64 s, 2 → 0.81 s, 4 (torch default) → 0.86 s, 8 → 1.42 s — torch intra-op threading *hurts* this workload on Apple Silicon (many small ops; thread hand-off costs more than it saves); sweep on your own CPU. `OMP_WAIT_POLICY=passive`: no effect at 1 thread.
- **Long-form (CPU):** single-pass and linear — 17 s → 0.85 s, 26 s → 1.20 s, 42 s → 1.94 s, 53 s → 2.40 s (~21–22× realtime). Against Photon CPU on the same concatenated clips: **byte-identical up to ~26 s**; beyond ~30 s Photon switches to VAD-chunked decoding and the outputs diverge (42 s: first difference at one character; 53 s: larger downstream drift) — keep parity-critical inputs within one pass, or chunk long audio yourself.
- **vs Photon on the same CPU:** 1.8× slower on short clips (0.63 vs 0.35 s — their kernels compute from 2-bit packed weights with NEON int8 dot-products (`gemm8`) and fused conformer blocks; this port materializes fp32 weights and runs stock ops, and dynamic int8 is unavailable in torch builds for macOS ARM — `NoQEngine`), narrowing to ~1.1–1.6× on longer audio where their chunking overhead grows. Other levers measured and not taken: `torch.compile` (~20% slower than eager on CPU, +2% on MPS), `scaled_dot_product_attention` fusing (dispatches −10%, wall time unchanged), a bit-exact manual LSTM cell for the decode loop (~10% of decode, <3% end-to-end), and a faster shifted depthwise-conv formulation that flipped borderline words on one clip.
- **Apple GPU (optional, faster):** `--encoder-device mps` runs the encoder on the Apple GPU and keeps greedy TDT decoding on the CPU (per-step dispatch dominates those small tensors on the GPU) — 0.37 s / 0.52 s, exact. Adding `--encoder-dtype float16` casts **the encoder only** to half precision — like Photon's own GPU path: 0.30 s / 0.43 s (~37× realtime, Photon-MPS parity). Numerics are approximate (encoder output differs from fp32 by up to ~4e-4); all six reference transcripts still match, but fp32 stays the default.

Step-by-step reproduction, from fresh clone to verified run: [Reproduce the results (CPU first)](#reproduce-the-results-cpu-first).

## Quickstart

Requires Python ≥ 3.10, PyTorch, numpy, and `ffmpeg` on PATH (for audio decoding).

```bash
git clone https://github.com/chemcoder-2020/parakeet-redux-torch
cd parakeet-redux-torch
python -m venv .venv && source .venv/bin/activate
pip install -e .

# weights (~178 MB, CC-BY-4.0) ship in-repo via Git LFS — a `git clone` with
# git-lfs installed fetches them automatically; otherwise:
git lfs pull          # or, without LFS:  python scripts/download_model.py

# CLI
parakeet-redux-torch testdata/bcn_weather.mp3 --words

# CPU, tuned — works everywhere (~17x realtime on the reference machine;
# sweep --threads on other CPUs)
parakeet-redux-torch testdata/bcn_weather.mp3 --threads 1

# fastest configuration on Apple Silicon, ~37x realtime (details and
# verification: the "Reproduce the results" section below; fp16 =
# approximate numerics, drop the flag for the strict-exact path)
parakeet-redux-torch testdata/bcn_weather.mp3 --threads 1 --encoder-device mps --encoder-dtype float16
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

## Reproduce the results (CPU first)

This is CPU inference: no GPU needed, works on macOS, Linux and Windows, and runs on **stock PyTorch — the `moondream` package is not installed** (step 4 is the only optional exception: a throwaway venv used purely to time the official engine for comparison). Reference machine: M1 (16 GB), Python 3.11, torch 2.14.1.

**1. Code, weights, environment (one-time):**

```bash
# git-lfs is how the weights ship (one-time: brew install git-lfs && git lfs install)
git clone https://github.com/chemcoder-2020/parakeet-redux-torch
cd parakeet-redux-torch
git lfs pull                        # weights/ (~178 MB); no git-lfs? python scripts/download_model.py

python -m venv .venv && source .venv/bin/activate
pip install -e .                    # torch + numpy only — nothing else
```

**2. Transcribe on CPU:**

```bash
parakeet-redux-torch testdata/bcn_weather.mp3 --threads 1
```

Output on the reference machine:

```
Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to minus 20 degrees.

[audio 11.04s | wall 0.64s | 17.23x realtime]
```

`--threads 1` is the tuned setting on Apple Silicon (thread sweep: 1 → 0.64 s, 2 → 0.81 s, 4 → 0.86 s, 8 → 1.42 s — the encoder is thousands of small ops and thread hand-off costs more than it saves). On other CPUs, sweep for yourself:

```bash
for t in 1 2 4 8; do parakeet-redux-torch testdata/bcn_weather.mp3 --threads $t; done
```

**3. Reproduce and verify (one command):**

```bash
python scripts/verify_photon_speed.py
```

Runs all six reference clips on CPU, checks every transcript against the recorded Photon outputs (`testdata/ref_*.json`), prints warmed wall times, and exits non-zero on any mismatch. Reference result:

```
PASS  bcn_weather.mp3                    0.624 s
PASS  librispeech_mr_quilter.wav         0.449 s
PASS  fleur_es_sample.wav                0.472 s
PASS  en-Alice_woman.wav                 0.568 s
PASS  mary_had_lamb.mp3                  0.862 s
PASS  f2641_0_throatclearing.wav         0.331 s

identical transcripts: 6/6
reference wall (M1, warmed, cpu/float32): bcn_weather.mp3 0.63 s, got 0.624 s
reference wall (M1, warmed, cpu/float32): mary_had_lamb.mp3 0.86 s, got 0.862 s
```

On an Apple-silicon Mac, `--device mps` (optionally with `--dtype float16`) runs the same verification on the Metal-GPU configurations.

**4. (Optional) time the official engine on the same CPU.** Only this step needs the `moondream` package, in a separate venv, only for comparison — nothing at runtime depends on it:

```bash
python -m venv .venv-ref
.venv-ref/bin/pip install "moondream>=2.4.1"
.venv-ref/bin/python scripts/timing_reference.py
# testdata/bcn_weather.mp3: ~0.35s ...  (this is the "Photon CPU" column of the Speed table)
```

**Apple GPU (optional, faster):** on an Apple-silicon Mac the encoder can also run on the Metal GPU — `--encoder-device mps`, and optionally `--encoder-dtype float16` (the same fp16 convention as Photon's GPU path; approximate numerics): that's the 0.30 s / 0.43 s row of the Speed table. Verify it the same way: `python scripts/verify_photon_speed.py --device mps`.

**Caveats:**

- Parity is scoped to what was measured: the six clips identical; concatenated audio identical up to ~26 s; beyond ~30 s Photon switches to VAD-chunked decoding and the transcripts diverge (see the Speed notes). The CPU path is fp32 end to end.
- `--encoder-dtype float16` is approximate (encoder output drifts ≤ ~4e-4 vs fp32). It matches all six oracle transcripts, but rounding can in principle flip a borderline greedy decision on other audio — which is why fp32 stays the default.
- The Apple-GPU flags need Apple Silicon (`torch.backends.mps.is_available()`); CUDA is wired through but untested for numerical parity.
- All numbers are warmed, min-of-repeated-runs, model load excluded — compare like-for-like only.

**Windows / non-Apple platforms:** the model and CLI are fully portable (pure PyTorch + numpy; ffmpeg for audio decoding) and the CPU path above works as-is — only the optional Apple-GPU flags are Apple-specific. Translations:

- **Setup commands:** venv activation is `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`); git-lfs ships with the Git for Windows installer (or `winget install GitHub.GitLFS`); ffmpeg via `winget install Gyan.FFmpeg` — or skip it and `pip install soundfile` if your audio is already 16 kHz wav; hashes via `Get-FileHash weights\model.safetensors -Algorithm SHA256` instead of `shasum`.
- **Threads:** the "1 thread wins" result is Apple-Silicon-tuned — on x86, sweep `--threads 1 2 4 8` before settling.
- **NVIDIA GPU:** `--device cuda` (optionally `--encoder-device cuda --encoder-dtype float16`) is wired through but **untested for numerical parity** — verify first: `python scripts/verify_photon_speed.py --device cuda` (expect 6/6 identical; treat a mismatch as a finding, not a fluke).
- **Don't** use `--encoder-dtype float16` on CPU (slower — x86 has no fast fp16).
- Verify on your machine: `python scripts/verify_photon_speed.py` (CPU by default) — expect 6/6 and your timings.

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
  verify_photon_speed.py     reproduction + verification script (CPU by default; --device mps for the Apple GPU)
  timing_reference.py        warmed-up Photon timing
tests/                  pytest suite (unpack checksums + end-to-end transcription)
testdata/               small public samples + recorded reference/our outputs (see testdata/README.md)
weights/                model files shipped via Git LFS (CC-BY-4.0) + upstream card, notice, checksums
```

## Notes and limitations

- **Long recordings:** this port processes the whole file in one pass and accepts up to 5000 encoder frames (~6.7 min per pass — the checkpoint's `max_position_embeddings`). The official runtime instead splits long recordings at VAD-pause boundaries (the checkpoint's `vad_head` weights are loaded and available in the state dict, but chunking is not implemented here). Beyond 5000 frames the transcriber raises with a clear message instead of silently degrading. **Measured parity scope:** byte-identical to Photon for single-pass inputs up to ~26 s (concatenated clips); beyond ~30 s the official runtime switches to VAD-chunked mode and the outputs diverge (see the Speed notes).
- **Hardware:** verified on CPU (Apple M1) and on the Apple-Silicon GPU encoder path (`--encoder-device mps` — transcript parity checked on all six test files); `--device cuda` is wired through `load_model` but untested for numerical parity.
- **fp32 by default** (there is an opt-in fp16 encoder path for Apple GPUs — see Speed); no int8 paths, no torch.compile, no batch mode.
- Transcripts preserve the model's own casing/punctuation; `fleur_es_sample.wav` shows expected Spanish output.

## Attribution & licenses

- Model weights: [`moondream/parakeet-redux`](https://huggingface.co/moondream/parakeet-redux) by Moondream — **CC-BY-4.0**, based on NVIDIA's `parakeet-tdt-0.6b-v3` (CC-BY-4.0). The weights ship in this repo via Git LFS (`weights/`), with the license text (`weights/LICENSE`), attribution (`weights/NOTICE`), the upstream model card (`weights/README.md`), and per-file SHA256s (`weights/checksums.txt`). `scripts/download_model.py` can still fetch a fresh copy from Hugging Face.
- This code: Apache-2.0. Developed by clean-rooming against the HuggingFace Parakeet reference implementations and the publicly documented file formats; the `moondream` package is only used by verification scripts, never at runtime.
- Test audio from [`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples).
