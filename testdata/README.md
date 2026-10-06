# Test fixtures

Small audio samples used by the end-to-end tests, plus recorded reference
outputs from the official `moondream` package (Photon engine) for comparison.

## Audio

All audio files come from the HuggingFace testing dataset
[`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples)
(see that repository's PROVENANCE.md). They are small, public, and intended
for exactly this kind of integration testing.

| file | source | notes |
| --- | --- | --- |
| `bcn_weather.mp3` | dummy-audio-samples | 48 kHz mp3; used in the official parakeet model cards |
| `librispeech_mr_quilter.wav` | LibriSpeech (dev-clean, via dummy-audio-samples) | "mister Quilter is the apostle..." |
| `fleur_es_sample.wav` | FLEURS (Spanish, via dummy-audio-samples) | multilingual check |
| `en-Alice_woman.wav` | dummy-audio-samples | conversational English |
| `mary_had_lamb.mp3` | dummy-audio-samples | 40 kbps mp3 fixture |
| `f2641_0_throatclearing.wav` | dummy-audio-samples | non-speech edge case |

## Reference outputs

`ref_*.json` files were produced with the official runtime
(`pip install "moondream>=2.4.1"`, CPU):

```bash
python scripts/reference_moondream.py testdata/<file> --timestamps word
```

`ours_*.json` files are produced by `scripts/compare_with_reference.py` with
this repository's pure-PyTorch implementation.
