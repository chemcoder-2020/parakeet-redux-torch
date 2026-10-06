"""End-to-end transcription tests using the bundled small samples.

Skipped automatically when the model weights have not been downloaded
(`python scripts/download_model.py`).
"""
from pathlib import Path

import pytest

from parakeet_redux_torch.loader import load_model
from parakeet_redux_torch.tokenizer import ParakeetTokenizer
from parakeet_redux_torch.transcribe import Transcriber

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "model.safetensors").exists(),
    reason="weights not downloaded (run scripts/download_model.py)")


@pytest.fixture(scope="module")
def transcriber():
    model, config, _ = load_model(WEIGHTS)
    tokenizer = ParakeetTokenizer(WEIGHTS / "tokenizer.json")
    return Transcriber(model, tokenizer, config)


def test_librispeech_quilter(transcriber):
    """LibriSpeech sample with a well-known reference transcript."""
    res = transcriber.transcribe(ROOT / "testdata/librispeech_mr_quilter.wav",
                                 with_words=True)
    text = res["text"].lower()
    assert "quilter" in text
    assert "apostle" in text
    assert "gospel" in text


def test_bcn_weather(transcriber):
    """48 kHz mp3 sample with a known reference transcript."""
    res = transcriber.transcribe(ROOT / "testdata/bcn_weather.mp3")
    text = res["text"].lower()
    assert "barcelona" in text
    assert "35 degrees" in text


def test_spanish_sample(transcriber):
    """FLEURS Spanish sample - the model is multilingual (25 languages)."""
    res = transcriber.transcribe(ROOT / "testdata/fleur_es_sample.wav")
    assert len(res["text"]) > 20


def test_words_are_monotonic(transcriber):
    res = transcriber.transcribe(ROOT / "testdata/librispeech_mr_quilter.wav",
                                 with_words=True)
    starts = [w["start"] for w in res["words"]]
    assert starts == sorted(starts)
    assert all(w["end"] >= w["start"] for w in res["words"])
