"""Ternary unpacking tests: digit order is validated against the
zero-fraction values recorded per module in ternary.json (a scale-independent
quantity computed from the packed codes alone).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from parakeet_redux_torch.safetensors_io import load_safetensors
from parakeet_redux_torch.ternary import unpack_ternary, zero_fraction_of_codes

MODEL_DIR = Path(__file__).resolve().parent.parent / "weights"
TERNARY = json.loads((MODEL_DIR / "ternary.json").read_text()) if (MODEL_DIR / "ternary.json").exists() else None

pytestmark = pytest.mark.skipif(TERNARY is None, reason="weights/ not downloaded")


@pytest.fixture(scope="module")
def tensors():
    return load_safetensors(MODEL_DIR / "model.safetensors")


def test_zero_fractions_match_recorded(tensors):
    """All 264 quantized modules: unpacked zero fraction == recorded value."""
    mismatches = []
    for m in TERNARY["quantized_modules"]:
        q = tensors[m["name"] + ".qweight"]
        zf = zero_fraction_of_codes(q, m["in_features"])
        if abs(zf - m["zero_fraction"]) > 1e-6:
            mismatches.append((m["name"], zf, m["zero_fraction"]))
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:3]}"


def test_unpacked_values_are_ternary(tensors):
    """De-scaling must recover exactly {-1, 0, +1} codes for every group."""
    checked = 0
    for m in TERNARY["quantized_modules"]:
        if checked >= 8:
            break
        q = tensors[m["name"] + ".qweight"]
        s = tensors[m["name"] + ".scales"]
        w = unpack_ternary(q, s, m["in_features"])
        assert w.shape == (m["out_features"], m["in_features"])
        scale = np.repeat(s.astype(np.float32), 128, axis=1)[:, :m["in_features"]]
        nonzero = scale != 0
        ratio = w[nonzero] / scale[nonzero]
        assert np.allclose(ratio, np.round(ratio), atol=1e-5), m["name"]
        assert set(np.unique(np.round(ratio).astype(np.int64))) <= {-1, 0, 1}, m["name"]
        checked += 1
    assert checked == 8


def test_all_quantized_modules_present(tensors):
    for m in TERNARY["quantized_modules"]:
        assert m["name"] + ".qweight" in tensors
        assert m["name"] + ".scales" in tensors
    assert len(TERNARY["quantized_modules"]) == 264
