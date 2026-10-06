"""Unpacking of the "thrush-ternary-v2" packed representation used by
moondream/parakeet-redux.

Format (from the model repo's ternary.json):
  - packing: base 3, 5 elements per byte, row-major
  - element i of a row is base-3 digit (i % 5) of byte (i // 5),
    least significant digit first; pad digits are 0
  - weight rule:  w[row, col] = scales[row, col // group_size] * (code - 1)
    with code in {0, 1, 2}  ->  values in {-1, 0, +1}
"""
from __future__ import annotations

import numpy as np

# LUT: for each byte value, its 5 base-3 digits (LSB first).
_DIGIT_LUT = np.array(
    [[(b // (3 ** k)) % 3 for k in range(5)] for b in range(256)],
    dtype=np.uint8,
)  # shape [256, 5]


def unpack_ternary(qweight: np.ndarray, scales: np.ndarray, in_features: int,
                   group_size: int = 128) -> np.ndarray:
    """Materialize a ternary-quantized weight matrix.

    Args:
        qweight: uint8 array of shape [rows, ceil(in_features / 5)].
        scales:  array of shape [rows, ceil(in_features / group_size)].
        in_features: the true (unpadded) number of columns.
        group_size: scale group size (128 for this model).

    Returns:
        float32 array of shape [rows, in_features].
    """
    rows = qweight.shape[0]
    if qweight.shape[1] * 5 < in_features:
        raise ValueError(
            f"qweight row holds {qweight.shape[1] * 5} digits, need {in_features}"
        )
    codes = _DIGIT_LUT[qweight]                    # [rows, nbytes, 5]
    codes = codes.reshape(rows, -1)[:, :in_features]  # [rows, in_features]
    w = codes.astype(np.float32) - 1.0             # {-1, 0, +1}
    scale = np.asarray(scales, dtype=np.float32)   # [rows, groups]
    w *= np.repeat(scale, group_size, axis=1)[:, :in_features]
    return w


def zero_fraction_of_codes(qweight: np.ndarray, in_features: int) -> float:
    """Fraction of ternary codes equal to 1 (i.e. weight value 0).

    The model repo records this per module; it is a strong, scale-independent
    check that the unpacking digit order is correct.
    """
    rows = qweight.shape[0]
    codes = _DIGIT_LUT[qweight].reshape(rows, -1)[:, :in_features]
    return float((codes == 1).mean())
