"""Load `moondream/parakeet-redux` weights into the pure-PyTorch model.

The repo ships one `model.safetensors` whose keys already follow the
HuggingFace conversion naming (see ternary.json "names": "hf"). Ternary-packed
linear/conv weights appear as `<name>.qweight` (uint8, base-3 packed) plus
`<name>.scales` (fp16); everything else is dense (fp16/fp32). This loader
materializes the packed modules to float32 `.weight` and performs a strict
state_dict load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import ParakeetRedux
from .safetensors_io import load_safetensors
from .ternary import unpack_ternary, zero_fraction_of_codes


def build_state_dict(tensors: dict[str, np.ndarray], ternary_meta: dict,
                     verify_ternary: bool = False) -> dict[str, torch.Tensor]:
    quant = {m["name"]: m for m in ternary_meta["quantized_modules"]}
    default_group = ternary_meta.get("quant", {}).get("group_size", 128)

    pending: dict[str, dict] = {}
    sd: dict[str, torch.Tensor] = {}
    for name, arr in tensors.items():
        if name.endswith(".qweight"):
            pending.setdefault(name[: -len(".qweight")], {})["q"] = arr
        elif name.endswith(".scales"):
            pending.setdefault(name[: -len(".scales")], {})["s"] = arr
        else:
            sd[name] = torch.from_numpy(np.array(arr, copy=True))

    for base, parts in pending.items():
        meta = quant.get(base)
        if meta is None:
            raise KeyError(f"no ternary metadata for module {base!r}")
        w = unpack_ternary(parts["q"], parts["s"], meta["in_features"],
                           meta.get("group_size", default_group))
        if meta.get("as_conv1d"):
            # ternary.json flags kernel-1 Conv1d modules; restore the kernel axis
            w = w[:, :, None]
        if verify_ternary:
            zf = zero_fraction_of_codes(parts["q"], meta["in_features"])
            ok = abs(zf - meta["zero_fraction"]) < 1e-6
            print(f"  ternary check {base}: zero_fraction {zf:.10f} "
                  f"vs {meta['zero_fraction']:.10f} -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise ValueError(f"ternary unpack mismatch for {base}")
        sd[base + ".weight"] = torch.from_numpy(w)

    # normalize dtypes: everything float32 except integer buffers
    final: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if v.dtype in (torch.float16, torch.bfloat16, torch.float64):
            v = v.to(torch.float32)
        final[k] = v
    return final


def load_model(model_dir: str | Path, device: str = "cpu",
               verify_ternary: bool = False):
    """Load the model from a directory containing model.safetensors,
    config.json and ternary.json.

    Returns (model, config, ternary_meta).
    """
    model_dir = Path(model_dir)
    tensors = load_safetensors(model_dir / "model.safetensors")
    ternary_meta = json.loads((model_dir / "ternary.json").read_text())
    config = json.loads((model_dir / "config.json").read_text())

    enc = config.get("encoder_config", {})
    model = ParakeetRedux(
        num_layers=enc.get("num_hidden_layers", 24),
        d_model=enc.get("hidden_size", 1024),
        n_heads=enc.get("num_attention_heads", 8),
        d_ff=enc.get("intermediate_size", 4096),
        num_mel_bins=enc.get("num_mel_bins", 128),
        conv_kernel_size=enc.get("conv_kernel_size", 9),
        subsampling_conv_channels=enc.get("subsampling_conv_channels", 256),
        subsampling_factor=enc.get("subsampling_factor", 8),
        decoder_hidden_size=config.get("decoder_hidden_size", 640),
        num_decoder_layers=config.get("num_decoder_layers", 2),
        vocab_size=config.get("vocab_size", 8193),
    )

    sd = build_state_dict(tensors, ternary_meta, verify_ternary=verify_ternary)

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    missing = model_keys - sd_keys
    unexpected = sd_keys - model_keys
    if missing or unexpected:
        msg = []
        if missing:
            msg.append(f"missing in checkpoint: {sorted(missing)[:5]} (+{max(0, len(missing)-5)} more)")
        if unexpected:
            msg.append(f"unexpected in checkpoint: {sorted(unexpected)[:5]} (+{max(0, len(unexpected)-5)} more)")
        raise RuntimeError("state_dict mismatch:\n  " + "\n  ".join(msg))

    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model, config, ternary_meta
