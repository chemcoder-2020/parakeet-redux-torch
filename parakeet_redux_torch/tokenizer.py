"""Metaspace-BPE decoding for the Parakeet tokenizer (no external dependency).

The repo ships a HuggingFace `tokenizer.json` (SentencePiece-style BPE with the
metaspace character U+2581 for word boundaries). We implement just enough of it
to turn generated token ids into text:

  - genuine control pieces ("<|...|>", "<unk>", "<pad>", "<blank>") are skipped
    (note: the digits "0"-"9" ARE listed as added tokens in this tokenizer but
    are ordinary text - they must not be skipped)
  - "<0xAB>" byte-fallback pieces are turned back into raw bytes
  - the metaspace character becomes a space; the result is stripped
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_CONTROL_PIECE = re.compile(r"^<\|.*\|>$")
_CONTROL_LITERALS = {"<unk>", "<pad>", "<blank>", "<s>", "</s>"}


def is_control_piece(piece: str) -> bool:
    return bool(_CONTROL_PIECE.match(piece)) or piece in _CONTROL_LITERALS


class ParakeetTokenizer:
    def __init__(self, tokenizer_json: str | Path):
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.id_to_piece: dict[int, str] = {int(v): k for k, v in data["model"]["vocab"].items()}
        self.vocab_size = len(data["model"]["vocab"])

    def decode(self, ids) -> str:
        """Decode a list of token ids to text."""
        buf = bytearray()
        for i in ids:
            piece = self.id_to_piece.get(int(i))
            if piece is None or is_control_piece(piece):
                continue
            m = _BYTE_PIECE.match(piece)
            if m:
                buf += bytes([int(m.group(1), 16)])
            else:
                buf += piece.encode("utf-8")
        text = buf.decode("utf-8", errors="replace").replace("\u2581", " ")
        return text.strip()
