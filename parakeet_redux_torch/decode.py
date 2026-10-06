"""Greedy TDT decoding with NeMo `GreedyTDTInfer` semantics.

Control flow (faithful to NeMo's reference implementation):

    while time_idx < T:
        inner loop up to max_symbols_per_step times:
            decoder:  g = predict(last_token, state)   (blank/SOS first, zero state)
            joint:    token = argmax(logits[:V+1]), dur = argmax(logits[V+1:])
            if token != blank: emit, update state, last_token = token
            time_idx += durations[dur]
            continue inner loop iff duration == 0
        if the inner loop ran out of budget with duration == 0: time_idx += 1
"""
from __future__ import annotations

import torch

from .model import BLANK_ID, SECONDS_PER_FRAME


@torch.no_grad()
def greedy_tdt_decode(model, enc_frames: torch.Tensor, durations: tuple[int, ...] = (0, 1, 2, 3, 4),
                      max_symbols: int | None = 10) -> list[tuple[int, int, int]]:
    """Decode encoder frames to a list of (token_id, frame_idx, duration_frames).

    Args:
        model: ParakeetRedux with `.encoder_projector`, `.decoder`, `.joint`.
        enc_frames: [T, d_model] encoder output for a single utterance.
        durations: the duration bins predicted by the joint network.
        max_symbols: max symbols per encoder frame (None = unlimited).
    """
    enc_proj = model.encoder_projector(enc_frames)  # [T, decoder_hidden]
    num_tokens = BLANK_ID + 1                        # vocab incl. blank
    T = enc_proj.shape[0]

    tokens: list[tuple[int, int, int]] = []
    time_idx = 0
    last_token: int | None = None
    state = None  # (h, c)

    while time_idx < T:
        f = enc_proj[time_idx]
        symbols_added = 0
        skip = 0
        need_loop = True
        while need_loop and (max_symbols is None or symbols_added < max_symbols):
            if last_token is None and state is None:
                label = torch.tensor([BLANK_ID], dtype=torch.long)
                dec_out, new_state = model.decoder.step(label, None)
            else:
                label = torch.tensor([last_token], dtype=torch.long)
                dec_out, new_state = model.decoder.step(label, state)

            logits = model.joint.head(torch.relu(f + dec_out[0]))
            k = int(torch.argmax(logits[:num_tokens]))
            d = int(torch.argmax(logits[num_tokens:]))
            skip = durations[d]

            if k != BLANK_ID:
                tokens.append((k, time_idx, skip))
                last_token = k
                state = new_state

            symbols_added += 1
            time_idx += skip
            need_loop = skip == 0

        if skip == 0:
            time_idx += 1

    return tokens


def tokens_to_words(tokens: list[tuple[int, int, int]], tokenizer) -> list[dict]:
    """Group tokens into words with approximate timestamps (seconds)."""
    words: list[dict] = []
    for token_id, frame, skip in tokens:
        piece = tokenizer.id_to_piece.get(token_id, "")
        start = frame * SECONDS_PER_FRAME
        end = start + skip * SECONDS_PER_FRAME
        if piece.startswith("\u2581") or not words:
            words.append({"ids": [token_id], "start": start, "end": end})
        else:
            words[-1]["ids"].append(token_id)
            words[-1]["end"] = max(words[-1]["end"], end)

    return [
        {"word": tokenizer.decode(w["ids"]),
         "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for w in words
    ]
