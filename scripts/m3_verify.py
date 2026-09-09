#!/usr/bin/env python3
"""Ablate interoception and the regulator (HUB-246)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx

from glr.intero import INTERO_DIM, from_logits, regulate
from glr.model import GLR, REPO, THINK_START

DEFAULT_CKPT = Path("checkpoints/head_m3.safetensors")
PROMPT = "What is 2+2?"


def _prefill(glr, tokenizer):
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(text, add_special_tokens=False) if isinstance(text, str) else list(text)
    if not ids or ids[-1] != THINK_START:
        ids.append(THINK_START)
    prompt_ids = mx.array(ids, dtype=mx.int32)
    h = glr.hidden(prompt_ids[None])
    return h[:, -1, :]


def main() -> int:
    ckpt = DEFAULT_CKPT
    if not ckpt.is_file():
        print(f"FAIL: missing {ckpt}", file=sys.stderr)
        return 1
    glr, tokenizer = GLR.load(REPO, intero_dim=INTERO_DIM)
    glr.load_head(ckpt)
    h = _prefill(glr, tokenizer)
    logits = glr.logits_from_hidden(h)
    intero = from_logits(logits, frac=mx.array([0.3], dtype=h.dtype))
    zeros = mx.zeros_like(intero)
    d_real = glr.stride(h, intero)
    d_zero = glr.stride(h, zeros)
    mx.eval(d_real, d_zero)
    gap = float(mx.mean(mx.abs(d_real - d_zero)))
    print(f"intero vs zeros mean |Δ| gap: {gap}")
    if gap < 1e-6:
        print("FAIL: head ignores interoception", file=sys.stderr)
        return 1

    high = mx.array([[6.0, 0.1, 0.9, 6.0, 0.1]], dtype=h.dtype)
    low = mx.array([[0.1, 0.9, 0.1, 0.1, 0.9]], dtype=h.dtype)
    _, s_high = regulate(d_real, high)
    _, s_low = regulate(d_real, low)
    sh, sl = float(s_high.reshape((-1,))[0]), float(s_low.reshape((-1,))[0])
    print(f"regulator scale high-err={sh:.3f} low-err={sl:.3f}")
    if not (sh < sl):
        print("FAIL: regulator did not shrink the stride under high error", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
