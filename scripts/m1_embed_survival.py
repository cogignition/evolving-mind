#!/usr/bin/env python3
"""M1: prove raw input embeddings survive attention/FFN.

Token-path hidden states must match embedding-path hidden states on
mlx-community/Qwen3-1.7B-bf16. A perturbed embedding must still yield
finite logits. See HUB-244 / arXiv:2606.02248.
"""

from __future__ import annotations

import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import does_model_support_input_embeddings

REPO = "mlx-community/Qwen3-1.7B-bf16"
DELTA_MAX = 1e-5
NOISE_SCALE = 0.05


def _prompt_ids(tokenizer) -> mx.array:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    kwargs = {"add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(prompt, str):
        return mx.array(tokenizer.encode(prompt))
    return mx.array(prompt)


def main() -> int:
    model, tokenizer = load(REPO)
    if not does_model_support_input_embeddings(model):
        print("FAIL: model does not accept input_embeddings", file=sys.stderr)
        return 1

    ids = _prompt_ids(tokenizer)
    E = model.model.embed_tokens(ids)
    h_tok = model.model(ids[None])
    h_emb = model.model(ids[None], input_embeddings=E[None])
    mx.eval(h_tok, h_emb)
    delta = float(mx.mean(mx.abs(h_tok - h_emb)))
    print(f"mean abs hidden delta: {delta}")
    if delta >= DELTA_MAX:
        print(f"FAIL: delta {delta} >= {DELTA_MAX}", file=sys.stderr)
        return 1

    noise = mx.random.normal(E.shape) * NOISE_SCALE
    h_pert = model.model(ids[None], input_embeddings=(E + noise)[None])
    logits = model.model.embed_tokens.as_linear(h_pert)
    mx.eval(logits)
    finite = bool(mx.all(mx.isfinite(logits)))
    tok = int(mx.argmax(logits[0, -1]))
    print(f"perturbed last-token: {tokenizer.decode([tok])!r}")
    print(f"finite: {finite}")
    if not finite:
        print("FAIL: perturbed logits not finite", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
