"""Frozen Qwen3 backbone plus a linear GLR transition head."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import does_model_support_input_embeddings

from glr.intero import INTERO_DIM, from_logits, regulate

REPO = "mlx-community/Qwen3-1.7B-bf16"
THINK_START = 151667  # <think>
THINK_END = 151668  # </think>
ERROR_CUT = 4.0
MIN_LATENT = 3


class GLR(nn.Module):
    """g_ϕ on a frozen mlx-lm causal LM.

    intero_dim=0 is M2 (hidden only). intero_dim=5 is the brainstem: concat
    body-state onto the hidden state before the stride is predicted.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, intero_dim: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intero_dim = intero_dim
        self.backbone = backbone
        self.head = nn.Linear(hidden_size + intero_dim, hidden_size, bias=False)
        self.backbone.freeze()
        self.head.unfreeze()

    @classmethod
    def load(
        cls, repo: str = REPO, intero_dim: int = 0
    ) -> tuple["GLR", object]:
        backbone, tokenizer = load(repo)
        if not does_model_support_input_embeddings(backbone):
            raise RuntimeError(f"{repo} does not accept input_embeddings")
        hidden = int(backbone.args.hidden_size)
        glr = cls(backbone, hidden, intero_dim=intero_dim)
        mx.eval(glr.parameters())
        return glr, tokenizer

    def embed(self, ids: mx.array) -> mx.array:
        return self.backbone.model.embed_tokens(ids)

    def hidden(
        self,
        ids: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.backbone.model(
            ids, cache=cache, input_embeddings=input_embeddings
        )

    def logits_from_hidden(self, h: mx.array) -> mx.array:
        return self.backbone.model.embed_tokens.as_linear(h)

    def stride(self, h: mx.array, intero: mx.array | None = None) -> mx.array:
        if self.intero_dim == 0:
            return self.head(h)
        if intero is None:
            intero = mx.zeros((*h.shape[:-1], self.intero_dim), dtype=h.dtype)
        return self.head(mx.concatenate([h, intero], axis=-1))

    def save_head(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.head.save_weights(str(path))

    def load_head(self, path: Path) -> None:
        self.head.load_weights(str(path))
        mx.eval(self.head.parameters())


def two_pass_loss(
    glr: GLR,
    ids: mx.array,
    thought_idx: mx.array,
    prev_idx: mx.array,
    answer_start: int,
    t0: int,
    t1: int,
    gamma: float,
    lam: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """L_CE + λ L_Δ. CE on answer tokens only. L_Δ on the second pass."""
    e = glr.embed(ids)
    h1 = glr.hidden(ids[None])[0]
    h_prev = h1[prev_idx]
    m = thought_idx.shape[0]
    frac = mx.arange(m, dtype=mx.float32) / mx.maximum(
        mx.array(m - 1, dtype=mx.float32), mx.array(1.0)
    )
    intero = None
    if glr.intero_dim:
        logits_prev = glr.logits_from_hidden(h_prev)
        intero = mx.stop_gradient(
            from_logits(logits_prev, frac=frac, true_ids=ids[thought_idx])
        )
    delta_hat = glr.stride(h_prev, intero)
    e_hat = e[prev_idx] + delta_hat
    true_delta = e[thought_idx] - e[prev_idx]

    e2 = mx.concatenate([e[:t0], e_hat, e[t1:]], axis=0)
    h2 = glr.hidden(ids[None], input_embeddings=e2[None])[0]
    h_prev2 = h2[prev_idx]
    intero2 = None
    if glr.intero_dim:
        logits2 = glr.logits_from_hidden(h_prev2)
        intero2 = mx.stop_gradient(
            from_logits(logits2, frac=frac, true_ids=ids[thought_idx])
        )
    delta_hat2 = glr.stride(h_prev2, intero2)

    weights = gamma ** mx.arange(m, dtype=mx.float32)
    se = mx.sum((delta_hat2 - true_delta) ** 2, axis=-1)
    l_delta = mx.mean(weights * se)

    logits = glr.logits_from_hidden(h2[None])[0]
    shift_logits = logits[:-1]
    shift_labels = ids[1:]
    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="none")
    pos = mx.arange(ids.shape[0] - 1)
    mask = ((pos + 1) >= answer_start).astype(ce.dtype)
    denom = mx.maximum(mask.sum(), mx.array(1.0, dtype=ce.dtype))
    l_ce = (ce * mask).sum() / denom
    return l_ce + lam * l_delta, l_ce, l_delta


def generate_latent(
    glr: GLR,
    tokenizer,
    prompt: str,
    *,
    k: int,
    max_new: int = 128,
    use_regulator: bool = False,
) -> tuple[str, dict]:
    """Prefill through <think>, take K embedding steps, then greedy decode."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(text, str):
        ids = list(text)
    else:
        ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or ids[-1] != THINK_START:
        ids.append(THINK_START)

    prompt_ids = mx.array(ids, dtype=mx.int32)
    cache = make_prompt_cache(glr.backbone)
    h = glr.hidden(prompt_ids[None], cache=cache)
    e = glr.embed(prompt_ids[-1:])
    scales: list[float] = []
    k_used = 0

    for i in range(k):
        h_last = h[:, -1, :]
        intero = None
        if glr.intero_dim:
            logits = glr.logits_from_hidden(h_last)
            frac = mx.array([i / max(k, 1)], dtype=h.dtype)
            intero = from_logits(logits, frac=frac)
            if use_regulator:
                err = float(intero[0, 0])
                if i >= MIN_LATENT and err > ERROR_CUT:
                    break
        delta = glr.stride(h_last, intero)
        if use_regulator and intero is not None:
            delta, scale = regulate(delta, intero)
            scales.append(float(scale.reshape((-1,))[0]))
        e = e + delta
        dummy = mx.zeros((1, 1), dtype=prompt_ids.dtype)
        h = glr.hidden(dummy, cache=cache, input_embeddings=e[:, None, :])
        k_used += 1

    out: list[int] = []
    for _ in range(max_new):
        logits = glr.logits_from_hidden(h)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        t = int(tok.item())
        if t in tokenizer.eos_token_ids:
            break
        out.append(t)
        nxt = mx.array([[t]], dtype=prompt_ids.dtype)
        h = glr.hidden(nxt, cache=cache)
    decoded = tokenizer.decode(out)
    return decoded, {"k_used": k_used, "scales": scales}
