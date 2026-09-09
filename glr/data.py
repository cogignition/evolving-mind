"""Load Open-R1 Mixture-of-Thoughts math traces and locate <think> spans."""

from __future__ import annotations

from typing import Any, Iterator

from glr.model import THINK_END, THINK_START

DATASET = "open-r1/Mixture-of-Thoughts"
DATASET_CONFIG = "math"


def _encode_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if isinstance(prompt, str):
        return tokenizer.encode(prompt, add_special_tokens=False)
    return list(prompt)


def _find_token(ids: list[int], tok: int, start: int = 0) -> int:
    try:
        return ids.index(tok, start)
    except ValueError:
        return -1


def parse_example(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_len: int,
    min_thought: int,
) -> dict[str, Any] | None:
    """Return index arrays for one CoT example, or None if it cannot be used."""
    try:
        ids = _encode_messages(tokenizer, messages)
    except Exception:
        return None
    if len(ids) < 8 or len(ids) > max_len:
        return None
    think_at = _find_token(ids, THINK_START)
    end_at = _find_token(ids, THINK_END, think_at + 1) if think_at >= 0 else -1
    if think_at < 0 or end_at < 0:
        return None
    thought_start = think_at + 1
    thought_end = end_at
    m = thought_end - thought_start
    if m < min_thought:
        return None
    answer_start = end_at + 1
    if answer_start >= len(ids) - 1:
        return None
    thought_idx = list(range(thought_start, thought_end))
    prev_idx = [think_at] + thought_idx[:-1]
    return {
        "ids": ids,
        "thought_idx": thought_idx,
        "prev_idx": prev_idx,
        "answer_start": answer_start,
    }


def iter_math(
    tokenizer,
    *,
    n: int,
    max_len: int,
    min_thought: int,
    seed: int = 0,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=256)
    got = 0
    seen = 0
    for row in ds:
        seen += 1
        messages = row.get("messages")
        if not messages:
            continue
        parsed = parse_example(
            tokenizer, messages, max_len=max_len, min_thought=min_thought
        )
        if parsed is None:
            if seen % 50 == 0:
                print(f"scanned {seen}, kept {got}", flush=True)
            continue
        yield parsed
        got += 1
        print(f"scanned {seen}, kept {got}/{n} (len={len(parsed['ids'])})", flush=True)
        if got >= n:
            return
