#!/usr/bin/env python3
"""Train the brainstem head: hidden ⊕ interoception → displacement (HUB-246)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from glr.data import iter_math
from glr.intero import INTERO_DIM
from glr.model import GLR, REPO, two_pass_loss

DEFAULT_CKPT = Path("checkpoints/head_m3.safetensors")


def _n_trainable(module: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(module.trainable_parameters()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--min-thought", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lambda-delta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = p.parse_args()

    mx.random.seed(args.seed)
    glr, tokenizer = GLR.load(args.repo, intero_dim=INTERO_DIM)
    n_train = _n_trainable(glr)
    n_head = _n_trainable(glr.head)
    print(f"trainable: {n_train} (head {n_head} in={glr.hidden_size + INTERO_DIM})", flush=True)
    if n_train != n_head or not (4_000_000 < n_head < 4_500_000):
        print(f"FAIL: expected ~4.20M brainstem head, got {n_train}", file=sys.stderr)
        return 1

    print(f"loading {args.n} math traces (max_len={args.max_len})...", flush=True)
    examples = list(
        iter_math(
            tokenizer,
            n=args.n,
            max_len=args.max_len,
            min_thought=args.min_thought,
            seed=args.seed,
        )
    )
    if len(examples) < min(4, args.n):
        print(f"FAIL: only {len(examples)} usable traces", file=sys.stderr)
        return 1
    print(f"examples: {len(examples)}", flush=True)

    optimizer = optim.AdamW(
        learning_rate=args.lr, betas=[0.9, 0.95], weight_decay=1e-4
    )
    loss_and_grad = nn.value_and_grad(glr, two_pass_loss)

    first_delta: float | None = None
    last_delta = 0.0
    step = 0
    t0 = time.time()
    accum_grads = None
    accum_n = 0

    def apply_accum() -> None:
        nonlocal accum_grads, accum_n
        if accum_grads is None:
            return
        scale = 1.0 / accum_n
        grads = tree_map(lambda g: g * scale, accum_grads)
        optimizer.update(glr, grads)
        mx.eval(glr.parameters(), optimizer.state)
        accum_grads = None
        accum_n = 0

    for epoch in range(args.epochs):
        for ex in examples:
            ids = mx.array(ex["ids"], dtype=mx.int32)
            thought_idx = mx.array(ex["thought_idx"], dtype=mx.int32)
            prev_idx = mx.array(ex["prev_idx"], dtype=mx.int32)
            (loss, l_ce, l_delta), grads = loss_and_grad(
                glr,
                ids,
                thought_idx,
                prev_idx,
                ex["answer_start"],
                ex["thought_idx"][0],
                ex["thought_idx"][-1] + 1,
                args.gamma,
                args.lambda_delta,
            )
            mx.eval(loss, l_ce, l_delta)
            loss_f = float(loss)
            ce_f = float(l_ce)
            delta_f = float(l_delta)
            if not bool(mx.isfinite(loss).item()):
                print(f"FAIL: non-finite loss at step {step}", file=sys.stderr)
                return 1
            if first_delta is None:
                first_delta = delta_f
            last_delta = delta_f
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
            accum_n += 1
            if accum_n >= args.accum:
                apply_accum()
                step += 1
                if step == 1 or step % 5 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "loss": loss_f,
                                "ce": ce_f,
                                "delta": delta_f,
                                "sec": round(time.time() - t0, 1),
                            }
                        ),
                        flush=True,
                    )
            mx.clear_cache()
        apply_accum()

    glr.save_head(args.ckpt)
    print(f"wrote {args.ckpt}")
    print(f"L_delta first={first_delta} last={last_delta}")
    if first_delta is None or last_delta >= first_delta:
        print("FAIL: L_delta did not decrease", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
