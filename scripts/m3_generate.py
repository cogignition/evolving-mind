#!/usr/bin/env python3
"""Latent generate with brainstem interoception and regulator (HUB-246)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glr.intero import INTERO_DIM
from glr.model import GLR, REPO, generate_latent

DEFAULT_CKPT = Path("checkpoints/head_m3.safetensors")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--prompt", default="What is 2+2? Put the answer after the reasoning.")
    p.add_argument("--no-regulator", action="store_true")
    args = p.parse_args()
    if not args.ckpt.is_file():
        print(f"FAIL: missing {args.ckpt}", file=sys.stderr)
        return 1

    glr, tokenizer = GLR.load(args.repo, intero_dim=INTERO_DIM)
    glr.load_head(args.ckpt)
    text, meta = generate_latent(
        glr,
        tokenizer,
        args.prompt,
        k=args.k,
        max_new=args.max_new,
        use_regulator=not args.no_regulator,
    )
    print(json.dumps({"k_used": meta["k_used"], "scales": meta["scales"]}))
    print(text)
    if not text.strip():
        print("FAIL: empty generation", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
