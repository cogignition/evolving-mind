#!/usr/bin/env python3
"""K-step latent generate with a trained GLR head (HUB-245)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glr.model import GLR, REPO, generate_latent

DEFAULT_CKPT = Path("checkpoints/head.safetensors")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--prompt", default="What is 2+2? Put the answer after the reasoning.")
    args = p.parse_args()
    if not args.ckpt.is_file():
        print(f"FAIL: missing {args.ckpt}", file=sys.stderr)
        return 1

    glr, tokenizer = GLR.load(args.repo)
    glr.load_head(args.ckpt)
    text = generate_latent(
        glr, tokenizer, args.prompt, k=args.k, max_new=args.max_new
    )
    print(f"{{k={args.k} latent steps}}")
    print(text)
    if not text.strip():
        print("FAIL: empty generation", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
