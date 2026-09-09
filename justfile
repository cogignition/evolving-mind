# Evolving Mind. `just` is the index; Linear is the work control surface.

default:
    @just --list

# M1: prove raw input embeddings survive attention/FFN on Qwen3-1.7B bf16
m1:
    uv run python scripts/m1_embed_survival.py
