# Evolving Mind. `just` is the index; Linear is the work control surface.

default:
    @just --list

# M1: prove raw input embeddings survive attention/FFN on Qwen3-1.7B bf16
m1:
    uv run python scripts/m1_embed_survival.py

# M2: train the GLR head on Open-R1 math CoT displacements
m2-train *args:
    uv run python scripts/m2_train.py {{args}}

# M2: K-step latent generate with the trained head
m2-gen *args:
    uv run python scripts/m2_generate.py {{args}}
