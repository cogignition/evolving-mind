# Plan: M1 repo + embedding survival ([HUB-244](https://linear.app/hublar/issue/HUB-244))

Proceed. Execute the [HUB-243](https://linear.app/hublar/issue/HUB-243) recommendations.

## In scope
- Create private `cogignition/evolving-mind` at `~/src/evolving-mind`
- Pin `mlx-lm==0.31.3` (uv lockfile)
- `just m1` runs `scripts/m1_embed_survival.py` against `mlx-community/Qwen3-1.7B-bf16`
- Record pass/fail on this issue

## Out of scope
- Transition head / two-pass train (M2)
- Interoception, regulator, harness, memory (M3–M5)
- SysOp tree (`press/` `web/` `cdn/` `services/`)
- `lms load` / touching Hermes's ornith slot

## Verify
- `gh repo view cogignition/evolving-mind` exists
- `uv lock` pins mlx-lm 0.31.3
- `just m1` prints `mean abs hidden delta` under `1e-5`, `finite: True`, and a decoded token
- Linear comment with the numbers
