# Plan: M2 GLR transition head ([HUB-245](https://linear.app/hublar/issue/HUB-245))

Proceed. Infuse the head. [HUB-244](https://linear.app/hublar/issue/HUB-244) proved
the frame accepts embeddings. This unit trains `g_ϕ` on real CoT displacements.

Paper: [arXiv:2606.02248](https://arxiv.org/abs/2606.02248) §3, Appendix A.

## In scope
- Linear head `d→d` on frozen `mlx-community/Qwen3-1.7B-bf16`
- Two-pass train: discrete CoT, replace think-span embeddings, `L_CE + λ L_Δ`
- `L_Δ`: discounted MSE, γ=0.999, no CE on latent tokens
- Data: `open-r1/Mixture-of-Thoughts` math, real traces (session subset)
- K-step latent generate via `model.model(..., input_embeddings=)`, then decode
- Head checkpoint + in-place `load_weights`
- `just m2-train` / `just m2-gen`

## Out of scope
- Paper-scale 10k × 8k × 5 epoch run (flags exist; not the gate)
- GSM8K / MATH500 eval
- M3–M5 (interoception, regulator, harness, memory)
- SysOp tree; `lms load`

## Verify
- Trainable params ~4M (head only)
- `just m2-train` : `L_Δ` finite and lower at last step than first
- `just m2-gen --k 10` emits text after the latent prefix
- Checkpoint reloads with `load_weights`
