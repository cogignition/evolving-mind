# Evolving Mind

Self-steering small-model agent with a persistent protoself (interoception +
geometric latent reasoning). Linear project: Evolving Mind. Spec and paper
live on the issue, not here.

Not a SysOp surface. Runtime is mlx-lm on a dense Qwen3, not LM Studio.

## M1

Prove token-path hidden states match embedding-path hidden states on
`mlx-community/Qwen3-1.7B-bf16`, then that a perturbed embedding still yields
finite logits. First run downloads ~3.44 GB.

```
just m1
```

mlx-lm is pinned at 0.31.3.
