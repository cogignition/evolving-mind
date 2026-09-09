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

## M2

Linear transition head on frozen Qwen3-1.7B. Two-pass train on
`open-r1/Mixture-of-Thoughts` math. Then K latent steps before decode.

```
just m2-train
just m2-gen --k 10
```

Paper-scale flags: `just m2-train --n 10000 --max-len 8192 --epochs 5`.
The default is a session subset (256 traces, 2048 ctx, 1 epoch).

## M3

Brainstem. Five-channel interoception from the frozen forward (error,
confidence, load, surprise, energy), concatenated into the head.
A regulator shrinks the stride when error is high.

```
just m3-train
just m3-verify
just m3-gen --k 10
```
