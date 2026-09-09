# Plan: M3 brainstem ([HUB-246](https://linear.app/hublar/issue/HUB-246))

Proceed. The cortex already takes a spark (M1) and walks in embedding
space (M2). This unit is the medulla: a body that feels its own error,
and a law that damps the stride when the error rises.

Spec: interoceptive vector from harness telemetry, concatenated into
`g_ϕ`; homeostatic regulator gating steps. Damasio's protoself sits
here, not in the language.

## In scope
- Five-channel interoception, read from the frozen forward, not imagined:
  prediction error, confidence, compute load, surprise, energy budget
- Head becomes `Linear(d+5, d)`; intero is `stop_gradient`
- Regulator: high error shrinks the displacement and can cut K short;
  low error lengthens the stride
- Retrain on Open-R1 math (session subset); new checkpoint
- `just m3-train` / `m3-gen` / `m3-verify`

## Out of scope
- Memory bank, self-modifying harness (M4–M5)
- Paper-scale 10k/8k/5ep
- SysOp; `lms load`

## Verify
- Trainable params ~4.20M (head only, d+5)
- `L_Δ` decreases
- Ablation: real intero vs zeros changes mean `|Δ|`
- High vs low injected error changes regulator scale
- `just m3-gen --k 10` emits text
