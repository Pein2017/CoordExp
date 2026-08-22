# Checkpoints and artifacts

Training checkpoints carry an inference payload and may separately carry an
exact-resume training state.

## Inference payload

The inference payload is self-authenticated at checkpoint publication. Its
manifest covers the adapter and any special-token embedding delta and binds the
payload to its declared model/tokenizer identity. Inference consumes this
payload without treating it as an exact-resume admission.

## Exact-resume state

Exact resume is an opt-in training artifact. It is admitted only with matching
identity, complete rank contributions, matching world size, and an
optimizer-boundary save. Any missing, corrupt, incomplete, or incompatible
state fails closed.

## Downstream artifacts

Inference writes its artifact directory before direct evaluation. The evaluator
requires the scored artifact family and emits `metrics.json` plus COCO
conversion artifacts. Visualization is derived from the run artifacts rather
than replacing the evaluator.

Use [infer-eval](infer-eval.md) for commands and [the OpenSpec contract](../../openspec/README.md)
for normative compatibility rules.
