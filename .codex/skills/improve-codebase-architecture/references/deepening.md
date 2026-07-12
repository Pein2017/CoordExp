# Deepening

Use this when consolidating a cluster of shallow modules. Assume the vocabulary
from the parent skill: module, interface, depth, seam, adapter, leverage, and
locality.

## Dependency categories

Classify each dependency before choosing a seam.

### In-process

Pure computation or in-memory state. Usually keep it inside the deepened module
and test through the external interface. An internal function does not by itself
justify an adapter.

### Local-substitutable

Dependencies with a faithful local stand-in: tiny JSONL fixtures, synthetic
image metadata, temporary artifact roots, config parsers, tokenizer fixtures,
or evaluator fixtures. Keep the seam internal and exercise the stand-in through
the module interface.

### Remote but owned

Infrastructure or services controlled by the project. Define a port only when
production and test/local adapters are both justified. Keep transport outside
research logic.

### True external

Third-party behavior such as model hubs, remote checkpoint stores, vLLM
servers, Baidu Netdisk, or benchmark submission systems. Inject an adapter and
test owned logic against a controlled substitute. Verify executed upstream
semantics separately when they affect correctness.

### Research contract

Data format, geometry/order, tokenizer/template behavior, forward semantics,
loss and normalization, config schema, metric scope, artifact names, cache
identity, and provenance are not ordinary dependencies. Deepening must preserve
or explicitly change their meaning with the corresponding user decision and
verification.

## Seam discipline

- One adapter is hypothetical; two adapters make variation real.
- Internal test seams need not become caller-visible configuration.
- Do not expose a framework object merely because the implementation uses it.
- Do not convert a correctness invariant into a user knob.
- Keep strict, diagnostic, compatibility, and temporary paths visibly separate.
- Accept variable dependencies explicitly instead of creating them invisibly.
- Return observable results; when side effects are inherent, expose a receipt or
  artifact.

## Replace instead of layering

- Add tests at the new interface before deleting existing coverage.
- Delete shallow tests only after proving equivalent behavior and contract
  coverage at the new seam.
- Assert outcomes, artifacts, receipts, and failure modes instead of internal
  call choreography.
- Include config parsing, geometry/image alignment, metric identity,
  artifact/manifest, replay, distributed, or smoke checks when those contracts
  cross the interface.

## Deepening receipt

Record:

```text
Concept and owner
Old caller knowledge
New interface
Hidden implementation
Dependency categories and seams
User-owned semantic decisions
Contract impact
Replacement/deletion plan
Verification
```
