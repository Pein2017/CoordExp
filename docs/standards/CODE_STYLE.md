---
doc_id: docs.standards.code-style
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Medium-weight code and architecture style guidance for the current CoordExp-Swift tree.
updated: 2026-07-11
---

# Code And Architecture Style

This is a medium-weight style guide for a research-grade, YAML-first Python ML
repository. It preserves stable interfaces and research meaning without
turning every experiment into a framework.

## Goals and non-goals

Goals:

- keep runs reproducible and paper-ready;
- keep imports light and optional dependencies isolated;
- make features easy to locate, test, and document;
- keep core contracts stable while experiments evolve;
- expose research-semantic choices rather than hiding them in orchestration.

Non-goals:

- perfect uniformity across every historical file;
- speculative abstractions or one-adapter interfaces for imagined variation;
- broad rewrites to make the tree look newer;
- changing a stable contract without the OpenSpec lifecycle.

## Source ownership

Use the [implementation map](../IMPLEMENTATION_MAP.md) for current modules,
entrypoints and verification owners. Keep source inventories there so style
guidance does not become a second, drifting implementation map.

## Interfaces and module depth

Use a deep module where substantial behavior is hidden behind a small, honest
interface. The interface includes types, invariants, ordering, configuration,
failure modes, artifacts, and research semantics.

- Prefer a real owner over a pass-through facade.
- Put variation at an explicit seam only when it exists in the current code.
- Accept dependencies explicitly rather than creating hidden global coupling.
- Return observable typed results or receipts when side effects matter.
- Test through the same semantic interface callers use.
- Do not hide user-owned choices about data order, geometry, targets, loss,
  optimization, metric scope, or artifact interpretation.

The deletion test is useful: if deleting a module makes complexity disappear,
it was probably pass-through structure; if complexity reappears in callers, the
module was hiding useful behavior.

## Typed outputs and contracts

Prefer small dataclasses or typed mappings over positional tuples. Document
units and invariants, especially norm1000 versus pixel coordinates, logical
versus packed positions, and raw versus scored artifacts.

Use cohesive typed records for shapes shared across modules; prefer immutable
records where callers should not mutate the contract. Dictionaries fit JSON/YAML
IO boundaries, intentionally loose metadata, and local short-lived values.
Normalize external data at the boundary and serialize records at the output
boundary instead of passing anonymous dictionaries through many layers.

Keep `src/__init__.py` and low-level import surfaces light. Stable records,
config models, artifact receipts, and metric events should be explicit enough
for tests and downstream readers to validate them without opening implementation
internals.

## Imports and optional dependencies

Do not import heavyweight optional dependencies at module import time when
avoidable. Keep backend-specific loading behind the current runtime/backend
seams and fail with an actionable message that names the missing dependency and
the supported alternative.

Do not introduce a new backend or framework merely to make a document or
interface symmetrical. Verify supported backends through the
[implementation map](../IMPLEMENTATION_MAP.md) and their live source; reserved
fields do not establish runtime support.

## Configuration: YAML first and strict

Treat configs as first-class artifacts:

- prefer a YAML key plus typed validation over a new CLI flag;
- keep names descriptive and reproducible;
- reject unknown keys and invalid combinations at load time;
- record authored/resolved config identity and fingerprints;
- keep defaults neutral unless a research choice is intentionally explicit.

Place label/loss choices near objective, template or data ownership; execution
choices near runtime ownership; metric-comparability choices near evaluation.
Names should distinguish serialized format, template contract, runtime mode and
research variant. Preserve one spelling for shared concepts and keep aliases at
explicit boundaries. Correctness that should always hold does not need a knob.

The current config seam is `src/config/loader.py` and `src/config/models.py`.
Do not copy an old schema from `configs/stage1/` or `configs/stage2/` into a
current `configs/coordexp_swift/` document without checking the live models and
stable specs.

When adding a config key:

1. name the owning model and invariant;
2. add strict schema validation;
3. document the user-visible meaning and default;
4. add a negative test for invalid/unknown input;
5. update the relevant stable spec if the compatibility contract changes.

## Logging and artifacts

Logs and artifacts should make correctness failures visible, not merely reduce
noise. Prefer the current typed metric stream and artifact receipts. Preserve
artifact names and provenance when moving ownership; a migration needs explicit
tests and documentation.

Checkpoint handoff identity is distinct from exact training-state resume. Do not
use the word “resume” without qualifying which payload and state are actually
restored.

## Documentation style

Document public entrypoints and contracts with:

- a first line stating the behavior;
- the important invariants and units;
- the inputs, outputs, and failure modes;
- a link to the owning `docs/` page or stable OpenSpec for longer semantics.

Keep canonical docs concise and pointer-first. Put dated evidence, experiments,
and historical architecture reasoning in their designated history/research
surfaces rather than duplicating it in current routers.

## Testing style

Tests should be deterministic, contract-focused, and runnable from the repo
root. Prefer small tests for pure geometry, parsing, encoding, packing, and
artifact functions; test the real interface rather than private incidental
helpers. Add a focused smoke test for a new end-to-end component and gate
expensive tests behind explicit opt-in.

Useful current test areas are `tests/config/`, `tests/data/`, `tests/templates/`,
`tests/qwen/`, `tests/packing/`, `tests/supervision/`, `tests/losses/`,
`tests/training/`, `tests/runtime/`, `tests/artifacts/`, `tests/inference/`, and
`tests/eval/`.

## Hard guardrails

- Preserve geometry, object order, token alignment, and artifact row identity.
- Keep Qwen chat-template and image-token contracts explicit.
- Do not silently resize, reorder, drop, or renormalize research inputs.
- Do not edit upstream model internals to work around a local contract.
- If a stable behavior, config schema, loss, metric, or artifact contract must
  change, use the OpenSpec lifecycle and update current docs after promotion.
