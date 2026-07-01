## Context

CoordExp dense prompting is currently global and open-vocabulary by default (`src/config/prompts.py`). COCO training data in this repo is intentionally bounded to 80 canonical classes (`public_data/coco/raw/categories.json`, COCO converter/validator scripts), but there is no first-class prompt profile that encodes this policy across both training and inference.

Current state:
- Training prompt defaults are resolved in `ConfigLoader.resolve_prompts` and consumed by dataset builders.
- Inference builds messages with hardcoded prompt constants in `src/infer/engine.py`.
- Fusion supports per-dataset prompt overrides, which must continue to work.

Constraints:
- Config-first workflow (no new CLI flags).
- Preserve Qwen3-VL chat-template compatibility.
- Preserve geometry/coord invariants; this change must only affect text instructions.
- Keep upstream HF internals untouched.

Stakeholders:
- Stage-1 COCO experimentation users.
- Inference/eval users who need training-inference prompt parity.

## Goals / Non-Goals

**Goals:**
- Add an extensible prompt-variant mechanism for future datasets.
- Introduce a built-in `coco_80` variant with compact canonical class list instructions.
- Ensure training and inference resolve prompts through one variant-aware path.
- Define deterministic resolution invariants so prompt text does not depend on call order or runtime surface.
- Persist resolved variant metadata in inference artifacts for reproducibility audits.
- Keep default behavior unchanged when no variant is selected.

**Non-Goals:**
- No runtime class whitelist filtering or post-hoc dropping in inference/eval.
- No changes to JSONL schema, geometry handling, or coordinate normalization.
- No changes to model architecture, tokenizer vocabulary, or upstream model files.

## Decisions

1) Centralize variant definitions in a dedicated registry module.
- Decision: create a new prompt variant registry in `src/config/` and keep `src/config/prompts.py` as the assembly layer.
- Why: keeps variants declarative, testable, and reusable by both training and inference.
- Alternatives considered:
  - Keep hardcoded `if variant` branches in `prompts.py`: rejected due to growth and maintenance risk.
  - Load variant text from external files at runtime: rejected for reproducibility and path fragility.

2) Use explicit YAML selectors for training and inference.
- Decision: consume `custom.extra.prompt_variant` for training and `infer.prompt_variant` for inference.
- Why: preserves strict custom parsing rules while using existing extension bucket for training; keeps inference YAML-first.
- Alternatives considered:
  - Add new top-level `custom.prompt_variant`: rejected to avoid schema expansion churn in a light change.
  - Add CLI flags: rejected (violates config-first guideline).

3) Freeze canonical COCO class source for the built-in variant.
- Decision: `coco_80` uses a frozen built-in class list aligned to `public_data/coco/raw/categories.json` snapshot names/order.
- Why: avoids runtime path coupling and guarantees deterministic prompt text across machines.
- Alternatives considered:
  - Load `categories.json` at runtime: rejected for reproducibility and environment/path fragility.

4) Keep class restriction prompt-only.
- Decision: `coco_80` changes instruction text only; no parser/evaluator enforcement changes in this spec.
- Why: minimal risk and light scope while still improving policy alignment.
- Alternatives considered:
  - Add runtime whitelist/dropper: rejected for now because it changes scoring semantics and is larger than this tracking change.

5) Preserve override precedence in fusion.
- Decision: dataset-level `prompt_user`/`prompt_system` overrides remain highest precedence over variant defaults.
- Why: compatibility with existing fusion behavior and experiments.

6) Prefer metadata-based parity auditing over hard runtime train/infer coupling checks.
- Decision: do not enforce checkpoint-to-inference variant equality at runtime in this light change; instead log resolved variant in inference artifacts and require docs/tests to recommend parity.
- Why: avoids checkpoint contract expansion while still enabling reproducibility audits.

## Data Flow

Data -> transforms/packing -> training/inference -> artifacts:
- Data and transforms remain unchanged (`public_data/...` and dataset preprocessors unaffected).
- Packing behavior remains unchanged.
- Training:
  - read variant key from `custom.extra.prompt_variant`
  - resolve variant-aware system/user prompts
  - build chat payloads with existing template machinery.
- Inference:
  - read variant key from `infer.prompt_variant`
  - resolve same variant-aware prompts
  - build backend request messages (HF/vLLM) with aligned prompt content.
- Artifacts:
  - include resolved `prompt_variant` in `resolved_config.json` and inference summary outputs for auditability.

## Configuration Shape

Training YAML (example):
```yaml
custom:
  extra:
    prompt_variant: coco_80
```

Inference pipeline YAML (example):
```yaml
infer:
  prompt_variant: coco_80
```

Parity policy:
- For reproducible COCO evaluation, inference SHOULD use the same variant used during training.
- If users intentionally diverge variants, artifact metadata MUST make that divergence visible.

## Risks / Trade-offs

- [Longer prompt text increases token budget] -> Mitigation: use compact COCO class list format and avoid ID+name verbosity.
- [Training/inference drift if one path misses variant wiring] -> Mitigation: single resolver usage and dedicated parity tests.
- [Users run inference with a different variant than training] -> Mitigation: document parity recommendation and persist resolved variant metadata in artifacts.
- [Future variant sprawl] -> Mitigation: strict registry key validation and explicit tests per variant.
- [Fusion override regressions] -> Mitigation: preserve current precedence and add regression test coverage.

## Migration Plan

1. Add variant registry and variant-aware prompt assembly (default + coco_80).
2. Wire training selector via `custom.extra.prompt_variant`.
3. Wire inference selector via `infer.prompt_variant`.
4. Add tests for resolver correctness and cross-surface parity.
5. Add resolved-variant metadata to inference artifacts (`resolved_config.json`/summary outputs).
6. Update docs with opt-in usage examples and parity guidance.

Rollback:
- Set configs back to default (remove variant keys), or revert change files if needed.
- Since no data/model/schema migration is introduced, rollback is low-risk.

## Open Questions

- None for this light tracking change; runtime class filtering remains intentionally out of scope.
