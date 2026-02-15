## Context

The current strict-validation implementation enforces important invariants, but the mechanism scales poorly:
- nested key allowlists are enumerated in multiple places,
- runtime validators duplicate schema concerns,
- adding a new key requires synchronized edits across loader and trainer checks.

External mature patterns (`transformers` + `ms-swift`) converge on one principle:
- define argument/config surfaces once via typed contracts,
- parse strictly,
- reject unused/unknown keys centrally.

This change adopts that principle for CoordExp YAML configs while keeping current constraints:
- YAML-first,
- strict fail-fast,
- no backward compatibility aliases,
- no new dependency.

## Goals / Non-Goals

**Goals**
- Replace manual nested key enumeration as the primary mechanism with schema-derived strict validation.
- Keep all existing high-signal semantic validations (range checks, required fields, cross-field constraints).
- Enforce unknown-key fail-fast with full dotted paths.
- Reduce runtime duplication by keeping schema shape checks in loader and runtime checks only for execution-dependent constraints.

**Non-Goals**
- Changing Stage-2 training algorithms or rollout semantics.
- Introducing Pydantic/Marshmallow/other third-party schema dependencies.
- Reintroducing compatibility aliases for removed keys.

## Decisions

### 1) Typed schema is the source of truth

Decision:
- Use typed dataclass contracts for each top-level config section and nested rollout/stage2_ab groups.
- Section parsing is performed by a generic strict loader that derives accepted keys from dataclass fields.

Rationale:
- Eliminates repeated allowlist edits.
- Keeps contracts inspectable and reviewable in code.

### 2) Generic strict parser contract

Decision:
- Introduce a reusable loader utility (name to be finalized in implementation) with behavior:
  - input: `(schema_type, payload, path)`,
  - verifies payload mapping type,
  - computes accepted keys from declared dataclass fields (exact-key matching),
  - recursively parses nested dataclasses/lists/mappings,
  - errors on unknown keys with `Unknown <path> keys: [...]`.

Error requirements:
- Unknown key errors must include full dotted key path to the failing key.
- For list elements, path format is `parent.list_field[<index>].<key>`.
- Type/range/value errors keep explicit sectioned messages.

### 3) Coverage includes all top-level config sections

Decision:
- First implementation slice covers all top-level sections currently loaded by `TrainingConfig.from_mapping`:
  - `model`, `quantization`, `template`, `data`, `tuner`, `training`, `rlhf`, `custom`, `debug`, `stage2_ab`, `rollout_matching`, `deepspeed`, `global_max_length`, `extra` (reserved/rejected at top-level).

Rationale:
- Avoid partial architecture split where only rollout is schema-derived but the rest remains ad-hoc.

Clarification:
- Unknown-key fail-fast applies to section-owned keys.
- Explicit extension buckets declared by schema remain allowed by design (currently `custom.extra`).
- `custom.extra` remains a narrow escape-hatch mapping and is not a location for canonical rollout keys.
- Top-level `extra` is not an extension bucket; under strict parsing, any top-level `extra:` presence (including `{}`) is unsupported and MUST fail fast.
- `extra` in section-coverage means explicit detection/rejection ownership at loader boundary; it does not authorize parsing arbitrary top-level `extra` payloads.

### 4) Runtime checks are narrowed to runtime-only invariants

Decision:
- Keep trainer-side validation only for invariants requiring runtime context (e.g., world size, distributed mode, backend readiness assumptions).
- Remove trainer-side duplicated unknown-key/shape enumeration that schema already guarantees.

Rationale:
- Single ownership for config shape (loader), single ownership for runtime constraints (trainer).

### 5) Strict no-compat policy remains

Decision:
- No alias support for legacy Stage-2 rollout key paths.
- No compatibility aliases are introduced by the strict parser in this change.
- Any `custom.extra.rollout_matching.*` usage remains fail-fast with migration guidance.
- Legacy rollout server paired-list shape (`vllm.server.base_url` + `vllm.server.group_port`) is removed; only `vllm.server.servers[]` is supported.
- Removed Stage-2 Channel-B knobs remain hard fail-fast with no compatibility aliases:
  - `stage2_ab.channel_b.reordered_gt_sft`
  - `stage2_ab.channel_b.desc_ce_weight_matched`
  - `stage2_ab.channel_b.semantic_desc_gate`

Rationale:
- Avoid shadowed behavior and maintain reproducibility contract.

### 6) Launcher preflight JSON contract remains normative

Decision:
- This refactor MUST preserve the shared preflight resolver contract already defined by Stage-2 hierarchy change:
  - launcher preflight consumes shared Python normalization output,
  - Python resolver emits shell assignment lines; the `ROLLOUT_CONTRACT_JSON` value MUST be newline-terminated single-line JSON,
  - required keys and types: `rollout_backend` (str), `vllm_mode` (str|null), `server_base_urls` (list[str]).
- The 3-key contract above is the minimum normative contract.
- Additional preflight payload fields are allowed, but MUST NOT weaken or replace the minimum contract.
- Parser architecture changes MUST NOT weaken this launcher contract.

Rationale:
- Prevent parser refactor from silently breaking launch-time safety checks.

## Proposed Architecture

### Data flow

1. YAML load + extends resolution (`ConfigLoader.load_yaml_with_extends`).
2. Top-level section extraction.
3. Strict schema parse per section via generic loader.
4. Section-level semantic checks (`__post_init__` / section validators).
5. Cross-section checks (e.g., trainer variant -> required section presence).
6. Runtime receives normalized typed config.

### Module shape (implementation target)

- `src/config/schema.py`
  - keeps top-level orchestration and cross-section constraints.
- `src/config/<new_schema_module>.py`
  - holds generic strict parsing utility + possibly split section dataclasses.
- `src/sft.py`
  - preserves rollout normalization/injection contract (`rollout_matching_cfg`) and legacy-path fail-fast behavior while moving static key ownership to schema parsing.
- `src/trainers/rollout_matching_sft.py`
  - keeps runtime-only checks; duplicated static key-shape ownership is removed only after schema parity checks pass.

## Trade-offs

- Slight upfront refactor cost to define consistent typed nested structures.
- Strong reduction in long-term maintenance churn and drift risk.
- Strictness is preserved; ergonomics improve by centralizing where key acceptance is defined.

## Verification Strategy

1. Positive config-load checks for canonical Stage-2 prod/smoke profiles.
2. Negative unknown-key checks at:
   - top-level,
   - each major section,
   - nested rollout sections (including list-indexed paths such as `servers[0]`).
3. Extension-bucket checks:
   - `custom.extra.<minor_key>` remains accepted,
   - `custom.extra.rollout_matching.*` fails fast.
   - presence of top-level `extra:` fails fast.
4. Runtime sanity:
   - preflight still resolves canonical rollout settings,
   - preflight output JSON contract (`rollout_backend`, `vllm_mode`, `server_base_urls`) remains valid and is emitted as newline-terminated single-line JSON,
   - rollout decode batch source-of-truth behavior unchanged.
5. Semantic/type-shape checks:
   - representative required-field/range/cross-field invariants remain enforced,
   - representative wrong-shape payloads fail at loader level with clear path-aware errors.
   - legacy rollout server paired-list shape fails fast with migration guidance to `vllm.server.servers[]`.

## Rollout / Migration

1. Introduce generic strict parser utility and migrate all declared top-level sections (including `deepspeed`, `global_max_length`, `extra`) with parity tests.
2. Keep runtime duplicated static checks during transition; prove schema parity first.
3. Remove duplicated runtime static-key ownership only after parity gates pass (runtime-only invariants remain).
4. Keep strict gates enabled throughout migration (no warning window).
5. Migrate known non-Stage2 configs that relied on permissive `custom.*` unknown-key behavior.
6. Update tests and finalize by removing stale duplicated validators.
