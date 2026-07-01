## Why

Stage-2 config validation is currently strict but duplicated: schema-side checks and trainer runtime checks both enumerate rollout and nested keys manually. This creates churn and drift risk whenever a field is added/renamed, and makes the contract harder to maintain over time.

We need one authoritative, typed config contract that:
- fails fast on unsupported keys,
- avoids repeated hand-maintained allowlists,
- remains YAML-first and config-driven,
- stays compatible with strict reproducibility requirements (no legacy/backward compatibility path).

## What Changes

- Introduce schema-derived strict validation from typed dataclass contracts (no external schema dependency).
- Add a reusable strict parser that:
  - accepts only declared dataclass fields,
  - validates nested structures recursively,
  - raises unknown-key errors with full dotted paths.
- Apply this strict schema contract across all top-level training config sections:
  - `model`, `quantization`, `template`, `data`, `tuner`, `training`, `rlhf`, `custom`, `debug`, `stage2_ab`, `rollout_matching`, `deepspeed`, `global_max_length`, `extra` (reserved/rejected: any top-level `extra:` presence fails fast).
- Remove duplicated rollout key-enumeration responsibility from runtime validators where schema guarantees are already sufficient.
- Preserve runtime-only checks in trainer code (checks requiring live runtime state).
- Preserve launcher preflight JSON contract used by `scripts/train_stage2.sh` (`rollout_backend`, `vllm_mode`, `server_base_urls`) while changing parser internals.
- Keep `custom.extra` as the only intentional narrow extension bucket; strict fail-fast remains for canonical grouped sections.
- Clarify top-level `extra` policy: it is not an author-facing escape hatch; top-level `extra:` is unsupported under strict parsing.
- Keep hard fail-fast policy (no alias/compat support) for removed Stage-2 rollout key paths and unknown keys in strict sections.
- Remove legacy rollout server paired-list shape (`vllm.server.base_url` + `vllm.server.group_port`) and support only `vllm.server.servers[]`.
- Preserve the narrowed Stage-2 Channel-B contract: removed keys (`reordered_gt_sft`, `desc_ce_weight_matched`, `semantic_desc_gate`) remain load-time fail-fast.

## Capabilities

### New Capabilities

- Deterministic schema-derived unknown-key detection with dotted-path errors at every nested config layer.
- Single-source typed config contracts that drive loading/validation behavior without duplicated allowlist definitions.

### Modified Capabilities

- `stage2-ab-training`: validation is driven by declared schema fields instead of ad-hoc nested allowlist sets.
- `rollout-matching-sft`: rollout config contract remains strict, but strictness is authored once in typed schema and reused by loader/runtime paths.

## Impact

- Affected loader/schema code:
  - `src/config/schema.py`
  - `src/config/loader.py`
  - new typed-schema helper module(s) under `src/config/*` (exact file split per implementation).
- Affected trainer validation path:
  - `src/trainers/rollout_matching_sft.py` (remove duplicate shape checks that schema already enforces).
- Affected runtime normalization/injection path:
  - `src/sft.py` (maintain rollout injection/normalization ownership while parser internals are refactored).
- Affected tests:
  - config load tests and rollout config validation checks under `tests/`.
- Affected config migrations (strict parser applies via shared loader path):
  - `configs/dlora/sft_coord_loss.yaml`
  - `configs/dlora/debug-sft_coord_loss.yaml`
  - `configs/dlora/stage2_rollout_matching_ckpt3106.yaml`
  - `configs/dlora/stage2_rollout_matching_ckpt3106_server_3v1.yaml`
- Scope guard for strict global-loader impact:
  - run a sweep over `configs/**/*.yaml` for unknown `custom.*` keys outside `custom.extra.*`, migrate, and record outcomes.
- No new CLI flags and no relaxation of strict fail-fast policy.
- Breaking behavior remains intentional: unknown/legacy unsupported keys fail immediately.
