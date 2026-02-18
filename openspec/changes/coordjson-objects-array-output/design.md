## Context

CoordExp trains Qwen3-VL style VLMs for grounding/detection and dense captioning using a JSON-structured assistant output. The current assistant output contract is a top-level dict keyed by `"object_{n}"` with geometry represented as quoted coord-token strings (e.g., `"<|coord_123|>"`). This change is a breaking-format overhaul intended for from-scratch pretraining: replace the keyed dict with an `{"objects": [...]}` array container and represent geometry as bare coord-token literals in a JSON-like DSL (“CoordJSON”), then transpile CoordJSON into strict RFC JSON (geometry → integer bins) before the existing matching/eval/loss pipeline.

Constraints:
- Config-first: no new ad-hoc CLI flags. Any behavior change is YAML/schema-gated.
- Preserve geometry invariants: never reorder or drop coordinates; continue using existing geometry utilities and norm1000 semantics.
- Keep Qwen3-VL chat-template compatibility; do not modify upstream HF model internals.
- Raw datasets `*.{train,val}.coord.jsonl` are NOT modified; only cooking/templating/parsing/tooling/docs are updated.

## Goals / Non-Goals

**Goals:**
- Emit a single assistant payload container `{"objects": [...]}`
  - reduce structural token overhead vs `"object_{n}"` keys,
  - make predicted order unambiguous (array appearance order).
- Use CoordJSON for model-facing assistant messages:
  - geometry arrays contain bare `<|coord_k|>` literals (no quotes),
  - `desc` remains a standard JSON string,
  - output is “CoordJSON-only” (no natural-language wrapper).
- Provide a CoordJSON → strict JSON transpiler:
  - strict mode: fail-fast for cooked SFT/GT (record-level errors),
  - salvage mode: drop invalid/truncated predicted records, keep valid prefix.
- Preserve the existing algorithmic structure (matching, losses, eval), adapting only the container shape and indexing (`pred_index = objects[] index`).
- Keep ordering ablations: `custom.object_field_order ∈ {geometry_first, desc_first}` controls per-record key order in canonical serialization; object instance ordering remains governed by `custom.object_ordering`.
- Support both geometry kinds, but do not mix within one run:
  - bbox-only datasets use `bbox_2d`,
  - poly-only datasets use `poly`.

**Non-Goals:**
- No new detection heads or architecture forks.
- No backward compatibility with old checkpoints or old assistant output format.
- No JSON “repair” that inserts missing braces/quotes; only suffix-only trimming + per-record dropping for predictions.
- No max-object hard stop (repeat-terminate); sequence length truncation controls tail length.

## Decisions

1) **Top-level container: `{"objects": [...]}`**
   - Decision: require a single top-level object with a single key `"objects"` whose value is an array of records.
   - Rationale: stable empty form (`{"objects": []}`), easier to extend, and avoids bare-array ambiguities in chat templating.

2) **CoordJSON for model-facing assistant messages (non-RFC JSON)**
   - Decision: allow `<|coord_k|>` as a bare literal ONLY inside `bbox_2d` / `poly` arrays in assistant messages.
   - Rationale: makes coordinate emission more “atomic” for AR models; removes quote punctuation tokens; aligns with coord-token vocabulary usage.

3) **Transpile to strict RFC JSON before downstream parsing/matching**
   - Decision: introduce a transpiler that converts CoordJSON into strict JSON by mapping `<|coord_k|> → k` (int in 0..999).
   - Rationale: Hungarian matching, evaluation, and losses can keep using existing JSON-based utilities and do not need to become CoordJSON-aware.

4) **Strictness split: SFT/GT vs rollout preds**
   - Decision:
     - Cooked SFT/GT: strict mode MUST fail-fast on any schema, ordering, or serialization violation (record-level).
     - Rollout preds: salvage mode MUST drop invalid records and ignore an incomplete final record caused by max-length truncation; keep earlier valid records.
   - Rationale: SFT/GT is controllable and should never violate the contract; rollouts are inherently noisy and must not crash training.

5) **No extra keys in records**
   - Decision: record schema is minimal and closed: only `bbox_2d` OR `poly`, plus `desc`.
   - Rationale: stabilizes pretraining template, reduces tokens, and simplifies deterministic serialization and validation.

6) **No mixed geometry per run**
   - Decision: records MUST NOT contain both `bbox_2d` and `poly`. Training runs select bbox-only or poly-only JSONL sources.
   - Rationale: avoid ambiguity in matching/eval and prevent accidental schema drift.

7) **Remove repeat-terminate**
   - Decision: delete the repeat-terminate / max-object gating feature and its config knobs; rely on max-length truncation.
   - Rationale: the new container is array-based and salvage parsing is sufficient; fewer moving parts and fewer template-dependent heuristics.

## Risks / Trade-offs

- **[Risk] CoordJSON is not strict JSON →** Mitigation: explicitly document the two-layer contract (CoordJSON model-facing; strict JSON pipeline-facing) and enforce conversion at a single entrypoint in parsing utilities.
- **[Risk] Truncated rollouts break parsing →** Mitigation: salvage mode keeps valid prefix, drops only the trailing incomplete record; do not attempt JSON repair.
- **[Risk] Ordering-ablation ambiguity with whitespace/tokenization →** Mitigation: deterministic serializer for cooked targets; token-aligned scanning for rollouts that does not decode+re-encode the whole string.
- **[Risk] Poly shape constraints mismatch with existing geometry utilities →** Mitigation: reuse existing poly validation rules; encode the constraints in the spec scenarios and unit tests.
- **[Risk] Ecosystem/tooling confusion (“JSON-only” vs CoordJSON) →** Mitigation: update docs and inspection tooling; keep strict JSON as the internal representation used everywhere after conversion.
