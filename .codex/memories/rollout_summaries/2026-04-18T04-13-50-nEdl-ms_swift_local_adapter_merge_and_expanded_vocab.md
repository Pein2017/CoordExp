thread_id: 019d9ecb-81f7-7e82-9ac2-610d5868e663
updated_at: 2026-04-18T10:18:31+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/18/rollout-2026-04-18T04-13-50-019d9ecb-81f7-7e82-9ac2-610d5868e663.jsonl
cwd: /data/CoordExp
git_branch: codex/raw-text-continuity-probe

# Investigated local ms-swift adapter loading/merge behavior and whether it can carry CoordExp `<|coord_*|>` expanded vocabulary

Rollout context: The user first asked to inspect the local `ms-swift` library only (no web search), to understand how its native adapter support works, how it merges checkpoints, and whether it can support CoordExp’s expanded coordinate vocabulary (`<|coord_*|>`). The investigation used the local `/data/ms-swift` source plus the CoordExp repo, and focused on the actual load/merge/export paths rather than theory.

## Task 1: Inspect local ms-swift adapter loading / merge semantics and CoordExp vocabulary implications

Outcome: success

Preference signals:

- The user explicitly corrected scope with: “请浏览`ms-swift`在本地的库，不需要联网搜索” -> future similar research should default to local source inspection first and avoid web search unless the user later reopens the scope.
- The user’s repeated emphasis on “原生支持的方式是怎么`合并`的” and whether it can support “`<|coord_*|>`的expanded vocabulary” -> future agents should treat merge semantics and vocabulary preservation as the two core questions when adapter-loading changes touch CoordExp.

Key steps:

- Confirmed local `ms-swift` installation and version from `/data/ms-swift`.
- Inspected `swift/llm/export/merge_lora.py` and `swift/llm/utils.py` to see how native export merges LoRA/DoRA into base weights and saves processor/tokenizer assets.
- Inspected `swift/llm/model/register.py` to verify `new_special_tokens` support: tokenizer is extended via `tokenizer.add_special_tokens(...)`, then model embeddings are resized with `resize_token_embeddings(...)`, and `vocab_size` is updated.
- Inspected `swift/llm/argument/base_args/base_args.py` and `swift/llm/train/tuner.py` to confirm that adapter loading/resume paths already exist natively in ms-swift via `model` + `adapters` / `resume_from_checkpoint`.
- Cross-checked CoordExp’s own merge flow (`scripts/merge_coord.sh`) and coord-offset code to determine where native ms-swift is sufficient and where repo-specific post-processing is still required.

Failures and how to do differently:

- The investigation showed that native ms-swift merge is sufficient for standard LoRA/DoRA delta merge, but not for CoordExp’s custom `coord_offset_adapter` without extra repo-side injection. Future changes should not assume `swift export --merge_lora true` alone will preserve CoordExp’s custom coord-offset behavior.
- The repo’s current custom merge script already proves the gap: it has to copy `coord_tokens.json` and then inject coord offsets after the native merge step. Future agents should treat that as the authoritative compatibility boundary.

Reusable knowledge:

- Local ms-swift already supports loading a base model plus one or more adapters natively through the `model` + `adapters` contract; it is not necessary to invent a new `base_model` abstraction for the core path.
- Native ms-swift merge/export behavior is: load base + adapters, `Swift.merge_and_unload(model)`, then `save_checkpoint(...)` to materialize a merged HF checkpoint together with processor/tokenizer artifacts.
- ms-swift can extend vocabulary with `new_special_tokens` and will resize embeddings if the tokenizer grows.
- ms-swift checkpoint arg restoration includes `new_special_tokens`, so adapter/checkpoint metadata can carry vocabulary-expansion intent when saved in the ms-swift style.
- CoordExp’s current `coord_offset_adapter` is not a standalone token-expansion mechanism; it assumes the coord tokens already exist in the base vocab and then learns offsets for those ids.
- CoordExp’s native merge script already encodes the real boundary: standard LoRA merge is handled by ms-swift, but CoordExp-specific `coord_offset` injection is a separate post-step.

References:

- [1] ms-swift merge path: `/data/ms-swift/swift/llm/export/merge_lora.py`
  - `prepare_model_template(args)` -> `Swift.merge_and_unload(model)` -> `save_checkpoint(...)`
- [2] ms-swift vocab extension path: `/data/ms-swift/swift/llm/model/register.py`
  - `new_special_tokens` -> `tokenizer.add_special_tokens(...)` -> `resize_token_embeddings(...)`
- [3] ms-swift adapter load path: `/data/ms-swift/swift/llm/train/tuner.py`
  - if `args.resume_from_checkpoint or args.adapters`, it loads with `tuner.from_pretrained(..., is_trainable=True)`
- [4] ms-swift adapter fields / checkpoint restoration: `/data/ms-swift/swift/llm/argument/base_args/base_args.py`
  - `adapters: List[str]`, `_check_is_adapter`, `load_args_from_ckpt()` restores `new_special_tokens`
- [5] CoordExp merge boundary: `scripts/merge_coord.sh`
  - native `swift export --merge_lora true` followed by copying `coord_tokens.json` and injecting `coord_offset` tensors
- [6] CoordExp coord-offset contract: `docs/training/STAGE1_OBJECTIVE.md`
  - coord-offset learns only offsets over `<|coord_0|>.. <|coord_999|>` and assumes the coord token ids already exist in vocab
- [7] CoordExp config note: `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`
  - base checkpoint is explicitly described as already having expanded `<|coord_*|>` vocab

## Task 2: Re-open local ms-swift source to verify native adapter/resume paths and checkpoint/vocab propagation

Outcome: success

Preference signals:

- The user’s second request (“请帮我了解一下`ms-swift`目前这种原生支持的方式是怎么`合并`的，是否可以支持我`<|coord_*|>`的expanded vocabulary 吗？”) indicates they want the merge path understood in concrete artifact terms, not just in abstract API terms.
- The user stayed on the same topic and narrowed it to vocabulary expansion, implying that for similar future questions, the next agent should proactively check whether token expansion is carried by the checkpoint metadata or only by the base model.

Key steps:

- Re-read the local ms-swift loader/argument code to see how checkpoint directories are inferred and how adapter checkpoints are restored.
- Inspected `swift/llm/model/register.py` for `new_special_tokens` and vocab resizing behavior.
- Inspected `swift/llm/utils.py` `save_checkpoint(...)` to confirm what gets copied when saving merged weights.
- Reconfirmed that CoordExp’s own codebase already contains a special merge/injection script for `coord_offset_adapter`, which is a strong indicator that native merge is not enough for that custom adapter.

Failures and how to do differently:

- A naive assumption that “ms-swift merge handles everything” would be wrong for CoordExp’s custom `coord_offset_adapter`. Future agents should always check whether a custom adapter is represented as standard LoRA weights or as repo-specific `modules_to_save`/hook logic before promising a one-step merge.
- For CoordExp, the existence of `coord_tokens.json` and the explicit `coord_offset` injection script means “expanded vocabulary” and “adapter merge” are separate concerns; don’t conflate them.

Reusable knowledge:

- `ms-swift` native merge/export preserves processor/tokenizer and can include newly added special tokens when they are part of the checkpoint/model load path.
- The local code path for `new_special_tokens` is in the base model/tokenizer loader, not in the merge function itself.
- If the base checkpoint already has the expanded coord vocabulary, adapter hot-loading is plausible; if not, `coord_offset_adapter` has no valid id range to bind to.
- CoordExp’s current architecture treats expanded vocab as a base-model contract, not an adapter-only contract.

References:

- `swift/llm/utils.py:222-258` — `save_checkpoint(...)`
- `swift/llm/model/register.py:764-776` — tokenizer extension and `resize_token_embeddings(...)`
- `swift/llm/argument/base_args/base_args.py:231-255` — checkpoint args restoration includes `new_special_tokens`
- `scripts/merge_coord.sh:106-145` — native merge followed by coord-token metadata copy and coord-offset injection
- `src/infer/checkpoints.py` / `src/infer/engine.py` (CoordExp) — inference code already has explicit coord-offset adapter handling, which reinforces that custom adapter logic is not covered by vanilla merge alone

