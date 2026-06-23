# Coverage Ledger Auxiliary Loss Design Audit

Date: 2026-06-23

Reviewed spec:
`/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md`

Context source of truth:
`/data/CoordExp/.worktrees/ledger-auxiliary-loss/research/ideas/ledger-auxiliary-loss/`

Scope: read-only design audit. No feature implementation, source edit, config edit, test edit, or training run was performed. The only local write is this audit markdown.

User clarification applied: the feature must update the detection template selection and use the existing ending wrapper tokens. In practical terms, the design should target a closed compact template that includes `<|object_ref_end|>` and `<|box_end|>` and should not invent new special tokens.

## Verdict

hold

The idea is plausible, and the research notes are internally coherent about a strict, training-only, single-image, same-forward auxiliary loss. The current design spec is not ready to implement as written because its central state target depends on `<|box_end|>`, while the spec simultaneously says not to change template bytes and the active Stage-1 compact config does not emit ending wrapper tokens. Two additional implementation-critical seams are also underspecified: where the trainable projection heads live, and how Qwen3-VL same-forward hidden states plus post-merge visual embeddings are captured without editing upstream HF code or recomputing vision features.

## Prioritized Findings

### P0 - The row-completion target is impossible under the active `compact` route unless the spec requires the closed ending-wrapper template

Evidence:

- The spec makes row completion a required coverage state at the row `<box_end>` position: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:56`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:144-162`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:173-184`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:342-347`.
- The same spec currently declares as a non-goal: no detection template bytes, tokenizer behavior, or special-token allocation changes: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:62-67`.
- The active compact Stage-1 config uses `detection_template.id: compact`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml:36-37`.
- That active config allocates adapter rows only for `<|object_ref_start|>` and `<|box_start|>`, not `<|object_ref_end|>` or `<|box_end|>`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml:49-56`.
- The template contract confirms `compact` has `include_object_ref_end=False` and `include_box_end=False`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/detection/template_contracts.py:82-89`.
- Existing closed variants already exist: `compact_box_closed` includes `<|box_end|>`, and `compact_object_box_closed` / `compact_object_box_closed_lines` include both `<|object_ref_end|>` and `<|box_end|>`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/detection/template_contracts.py:90-121`.
- Current docs describe the exact distinction: active compact trains 1000 coordinate rows plus `<|object_ref_start|>` and `<|box_start|>`, while `compact_object_box_closed` trains 1004 rows including `<|object_ref_end|>` and `<|box_end|>`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/training/STAGE1_OBJECTIVE.md:238-242`.
- Tests enforce the distinction: `BOX_END_TOKEN` is absent for `compact` and present for box-closed templates: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/tests/test_teacher_forcing_target_builder.py:310-321`.

Impact:

If an implementer follows the current text literally, strict sidecar construction will fail every active `compact` sample because there is no `<|box_end|>` token to bind. If they infer completion from the final coordinate token instead, the implementation violates the stated idea and makes a misleading state target. If they silently switch the template, they change training bytes, token rows, and artifact semantics outside the spec's stated non-goals.

Required amendment:

Replace the "no detection template bytes" constraint with a narrower constraint:

- Do not add new special tokens and do not globally change production defaults.
- The v0 ledger pilot must use an existing closed compact template with ending wrapper tokens, preferably `compact_object_box_closed` unless there is a deliberate reason to omit `<|object_ref_end|>`.
- The smoke/prod pilot config must set the closed template explicitly and include adapter rows for `<|object_ref_end|>` and `<|box_end|>`.
- The sidecar builder must assert that each object row has exactly one `<|box_end|>` control span and, if using `compact_object_box_closed`, exactly one `<|object_ref_end|>` control span.

Minimum verification before implementation is considered launchable:

- Config parse for the closed-template ledger smoke config.
- Render/tokenize one closed-template sample and assert per-object control spans include the ending wrapper tokens.
- Existing compact-vs-closed target-builder tests remain explicit.
- A new sidecar-builder test fails on plain `compact` and passes on the selected closed template.

### P1 - Trainable projection/head ownership is underspecified and likely to be implemented outside the optimizer/checkpoint path

Evidence:

- The spec introduces trainable projections through `state_projection` and `object_projection`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:360-364`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:381-383`.
- The proposed integration point is `TrainerLossBridge`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:189-235`.
- `TrainerLossBridge` is a plain class, not an `nn.Module`, and currently has no parameter ownership surface: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/training/bridge/loss_bridge.py:65-79`.
- The teacher-forcing mixin instantiates a new `TrainerLossBridge()` inside `compute_loss` every call: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/trainers/metrics/teacher_forcing.py:24-44`.
- Current Stage-1 config freezes the vision tower and aligner: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml:106-110`.

Impact:

A natural implementation would put projection modules inside the bridge/helper. In the current trainer path, that would either recreate the modules each step, keep them outside `model.named_parameters()`, omit them from the optimizer, omit them from checkpoints, or place them on the wrong device/dtype. The loss could appear to compute while only LoRA/model parameters move through fixed or reset random heads, which would make the experiment uninterpretable.

Required amendment:

The spec needs an explicit `CoverageLedgerHead(nn.Module)` ownership contract:

- Where it is registered before optimizer construction.
- How its parameters are named, device-moved, dtype-managed, and checkpointed.
- Whether it is attached to the model wrapper, trainer, or an objective module registry.
- How it interacts with LoRA/module saving and frozen vision/aligner settings.
- What happens when the feature is disabled.
- Whether head LR/weight decay follows the main optimizer or gets an explicit param group.

Minimum verification:

- Unit test that enabled config exposes ledger head parameters in `named_parameters()`.
- Unit test that those parameters are present in optimizer param groups.
- Tiny backward/optimizer-step test showing at least one ledger head parameter changes.
- Checkpoint save/load test proving the ledger head state is persisted and restored.

### P1 - Qwen3-VL same-forward capture is not concrete enough for the installed model behavior

Evidence:

- The spec says the helper injects `output_hidden_states=True`, calls the model once, and exposes final hidden states plus projected `image_embeds`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:211-225`.
- Repo standards forbid editing upstream HF model files: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/standards/UPSTREAM.md:56-64`.
- Qwen guidance notes that full logits are required for teacher forcing, and `logits_to_keep` slices hidden states before the LM head: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/standards/upstream/QWEN_VL.md:62-68`.
- In the installed Qwen3-VL model, `get_image_features` returns `image_embeds`: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1050-1064`.
- `Qwen3VLModel.forward` scatters `image_embeds` into text embeddings but returns only `last_hidden_state`, `past_key_values`, and `rope_deltas` in this local version: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1106-1143`, `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1223-1239`.
- `Qwen3VLForConditionalGeneration.forward` applies `lm_head` and returns loss/logits/past/rope, not hidden states or image embeddings: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1314-1373`.

Impact:

Simply setting `output_hidden_states=True` is not enough in the current environment. Without a precise CoordExp-owned seam, implementers may patch HF code, recompute vision features, accidentally call vision twice, or obtain hidden states from a different forward path than the logits. Any of those would violate the design's core same-forward correctness claim.

Required amendment:

The spec should prescribe one supported capture path, for example:

- A CoordExp-owned wrapper or hook that captures the single call to `model.model.get_image_features` and the post-merge `image_embeds`.
- A route that obtains final language hidden states from the same forward used to compute logits, such as a direct lower-level call plus `lm_head`, if that is the only stable way in the installed Qwen version.
- Explicit call-count and parity requirements: one vision call, full logits identical to baseline forward for the same inputs, final hidden states aligned with logits sequence length, and captured visual embedding count matching placeholder/grid count.
- Compatibility checks for autocast, gradient checkpointing, flash-attn/eager attention, DDP, and the current `TrainerLossBridge` full-logits guard.

Minimum verification:

- Fake-Qwen contract test that proves the helper captures hidden states and visual embeddings without forwarding sidecars.
- Real tiny Qwen parity probe on one batch: baseline logits vs capture-helper logits match within tolerance, `get_image_features` is called exactly once, and captured `image_embeds` length equals the number of post-merge image placeholders.

### P1 - The objective profile/baseline choice is underspecified and can confound interpretation

Evidence:

- The spec's example uses `objective.profile: hard_sft`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:99-103`.
- The active compact Stage-1 config uses `objective.profile: pure_valid_set_marginal` with several teacher-forcing terms explicitly disabled: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml:60-79`.
- The repo docs list the public Stage-1 research teacher-forcing route and active compact config route: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/training/STAGE1_OBJECTIVE.md:55-76`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/training/STAGE1_OBJECTIVE.md:180-185`.
- The schema treats `hard_sft` and valid-set teacher-forcing profiles differently: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/config/schema.py:3992-4062`.

Impact:

If the pilot changes both the template route and the objective profile, then any observed effect is ambiguous. A hard-SFT-plus-ledger run is not directly comparable to the current active `pure_valid_set_marginal` route. This is not necessarily wrong, but the spec must say which baseline the mechanism is meant to augment and how comparison runs are paired.

Required amendment:

Choose one primary v0 comparison shape:

- Closed-wrapper hard-SFT baseline vs closed-wrapper hard-SFT plus ledger, if the intended question is "can CE plus ledger learn the closed compact route?"
- Closed-wrapper `pure_valid_set_marginal` baseline vs closed-wrapper `pure_valid_set_marginal` plus ledger, if the intended question is "does ledger improve the active Stage-1 research objective?"

For quality claims, the spec should require a same-template, same-profile, same-data, same-seed baseline without ledger. The resolved config diff should be limited to ledger enablement, ledger weights, and run identity.

Minimum verification:

- Config-load tests for both selected ledger and paired baseline smoke configs.
- Resolved-config diff check that confirms the comparison pair does not accidentally change template/profile/data/packing/batch settings outside the intended closed-wrapper move.

### P2 - Metric producer semantics are easy to implement incorrectly with the existing `MetricEvent` API

Evidence:

- The spec defines several `weighted_mean` metrics and describes numerator/denominator as raw loss sums and valid counts: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:416-465`.
- The current metric API's `weighted_mean_event(key, value, weight)` stores `numerator=value * weight` and `denominator=weight`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/metrics/events.py:207-255`.
- Metric docs state that zero-denominator ratio/weighted-mean metrics are omitted from flattened logs: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/training/METRICS.md:180-183`.
- Existing tests cover ratio, weighted mean, and zero-denominator behavior: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/tests/test_metric_events.py:50-77`.

Impact:

If an implementer passes a raw summed BCE as `value` and the valid count as `weight`, the reducer will multiply by count again and inflate the metric. AUC and coverage-only metrics can also become misleading if single-class or zero-valid batches are logged as 0.0 instead of omitted.

Required amendment:

Add concrete producer examples for each metric family:

- For a batch BCE sum and valid count, either pass `value=sum/count, weight=count` to `weighted_mean_event`, or construct a direct `MetricEvent` with explicit numerator and denominator.
- For AUC, log only when both classes are present; use the number of comparable positive-negative pairs as the denominator if representing exact rank-pair AUC.
- Keep region-anchor AUC/accuracy out of the contract, as the current spec already recommends.
- State that absent metrics are omitted, not coerced to zero.

Minimum verification:

- Synthetic two-batch reducer test proving every new `teacher_forcing/ledger/*` flat value equals the hand-computed value.
- Zero-valid and single-class tests proving metrics are omitted.
- Contract test updating the canonical teacher-forcing metric key list only for metrics that are guaranteed to appear under valid denominators.

### P2 - Sidecar positions need a typed derivation contract rather than implicit control-span parsing

Evidence:

- The spec requires `prompt_end_position`, per-row `box_start_position`, and `box_end_position`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:144-162`.
- Current tokenized object metadata has `bbox_start_span`, `bbox_span`, `coord_spans`, `separator_span`, and generic `control_spans`, but no first-class `box_end_span`: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/detection/tokenization.py:55-73`.
- Rendering creates control spans according to template contract flags, including optional object and box end spans: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/src/detection/template.py:719-790`.
- Research notes say structured object metadata should be the source of truth and token parsing should be fallback only: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/research/ideas/ledger-auxiliary-loss/overview.md:56-63`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/research/ideas/ledger-auxiliary-loss/discussion.md:47-75`.

Impact:

The implementation can easily pick the wrong structural span from the generic `control_spans` list, especially across geometry-first, description-first, and line-separated closed templates. Prompt-end can also be off by one if derived after assistant prefixing, padding, or packing changes.

Required amendment:

Define exact sidecar extraction rules:

- Require a template contract with `include_box_end=True`.
- Locate exactly one `box_end` control span per object entry by label, not by token-string search over raw input ids.
- Locate `box_start` through existing `bbox_start_span`.
- Define `prompt_end_position` as the final prompt-side token index before the first supervised assistant/object token, with a test fixture showing the raw token sequence.
- Reject packed/static-padding-free batches until offset rewriting is proven, matching existing packing guidance.

Minimum verification:

- Render/tokenize tests for selected closed template in geometry-first and description-first order.
- Line-separated variant test if `_lines` is allowed.
- Packed/static-padding-free rejection test remains explicit.

### P2 - Smoke and artifact contracts are directionally good but not yet reproducible enough

Evidence:

- The spec lists ledger artifacts and debug outputs: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:466-529`.
- It also requires all-128 preflight and production-launch gates: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:513-529`, `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md:638-682`.
- Training artifact docs require resolved configs, effective runtime info, experiment manifests, data provenance, and rank-0 writer discipline: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/ARTIFACTS.md:208-303`.
- Artifact docs also warn not to fabricate `pipeline_manifest` for Stage-1 runs that do not produce it: `/data/CoordExp/.worktrees/ledger-auxiliary-loss/docs/ARTIFACTS.md:261-266`.

Impact:

The artifact list is useful, but a future implementation can still produce a non-comparable smoke packet if the spec does not name the exact smoke config path, selected source row contract, digest fields, seed policy, baseline pairing, and manifest pointers. "Overfit signal" can become ambiguous if step count, closed-template baseline, and source-data selection are not fixed.

Required amendment:

Add a v0 smoke recipe:

- Exact ledger smoke config path and paired no-ledger baseline config path.
- Selected 128-row source manifest with source JSONL path, source JSONL digest, row indices, sample ids if available, seed, template id, tokenizer/model id, processor resize policy, and image-grid metadata version.
- Step budget, batch size, gradient accumulation, and effective batch size.
- Required artifact roots and manifest pointers for `ledger/selected_samples.json`, `ledger/alignment_debug.jsonl`, overlays, metric reducer output, and resolved config.
- Explicit statement that smoke evidence is tiny/proxy and cannot be presented as full validation.

Minimum verification:

- Preflight command that builds the selected-sample manifest and sidecars without training.
- Artifact-structure test that validates all ledger debug files and manifest pointers after a tiny run.

## Open Questions And Decisions

1. Which exact closed template is the v0 target: `compact_object_box_closed` or `compact_object_box_closed_lines`? Given the user clarification and existing token-adapter docs, `compact_object_box_closed` is the lowest-friction default unless line breaks are needed for debugging or model behavior.
2. Is the intended v0 comparison against hard SFT or against the active `pure_valid_set_marginal` Stage-1 objective? The spec should not leave this implicit because it changes interpretation.
3. Where will `CoverageLedgerHead` be registered so it is optimized, checkpointed, and restored? This is a design blocker, not an implementation detail.
4. What exact same-forward capture seam will be supported for Qwen3-VL in this repo without editing HF files? The spec should name one route and its parity tests.
5. Should `object_projection(e.detach())` be trainable while visual embeddings are detached? The current text says yes implicitly, but the gradient path should be stated: no gradient into visual encoder/aligner through `e`, gradient into ledger projection heads and language-side hidden states.
6. Are object rows with empty or degenerate mapped visual regions always hard failures, or are they counted and excluded? Research notes lean hard-fail for malformed/zero-object cases; the visual-region contract should say what happens for valid tiny boxes that map to zero cells before clamping.
7. Which artifacts are normative enough to update `docs/ARTIFACTS.md` and `docs/training/METRICS.md` during implementation? If ledger metrics/artifacts are experimental-only, the spec should state the promotion threshold.

## Suggested Spec Amendments

1. Change the top-level summary from "no inference/template/tokenizer/default changes" to "training-only, no inference behavior change, no new special tokens; v0 uses an existing closed compact template with ending wrapper tokens."
2. In the proposed YAML, set the template explicitly to the selected closed variant and include adapter rows for `<|object_ref_end|>` and `<|box_end|>`.
3. Add a `CoverageLedgerHead` section that defines module ownership, optimizer integration, save/load, dtype/device, DDP, and disabled behavior.
4. Replace "inject `output_hidden_states=True`" with a concrete Qwen3-VL capture design and a required parity probe against normal forward logits.
5. Add a paired-baseline section requiring same closed template, same profile, same data, same seed, and no ledger as the comparator for any smoke or quality claim.
6. Add metric producer examples that match `MetricEvent` reducer semantics and explicitly omit zero-denominator/single-class metrics.
7. Add sidecar extraction rules tied to structured rendered/tokenized object metadata and exact control-span labels, with token-string parsing only as a diagnostic fallback.
8. Add the v0 smoke recipe and manifest schema fields needed for reproducibility.
9. Keep the existing fail-fast stance for single-image, no-video, no-packing, no-resize-drift, and full-logits requirements; those are aligned with repo conventions.

## Positive Notes

- The training-only boundary is correct and fits the repo's config-first norm.
- The design correctly avoids editing upstream HF model files.
- The single-image, strict, no-video, no-packing, full-logits, and detached-visual defaults are appropriate for a first mechanism test.
- The proposal to keep region-anchor metrics out of AUC/accuracy is good; it avoids generating attractive but semantically weak metrics.
- The audit/debug artifact direction is strong, especially overlaying mapped region cells and logging selected-sample manifests.

## Verification Performed

Read-only checks and source inspection only:

- Reviewed the target design spec, research overview, and research discussion notes.
- Checked current Stage-1 config, schema, bridge, trainer mixin, objective runner, metric events, template contracts, tokenization, rendering, and relevant tests.
- Checked repo docs for Stage-1 objective routes, artifacts, metrics, packing, upstream Qwen guidance, and data coordinate contracts.
- Inspected installed local Qwen3-VL implementation under `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/` for forward-output and image-embedding behavior.

No tests, training, or smoke runs were executed because this was requested as a read-only design audit. No source/config/test files were changed.
