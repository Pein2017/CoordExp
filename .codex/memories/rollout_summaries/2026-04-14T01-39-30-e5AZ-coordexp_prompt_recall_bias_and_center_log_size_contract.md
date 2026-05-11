thread_id: 019d89a4-c452-7f03-aec4-de65df06e569
updated_at: 2026-04-14T02:02:25+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/14/rollout-2026-04-14T01-39-30-019d89a4-c452-7f03-aec4-de65df06e569.jsonl
cwd: /data/CoordExp
git_branch: main

# Prompt-contract cleanup and recall-oriented prompt rewrite across shared base + variants

Rollout context: The user first asked for explanations of the training/data-preprocessing path for `configs/stage1/profiles/2b/center_log_size_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml` + `scripts/train.sh`, then narrowed into prompt semantics (`bbox_format`, `u(w)/u(h)`, prompt text, and finally prompt edits). The meaningful durable work happened in the prompt-contract area under `/data/CoordExp`, especially `src/config/prompts.py` and `src/config/prompt_variants.py`.

## Task 1: Explain training/data preprocessing and raw text path

Outcome: success

Preference signals:
- The user repeatedly asked for concrete, code-grounded examples of the “raw text just before tokenization” and later asked “Give me real demo,” indicating they want actual repo-backed examples rather than abstract descriptions or synthetic placeholders.
- The user asked follow-up clarification questions (`How does bbox_format change the final raw text?`, `Tell me how the u(w) and u(h) are computed`, `What is the system/user prompt for the center bbox expression, how does it differ?`) indicating they want the exact prompt and geometry contract spelled out precisely, not loosely paraphrased.

Reusable knowledge:
- For this repo, `scripts/train.sh` is config-only: it resolves the YAML, runs JSONL validation prechecks, then launches `torchrun -m src.sft`.
- Stage-1 training uses a strict single-dataset JSONL contract with `BaseCaptionDataset.from_jsonl(...)`; `src/datasets/dense_caption.py` applies max-pixels enforcement, object ordering, bbox conversion, and coord-token annotation before `template.encode(...)`.
- `src/datasets/builders/jsonlines.py` is the point where final assistant target text is built as strict CoordJSON (via `dumps_coordjson`) and then serialized into the chat payload.
- `bbox_format` changes the model-facing geometry meaning: `xyxy` means `[x1, y1, x2, y2]`; `center_log_size` means `[cx, cy, u(w), u(h)]` using the log-size chart in `src/common/geometry/bbox_parameterization.py`.

Failures and how to do differently:
- The repo snapshot did not include the expected raw COCO `.jsonl` files at the initially guessed path, so the agent had to fall back to fixture-backed examples from `public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl`.
- The environment lacked `yaml` and `torch` in ad hoc Python imports, so direct runtime reproduction of some config/template code paths was blocked; use source inspection and existing tests instead.

References:
- [1] `scripts/train.sh`: environment-only launcher; prechecks `custom.train_jsonl`, `custom.val_jsonl`, `offline_max_pixels`, then runs `torchrun --nproc_per_node=... -m src.sft --config ...`.
- [2] `src/datasets/dense_caption.py`: `_enforce_max_pixels`, `_apply_bbox_format`, `_maybe_annotate_coord_tokens`, `_encode_rendered_record`, `_render_prepared_record`.
- [3] `src/datasets/builders/jsonlines.py`: `build_many`, `_build_group_entry`, `_format_points`, `_render_json_text`.
- [4] `src/config/prompts.py`: prompt assembly for `center_log_size` vs `xyxy`, and user/system prompt rendering.
- [5] `src/common/geometry/bbox_parameterization.py`: `xyxy_norm1000_to_center_log_size_bins`, `log_size_encode`, `BBOX_SIZE_FLOOR = 1/1024`.
- [6] `tests/test_chat_template_regression.py`, `tests/test_bbox_parameterization.py`, `tests/test_prompt_variants.py`: contract checks for chat framing, bbox conversion, and prompt rendering.

## Task 2: Explain/modify bbox_format behavior and center-log-size math

Outcome: success

Preference signals:
- The user’s follow-up questions were very specific and math-oriented, suggesting they want concise but exact derivations for geometry transforms rather than high-level prose.
- The user cared about how prompt wording maps to tokenized raw text, implying future explanations should bridge configuration → text → tokenization.

Reusable knowledge:
- `BBOX_SIZE_FLOOR = 1/1024` is a floor used to stabilize the log-size transform, not because the vocabulary is 0..999; quantization to 0..999 happens separately through `quantize_norm1000_slot`.
- `u(w)` and `u(h)` are computed by converting normalized xyxy width/height to log-size slots: `u(s) = (log(max(s, 1/1024)) - log(1/1024)) / -log(1/1024)`, then quantized to coord bins.
- The prompt text for `center_log_size` explicitly states `[cx, cy, u(w), u(h)]` and `log(max(s, 1/1024))` in `src/config/prompts.py`.

Failures and how to do differently:
- Direct imports of `src.common.geometry.bbox_parameterization` via package paths hit repo dependency issues because the package tree imports `torch`; a direct file-level import workaround was needed for pure math verification.

References:
- [1] `src/common/geometry/bbox_parameterization.py`: `BBOX_SIZE_FLOOR = 1.0 / 1024.0`, `log_size_encode`, `xyxy_norm1000_to_center_log_size_bins`.
- [2] `docs/data/CONTRACT.md`: canonical raw bbox vs model-facing parameterization explanation.
- [3] `docs/training/STAGE1_OBJECTIVE.md`: Stage-1 `center_log_size` V1 experiment notes and loss constraints.

## Task 3: Remove the shared empty-object fallback from prompts

Outcome: success

Preference signals:
- The user explicitly objected to the fallback text: “Help me remove the `If none, return {"objects": []}.` in all prompts since it raises the possibility to output enter list,” which strongly indicates they prefer prompt wording that does not encourage empty outputs.
- The user also reacted to an editing-method issue (`apply_patch was requested via exec_command. Use the apply_patch tool instead of exec_command.`), indicating a clear preference for using the dedicated patch tool for edits rather than shelling out an inline patch command.

Reusable knowledge:
- The empty-object fallback came from one shared constant in `src/config/prompts.py` (`PRIOR_RULES`), so removing it there updates all coord-token prompt variants consistently.
- A repo-wide search after the edit confirmed no remaining matches for the exact fallback phrase in `src`, `tests`, or `docs`.

Failures and how to do differently:
- The first edit attempt used `exec_command` with `apply_patch`; the user corrected this. Future edits should use the dedicated apply-patch mechanism directly when available.

References:
- [1] `src/config/prompts.py`: `PRIOR_RULES` originally contained `If none, return {"objects": []}.` and was updated to remove it.
- [2] Verification search: `rg -n "If none, return \{\"objects\": \[\]\}" src tests docs -S` returned no matches after the change.

## Task 4: Broaden prompt wording to be more recall-oriented across variants

Outcome: success

Preference signals:
- The user asked, “For all my prompts, any recommendation to become more concise or more encouraging to increase the recall rate,” then later, “Please update the prompts based on your suggestions over all the variants,” indicating a preference for prompt wording that is intentionally recall-biased across the entire prompt stack, not just one isolated prompt.
- The user wanted the change applied “for all my prompts,” which suggests that future prompt edits should be propagated consistently through shared base prompts and all variants, not limited to one branch.
- The user specifically accepted the idea of making the prompts “more concise or more encouraging,” so future prompt rewrites should prioritize tighter wording and inclusion bias instead of extra cautionary/negative phrasing.

Reusable knowledge:
- The prompt surface is concentrated in two files: `src/config/prompts.py` for shared base prompt construction and `src/config/prompt_variants.py` for COCO/LVIS variant suffixes/overrides.
- The shared prompt assembly path is:
  - `_prompt_fragments(...)` for bbox-format/object-field-order dependent text,
  - `build_dense_system_prompt(...)` / `build_dense_user_prompt(...)` for final prompt strings,
  - `get_template_prompts(...)` as the public API.
- The COCO and LVIS variants are where recall-oriented policy language belongs for dataset-specific behavior; the shared base should stay concise and contract-focused.
- The prompt tests in `tests/test_prompt_variants.py` encode expectations about prompt content and were updated to reflect the new wording.

Failures and how to do differently:
- The repo already had a contradiction between the shared `desc="unknown"` fallback and the COCO closed-class policy saying to choose the closest canonical class. That contradiction was a high-signal cleanup target; future prompt changes should scan for cross-layer conflicts before adding more wording.
- The first pass exposed that `center_log_size` user-side wording was longer than necessary; the math belongs in the system prompt, while the user prompt can stay shorter and more action-oriented.
- A background graph rebuild (`graphify.watch._rebuild_code`) ran long after the prompt edits; if a future agent needs only prompt semantics, it should avoid kicking off expensive rebuilds unless they are actually needed for the task.

References:
- [1] `src/config/prompts.py` changes:
  - removed `- If uncertain, set desc="unknown" and give the reason succinctly.` from the shared system prefix,
  - added `- Prefer including a clearly visible, localizable instance rather than omitting it because the boundary is slightly uncertain.`,
  - shortened `center_log_size` user rule to `Use bbox_2d as [cx, cy, u(w), u(h)].`
- [2] `src/config/prompt_variants.py` changes:
  - COCO variant now says localize visible instances, including small/partially occluded ones when still localizable,
  - replaced the precision-biased `If you cannot localize a single instance, omit it.` with inclusion-biased wording,
  - LVIS stage-1/stage-2 variants now emphasize including verified/localizable instances and preferring inclusion over omission when grounded geometry is available.
- [3] `tests/test_prompt_variants.py` updates:
  - checks for the new recall-oriented phrasing,
  - stops expecting the old full center-log-size formula in every prompt,
  - preserves the bbox-format rendering assertions.
- [4] Prompt-variant registry handles: `coco_80`, `lvis_stage1_federated`, `lvis_stage2_federated` in `src/config/prompt_variants.py`.

Overall takeaway: the user prefers prompt and contract edits to be applied consistently across the whole prompt stack, wants concise but recall-biased wording, and dislikes prompt text that encourages empty outputs or overly cautious omission. When modifying prompts in this repo, the safest high-value pattern is to keep the geometry/JSON contract strict, remove contradictory fallback language, and make the user-side instructions explicitly favor inclusion of clearly visible, localizable instances.
