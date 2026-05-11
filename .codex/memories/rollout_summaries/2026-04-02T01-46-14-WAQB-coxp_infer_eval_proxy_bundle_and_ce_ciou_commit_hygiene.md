thread_id: 019d4bde-a156-7b83-b75c-347e2f82f0cd
updated_at: 2026-04-02T03:21:21+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/02/rollout-2026-04-02T01-46-14-019d4bde-a156-7b83-b75c-347e2f82f0cd.jsonl
cwd: /data/CoordExp
git_branch: main

# The rollout stabilized a COCO+LVIS-proxy evaluation workflow, added a reusable infer/eval skill, and then committed the work in clean logical chunks.

Rollout context: The user first asked to verify docs and design an eval path for a COCO-1024 run trained with an LVIS-proxy Stage-1 checkpoint, then later asked to package the infer→score→evaluate process as a reusable skill, and finally asked for proper git hygiene / commits.

## Task 1: Design and verify COCO+LVIS-proxy inference/evaluation

Outcome: success

Preference signals:

- The user asked to “Please refer to relevant documents about this design first.” and later focused on whether LVIS-expanded annotations should be evaluated separately from standard COCO -> the user prefers doc-grounded, contract-aligned design before implementation.
- The user repeatedly asked about “original coco GT version” vs “lvis expanded based on strict and plausible” -> this indicates they want benchmark metrics kept explicit and not silently mixed with proxy-expanded analysis.
- The user later asked “Can you help me check/verify whether all the `extended proxy` from `lvis` annotation have converted into the `coco 80` classes instead of bringing the new categories?” -> this suggests future eval work should explicitly audit that proxy labels stay inside COCO-80 and not assume it.

Key steps:

- The assistant read the canonical docs first: `docs/PROJECT_CONTEXT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/eval/README.md`, `docs/eval/CONTRACT.md`, `docs/eval/WORKFLOW.md`, and the relevant OpenSpec docs.
- The repo already had a strong proxy-supervision contract in `src/analysis/coco_lvis_missing_objects.py` and in `openspec/changes/add-lvis-coco-proxy-supervision/...`.
- The proxy JSONL artifact `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl` was checked directly and shown to keep all proxy objects in the COCO-80 label space while retaining LVIS provenance in metadata.
- The assistant introduced a proxy-view workflow that materializes three GT views from one scored artifact: `coco_real`, `coco_real_strict`, and `coco_real_strict_plausible`.

Failures and how to do differently:

- The evaluator initially risked treating COCO proxy artifacts like LVIS-federated data when `metrics: both` was used. The fix was to make the metrics routing explicitly preserve COCO behavior for COCO-like proxy artifacts instead of forcing LVIS backfill.
- A first attempt to validate the training config through the full training loader hit a local DeepSpeed/device-map environment guard; the safer verification path was to inspect the resolved YAML inheritance before trying to build a training run object.

Reusable knowledge:

- In this repo, the official evaluation split for proxy-supervised COCO runs should be:
  - `coco_real` = benchmark headline,
  - `coco_real_strict` = COCO + strict proxy analysis,
  - `coco_real_strict_plausible` = COCO + strict + plausible analysis.
- The proxy export contract keeps LVIS provenance fields like `lvis_category_name` and `lvis_category_id`, but the emitted class labels for proxy objects stay in the COCO-80 space via `mapped_coco_category_name` / `mapped_coco_category_id`.
- For this artifact, the direct audit found:
  - `4951` records,
  - `40478` objects,
  - `4205` LVIS proxy objects,
  - `0` label-space leaks outside COCO-80,
  - `strict=1219`, `plausible=2986`.

References:

- [1] `docs/eval/WORKFLOW.md` now documents the COCO+LVIS-proxy bundle flow and the meaning of the three GT views.
- [2] `configs/infer/coco_1024/val_200_lvis_proxy_merged.yaml`, `configs/postop/coco_1024/val_200_lvis_proxy_merged.yaml`, and `configs/eval/coco_1024/val_200_lvis_proxy_bundle.yaml` were created as the one-inference / one-score / three-view-eval path.
- [3] `src/eval/proxy_views.py`, `src/eval/proxy_eval_bundle.py`, `scripts/materialize_proxy_eval_views.py`, and `scripts/evaluate_proxy_detection_bundle.py` implement the reusable bundle workflow.
- [4] The direct audit on `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl` confirmed there were no leaked non-COCO categories in proxy objects.

## Task 2: Pack the infer→score→evaluate workflow into a reusable skill

Outcome: success

Preference signals:

- The user asked, “How many do you remember about the `run inference and run evaluation` process? Can you pack it as a reusable skill for later use?” -> they want this workflow reusable without re-deriving commands each time.
- They asked for the skill after the workflow had already been stabilized -> they prefer the durable command sequence to be captured as a local tool/skill, not repeated in chat.

Key steps:

- A new local skill was added at `.codex_config/pein/skills/coordexp-infer-eval-workflow/SKILL.md`.
- The skill captures the YAML-first production path:
  - `scripts/run_infer.py`
  - `scripts/postop_confidence.py`
  - `scripts/evaluate_detection.py`
  - `scripts/evaluate_proxy_detection_bundle.py`
- It also stores the key output artifacts and verification checks:
  - `gt_vs_pred.jsonl`
  - `gt_vs_pred_scored.jsonl`
  - `summary.json`
  - `confidence_postop_summary.json`
  - per-view `metrics.json`
  - `proxy_eval_bundle_summary.json`
- The skill explicitly distinguishes benchmark COCO (`coco_real`) from additive proxy-analysis views (`coco_real_strict`, `coco_real_strict_plausible`).

Failures and how to do differently:

- None material; the only caution is that the skill was intentionally kept narrow and should be updated only if the workflow changes again.

Reusable knowledge:

- The repo’s preferred infer/eval flow is YAML-first and artifact-driven; a single scored artifact can drive multiple GT views.
- For proxy-supervised COCO runs, the skill should remind future agents to:
  - infer once,
  - score once,
  - evaluate all GT views from the same scored artifact,
  - read the bundle summary for reporting.

References:

- [1] `.codex_config/pein/skills/coordexp-infer-eval-workflow/SKILL.md`.
- [2] The skill references the exact validated configs:
  - `configs/infer/coco_1024/val_200_lvis_proxy_merged.yaml`
  - `configs/postop/coco_1024/val_200_lvis_proxy_merged.yaml`
  - `configs/eval/coco_1024/val_200_lvis_proxy_bundle.yaml`
- [3] It also encodes the direct command shapes for `run_infer.py`, `postop_confidence.py`, `evaluate_detection.py`, and `evaluate_proxy_detection_bundle.py`.

## Task 3: Update Stage-1 training to CE + CIoU only

Outcome: success

Preference signals:

- The user asked to “use only the `CE` loss and `CIOU` loss and disable all the other `coord-reg` relevant loss like `soft_ce`, `w1` and ect.” -> this is a strong preference for a sharper, simpler objective when testing duplication collapse.
- They explicitly framed the purpose as “make those `prediction` sharp” and “see whether this improve the decoding performance and avoid `duplication collapse`” -> the training config should be treated as an experiment for sharpness/duplication behavior, not as a generic config cleanup.

Key steps:

- The profile `configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first_1024_lvis_proxy.yaml` was patched to disable coord soft-CE/W1 and enable bbox CIoU-only geometry support.
- The final resolved values verified through the YAML loader were:
  - `coord_soft_ce_w1.enabled = False`
  - `bbox_geo.enabled = True`
  - `bbox_geo.smoothl1_weight = 0.0`
  - `bbox_geo.ciou_weight = 1.0`
- The run naming was updated to reflect the CE+CIoU experiment.

Failures and how to do differently:

- A direct loader-based validation path hit the repo’s training-environment DeepSpeed/device-map guard. The safer validation method was to inspect the YAML inheritance with `load_yaml_with_extends` rather than building full training arguments.

Reusable knowledge:

- In this config family, `coord_soft_ce_w1.enabled: false` is the clean switch that makes coord-token training fall back to plain CE.
- `bbox_geo` can be used as the geometry-side auxiliary while keeping SmoothL1 at zero and CIoU active.
- `bbox_size_aux` was not enabled in this profile, so no size auxiliary loss was introduced.

References:

- [1] `configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first_1024_lvis_proxy.yaml`.
- [2] Inherited base: `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`.
- [3] Relevant loss schema code: `src/config/schema.py` and `src/trainers/losses/bbox_geo.py`.

## Task 4: Explain and keep DeepSpeed `zero2` default behavior in mind

Outcome: success

Preference signals:

- The user asked whether to enable DeepSpeed for Stage-1 and then “What does `zero2` improve?” -> they want practical training-system guidance tied to their current config, not generic theory.

Key steps:

- The assistant confirmed `deepspeed: enabled: true, config: zero2` is already present in the Stage-1 shared base config.
- It was explained that `zero2` mainly improves memory efficiency and training stability, with throughput gains being indirect rather than guaranteed.

Reusable knowledge:

- For the repo’s 4B Stage-1 runs, `zero2` is the default balancing point between memory savings and complexity.
- It is mainly useful to make the model fit and to support larger effective batches; it is not a universal wall-clock speedup.

References:

- [1] `configs/stage1/sft_base.yaml`.
- [2] `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`.
- [3] Training-config verification was done via `src/config/loader.py`.

## Task 5: Commit and push the changes cleanly

Outcome: success

Preference signals:

- The user explicitly asked for git hygiene: “Help me commit the changes properly” -> they prefer logically separated commits instead of a single mixed diff.
- The assistant split the work into clean reviewable chunks, which appears to match the desired workflow for future changes.

Key steps:

- The worktree was split into three commits:
  - `0b27b9e chore(stage1): use ce-ciou coord profile`
  - `cc65ac9 fix(eval): preserve coco proxy metrics routing`
  - `7a02ea7 feat(eval): add bundled coco proxy evaluation`
- The branch `main` was pushed to `origin/main` successfully.
- One untracked local directory remained and was intentionally left out of the commits: `.codex_config/pein/skills/coordexp-infer-eval-workflow/`.

Failures and how to do differently:

- A small amount of related-but-not-core workspace metadata remained uncommitted; future agents should explicitly ask whether local skill metadata should be committed or kept local if it appears in the worktree.

Reusable knowledge:

- Logical commit grouping that worked here:
  1. stage-1 config change,
  2. evaluator/test fix,
  3. proxy-eval workflow/tooling/docs.
- This repo is on `main` with `origin` set to `git@github.com:Pein2017/CoordExp.git`.

References:

- [1] Commit SHAs above.
- [2] `git push` completed to `origin/main`.
- [3] Final worktree state: only the local skill directory remained untracked after push.
