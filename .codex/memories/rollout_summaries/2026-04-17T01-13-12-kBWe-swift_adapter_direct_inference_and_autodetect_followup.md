thread_id: 019d98ff-c750-71b3-b2de-271d004af50e
updated_at: 2026-04-17T02:19:09+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/17/rollout-2026-04-17T01-13-12-019d98ff-c750-71b3-b2de-271d004af50e.jsonl
cwd: /data/CoordExp
git_branch: main

# Added direct HF Swift-adapter inference support, then began an auto-detection follow-up that was interrupted before completion.

Rollout context: The user wants to stop merging adapter checkpoints before inference. They provided a concrete adapter checkpoint under `/data/CoordExp/output/stage1_2b/.../checkpoint-716` and asked to keep the adapter separated while still making inference work directly. Later they asked whether `scripts/run_infer.py` could automatically detect whether the input is a full checkpoint or an adapter layer that should be temporarily merged for inference, and explicitly said they are migrating future inference from merged-full to adapter-only.

## Task 1: Direct inference from base model + separate Swift adapter

Outcome: success

Preference signals:
- The user said: "I want to use `<adapter checkpoint>` and the base model to inference directly" and "I want to keep the separated and still feasible for inference." -> future inference work should default to supporting adapter-separated inference instead of requiring an up-front merge.
- The user’s concrete checkpoint path and phrasing showed they want the exact existing artifact tree to remain usable; do not force a new merged artifact as the only path.

Key steps:
- Traced the inference path through `src/infer/pipeline.py` and `src/infer/engine.py`; confirmed the current HF loader used `Qwen3VLForConditionalGeneration.from_pretrained(self.cfg.model_checkpoint, ...)` and `AutoProcessor.from_pretrained(self.cfg.model_checkpoint, ...)`, so the code assumed one loadable checkpoint path.
- Checked the installed Swift API in the `ms` environment and confirmed `Swift.from_pretrained(model, model_id=..., inference_mode=True, **kwargs)` exists in `swift 3.10.0.dev0`.
- Inspected the adapter directory and confirmed it contained a standard PEFT/Swift adapter layout (`adapter_config.json`, `adapter_model.safetensors`, etc.), with `adapter_config.json` showing `base_model_name_or_path` pointing at `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`.
- Implemented `infer.adapter_checkpoint` as an optional config field, wired it through `InferenceConfig` and `_run_infer_stage`, and taught the HF loader to:
  - load the base model from `infer.model_checkpoint`,
  - keep the processor anchored to the base model path,
  - layer the adapter via `Swift.from_pretrained(..., model_id=adapter_checkpoint, inference_mode=True)`.
- Added a fast-fail guard so `infer.adapter_checkpoint` is rejected for `backend.type: vllm` for now.
- Updated `resolved_config` / `summary.json` provenance to record both `model_checkpoint` and `adapter_checkpoint`.
- Updated docs/config comments/spec text so the repo no longer implies "merged checkpoint only" for HF inference.
- Added a focused regression test that verifies the loader uses the base checkpoint for HF model + processor loading and applies the adapter via Swift on top.

Failures and how to do differently:
- No major failure in this task. The only adjustment was a dataclass field-order fix after adding `adapter_checkpoint` to `InferenceConfig`.
- The graph rebuild process was started per repo policy but was still running when the rollout ended; that was unrelated to the code change itself.

Reusable knowledge:
- For HF inference in this repo, the canonical loading seam is `InferenceEngine.load_model()` in `src/infer/engine.py`; this is where adapter-separated inference can be introduced without touching the rest of the pipeline.
- Swift direct adapter loading is available as `Swift.from_pretrained(model, model_id=<adapter_dir>, inference_mode=True)` in the active environment.
- The adapter directory itself can be recognized as PEFT-style when it contains `adapter_config.json` and `adapter_model.safetensors`; in this case, `adapter_config.json` pointed back to the Qwen3-VL 2B base model.
- The inference summary/artifact provenance is built in `src/infer/artifacts.py`, so that is the right place to add new config provenance fields once inference starts supporting a second checkpoint input.

References:
- [1] `src/infer/engine.py:138-170` — `InferenceConfig` gained `adapter_checkpoint: Optional[str] = None`.
- [2] `src/infer/engine.py:561-668` — HF loader now loads base model, then conditionally layers Swift adapter; `vllm` rejects adapter-separated mode.
- [3] `src/infer/pipeline.py:725-810` — pipeline reads `infer.adapter_checkpoint`, validates it only with HF, and passes it into `InferenceConfig`.
- [4] `src/infer/artifacts.py:47-126` — resolved meta and summary now include `adapter_checkpoint`.
- [5] `tests/test_infer_batch_decoding.py:295-416` — new regression test for base-model + Swift-adapter loading.
- [6] `configs/infer/ablation/coco80_desc_first.yaml` and `configs/infer/ablation/coco80_testdev_desc_first.yaml` — comments now mention merged checkpoint OR base model plus adapter checkpoint.
- [7] `docs/eval/COCO_TEST_SUBMISSION.md` and `openspec/specs/inference-engine/spec.md` — docs/spec updated to describe optional adapter checkpoint support.
- [8] Verification: `conda run -n ms python -m pytest tests/test_infer_batch_decoding.py -q` passed (`11 passed`).
- [9] Evidence of the upstream Swift API: `swift_version 3.10.0.dev0` and `Swift.from_pretrained sig: (model, model_id=None, adapter_name=None, revision=None, **kwargs)`.

## Task 2: Auto-detect full checkpoint vs adapter checkpoint for `scripts/run_infer.py`

Outcome: partial

Preference signals:
- The user asked: "Do you think we can upgrade the `scripts/run_infer.py` to automatically detect whether input is a full checkpoint or a `adapter` layer that need to be temporarily merged for inference?" -> future inference work should anticipate that the user wants a single config/input that can accept either artifact type.
- The user said: "Currently, once I got an adapter artifact, I use `scripts/merge_coord.sh` to merge the checkpoint first and then send to `infer.py`. Help me support the both ways." -> future work should preserve both workflows and reduce the need for manual merging.
- The user added: "I am going to migrate from `merged full` to `adapter` only for future inference" -> future defaults should bias toward adapter-only inference as the preferred path, while still supporting merged checkpoints for backward compatibility.

Key steps:
- Re-opened the already modified inference seam and confirmed the current implementation only supported adapter handling when an explicit `infer.adapter_checkpoint` was provided.
- Began reasoning about auto-detection as a loader responsibility: a single `infer.model_checkpoint` should be able to represent either a fully merged checkpoint or a local adapter directory, with the loader deciding by inspecting local metadata such as `adapter_config.json`.
- The turn was aborted before the auto-detection patch was completed, so no new code for automatic detection was landed in this rollout.

Failures and how to do differently:
- The auto-detect follow-up was interrupted before implementation, so there is no verified behavior to preserve yet.
- The correct next step is to implement detection in the inference loader rather than adding a second user-facing mode: keep `scripts/run_infer.py` accepting one canonical checkpoint input, but internally resolve whether it is a merged/full checkpoint or an adapter artifact.
- Because the user explicitly wants to migrate to adapter-only inference, future work should treat adapter-separated inference as the primary design and merged checkpoints as compatibility support.

Reusable knowledge:
- The current codebase already has a clean seam for this follow-up: `src/infer/engine.py:load_model()` and `src/infer/pipeline.py:_run_infer_stage()`.
- The adapter artifact directory shape seen in this rollout is a reliable detection hint: `adapter_config.json`, `adapter_model.safetensors`, `trainer_state.json`, `training_args.bin`, plus the adapter config’s `base_model_name_or_path`.
- The helper `scripts/merge_coord.sh` remains the existing merge path for cases where merged deployment is still desired, but the user wants inference to stop depending on it.

References:
- [1] User request: "automatically detect whether input is a full checkpoint or a `adapter` layer that need to be temporarily merged for inference"
- [2] User request: "Currently, once I got an adapter artifact, I use `scripts/merge_coord.sh` to merge the checkpoint first and the send to `infer.py`. Help me support the both ways."
- [3] User request: "I am going to migrate from `merged full` to `adapter` only for future inference"
- [4] Existing adapter directory shape observed at `output/stage1_2b/.../checkpoint-716/`: `README.md`, `adapter_config.json`, `adapter_model.safetensors`, `additional_config.json`, `coordexp_checkpoint_state.pt`, `trainer_state.json`, `training_args.bin`.
- [5] Existing merge helper: `scripts/merge_coord.sh` (still the reference workflow for merged checkpoints)." ,
