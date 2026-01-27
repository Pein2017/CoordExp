## 1. Spec & Config Surface
- [x] 1.1 Update `openspec/specs/rollout-matching-sft/spec.md` to allow vLLM `server` mode (keep `colocate` default) and make server-mode determinism/logging requirements explicit.
- [x] 1.2 Specify a config contract under `custom.extra.rollout_matching.vllm`:
  - `custom.extra.rollout_matching.vllm.mode: server`
  - `custom.extra.rollout_matching.vllm.server.servers: [...]` (preferred) and strict legacy paired-list support
  - `custom.extra.rollout_matching.vllm.server.timeout_s` (startup/communicator) and `custom.extra.rollout_matching.vllm.server.infer_timeout_s` (per-request)
  - deterministic multi-server request chunking + order preservation (including N==0 no-op)
  - `custom.extra.rollout_matching.vllm.sync.mode` (`full|adapter|auto`) + `custom.extra.rollout_matching.vllm.sync.fallback_to_full`
- [x] 1.3 Add a new config template (YAML) for the 3v1 workflow (3 rollout GPUs + 1 learner GPU) with dlora + post-rollout packing and explicit sync mode.

## 2. Trainer Integration (No Behavior Change by Default)
- [x] 2.1 Extend `RolloutMatchingSFTTrainer` to support `custom.extra.rollout_matching.vllm.mode: server`:
  - fail fast if learner `world_size > 1` (v1 is single-process learner)
  - connect to pre-launched ms-swift rollout server(s)
  - send JSON-serializable rollout requests compatible with ms-swift `RolloutInferRequest` (images as strings)
  - request rollouts returning `prompt_token_ids` and `response_token_ids`
  - implement deterministic multi-server chunking + reassembly that preserves request order
  - handle N==0 (no requests) as a no-op
  - apply `infer_timeout_s` to `/infer/` calls when set
- [x] 2.2 Implement weight sync to server without disk reload:
  - default: full merged weights sync (robust for multimodal + DoRA)
  - optional: adapter-only sync when enabled, with `fallback_to_full` behavior
- [x] 2.3 Define sync cadence:
  - sync-on-demand only when generating fresh rollouts
  - in buffered mode, sync only on E-steps (fresh rollouts)
- [x] 2.4 Log server-mode reproducibility metadata (server list, effective sync mode, rollout seed).

## 3. Dataloader / Packing Semantics
- [x] 3.1 Keep dataset iteration deterministic and avoid skipping samples:
  - learner owns the dataloader
  - rollout_buffer window repeater behavior remains intact
- [x] 3.2 Keep post-rollout packing semantics unchanged (micro/window scheduling) and document how 3v1 interacts with packing.

## 4. Validation & Docs
- [x] 4.1 Add a small integration test or debug harness that validates server-mode plumbing:
  - strict prompt-prefix alignment
  - stable construction of `Y_train` (prefix + FN append)
  - deterministic multi-server chunking preserves output order
- [x] 4.2 Update `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` with:
  - how to launch `swift rollout` on 3 GPUs (server mode)
  - how to launch the learner on 1 GPU
  - sync-mode choices, fallback behavior, and common failure modes
