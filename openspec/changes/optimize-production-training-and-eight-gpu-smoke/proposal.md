## Why

Production training currently submits every preprocessing task at once and has no completed current-code end-to-end runtime witness. The upstream comparison identified useful execution patterns, but functional correctness, resource bounds, and performance gains must be established separately before changing defaults.

## What Changes

- Bound preprocessing submissions in the existing cache-workflow owner, preserving encoded content, canonical order, pack/step membership, supervision, and cache identity rules.
- Add a production-shaped **eight-GPU, four-update smoke** crossing preparation, training, evaluation, checkpoint publication, fresh same-world-size resume, HF/vLLM inference and direct evaluation. Keep the existing one-to-four-GPU interface; require no full training or all-topology runtime matrix.
- Preserve data/template/geometry/token, segment-balanced objective, optimizer and schedule semantics. Record resource observations and compare like workloads; distinguish smoke success from a measured speedup.
- Preserve supported pre-change checkpoint **loading and inference**, including independently configured DoRA and embedding-delta components without a root publication manifest. **BREAKING scope boundary:** pre-change optimizer/RNG/scheduler/cursor states receive no training-resume compatibility guarantee. New implementation checkpoints retain strict exact resume.
- Apply the user's accepted one-grid coordinate tolerance to dynamic-HF versus merged-BF16 composition qualification. Keep prompt, non-coordinate tokens, sequence length, EOS, component bytes and model identities exact; retain numeric drift as diagnostics. vLLM executes the authenticated merged model with its own likelihood semantics, without claiming dynamic-HF probability parity.
- Keep cache/runtime, loss, adapters, checkpoints and inference with their current module owners. Correct real failures exposed by the narrow vertical slice; add only missing load-bearing regression cases.
- Cover lazy cache hydration, kernel/optimizer options, data-mixture preparation, adapter precision, vLLM concurrency/weight handoff, FSDP2, RL, and asynchronous checkpointing in an evidence-triggered roadmap. Untriggered capabilities do not require implementations or new abstraction layers.
- Complete one independent Astra review round before implementation; resolve material findings and proceed directly under the user's explicit authorization.

## Capabilities

### New Capabilities

None. The changes extend the existing infrastructure capability rather than creating a parallel training framework.

### Modified Capabilities

- `infra-base`: bounded preparation and semantic preservation; eight-process smoke support and evidence-scoped acceptance; supported historical inference payloads separated from current exact-resume state; performance claims bound to comparable observations.

## Impact

- Implementation owners: `src/training/cache_workflow.py`, existing training/runtime/loss/checkpoint/inference owners only where a reproduced failure requires correction.
- Configuration/docs: a new smoke overlay under `configs/smoke/`, current train/operations contracts, and this change's validation records. Existing command-line interfaces remain stable.
- Tests: preparation order/concurrency/failure behavior, existing cache and loss oracles, payload loading and current exact resume; a few-step real eight-rank production slice.
- Baseline: infras `70e576f9606c48d322b6c67df9baac78c22fc3f6`; comparator ms-swift `0673cf75dca7d0b9b608b4a76632fb508ead5076` (`4.6.0.dev0`). No ms-swift runtime dependency or blanket library upgrade.
- Resources: local eight A100 80GB GPUs; no user-imposed GPU-hour or per-run wall-time budget. Work remains finite (few-step smoke, bounded requests/workers); failure detection timeouts are not spending caps. Expected shared GPU occupancy is not a reason to wait before an authorized launch.
- Non-goals: research-probes changes, new scientific objectives, full epochs, multi-node training, exhaustive model-size qualification, old-training-state migration, or unconditional RL/FSDP/Ray/serving adoption.

The user resolved the grilling decisions and explicitly authorized implementation after proposal and independent review. Detailed decisions, acceptance and conditional triggers are owned by `design.md`; executable work is owned by `tasks.md`.
