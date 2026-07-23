# Pi Stage 0 Network Failure Checkpoint

Captured: `2026-07-22T17:15:54Z`.

The user authorized a sixteen-execution capability screen: twelve Pi cells
covering Luna, Terra, and Sol across `medium`, `high`, and `xhigh`, plus four
native Codex Sol controls. `max` was explicitly excluded. Pi had maximum
built-in tool permissions inside disposable task clones, unrestricted outbound
network, no graphics-processing-unit access, and all persistent home, auth,
session, sandbox, and result state below the worktree-local `.pi-worker/`.

All four native controls returned task answers. Frozen verification passed
tasks 1, 3, and 4. Task 2 failed because the answer misspelled the exact
`HFBackendSession` symbol; its richer trace also exposed an independent
one-row-per-path overwrite defect in the frozen verifier. All twelve Pi shell processes
launched, but no task reached a model: each final assistant event reports
`fetch failed` with zero total tokens. The common runner used
`/usr/bin/env -i` and failed to forward the host proxy variables. A zero-model
probe returned `401` from the OpenAI endpoint both on the host and in the
chroot when proxy variables were present, while the identical chroot without
them timed out with code `000`. This establishes a harness network failure,
not a Pi, model, task, or reasoning-level failure.

The Pi-home runner `.pi-worker/home/pi-worker/run-stage0-pi-matrix.sh` now preserves only the
proxy variables across the clean environment boundary. It has not been rerun.
The user's cap is ambiguous as to whether zero-token transport launches count
as executions; because twelve Pi launches plus four native controls have
already occurred, do not start another model call without resolving that cap.

Durable evidence:

- `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/results.md`;
- `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/pi-shared-network-failure-receipt.md`.

Raw Pi events and per-run digests remain under `.pi-worker/results/`. The unit
is `blocked` with `partial` evidence. No Pi adoption, rejection, ranking, cost,
or harness-effect claim is supported.
