# Native qualification accepted; finite primary contrast released

The lead accepts the completed one-update B qualification as technical evidence
for the frozen primary package. Execution used `22ea4e1175b8e11fd16b9d8d062eb49769a6207b`
and release SHA256 `fabc492e6fd3e56b04619b451932d649cd7942cdcf5b4774ba1a234d5564503e`.
The subsequent `62be9514622022891249837dc269a9967b6c1e9e` changes only the
worker's [results](results.md). Accepted implementation remains `cbbce5e588daeb5a9f3bad7422be3abe0e5e9fdc`.
No model, checkpoint load, or successful readback was repeated for lead acceptance.

Saved-consumer verification is
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/lead-native-consumer-check-01.json`.
It verifies terminal/readback/metric bindings, the exact 18-image/570-owner
denominator, per-image aggregation and owner transitions, 73 requests, all 16
rank completion receipts, checkpoint versions 0/1, finite original action
likelihoods, the separate six-action media diagnostic, and 18/18 fresh-reload
token/stop matches. Owner and both torchrun exits are zero; all owned processes
settled. Prior CPU exception and unrelated fixture failures remain in results.

## Scientific observation and numerical limit

The qualification regressed: matched owners 255 to 246, duplicate events 5 to
119, geometry-invalid rows 4 to 226, and caps 0 to 1. It gained 17 owners, lost
26, and retained 229. Image 351017 contributes 113 duplicate events and 221
invalid rows after reaching 3084 actions. These are retained one-update B
observations, not evidence that the full A/B contrast succeeds or fails, and not
physical-negative or generalization evidence. No quality gate is introduced.

Focused source and saved-action review found no demonstrated policy-definition,
causal-position, media, or gradient-connectivity counterexample. Both callers use
the same intended median-normalized policy; replay retains current-weight factor
gradients. Nevertheless, cached generation and full-history replay are numerically
different. The largest sampled policy log-probability difference is -0.857063
(raw -0.858914), image 477415/action 185/token 152200; that action has zero
advantage. The active duplicate image 14038 has maximum policy difference
0.173078. Its saved behavior likelihoods give scalar loss 0.488464 with the same
fixed advantages, versus replay loss 0.487500. This comparison does not bound
gradient error. The separate diagnostic differs by up to 0.385347 even with both
paths in evaluation mode. The evidence is compatible with cached/full numerical
approximation, but does not identify its exact cause or prove negligible bias.
Claim numerically approximate replay, never exact cached-policy gradients.

## Exact primary release and continuation

Release the already authorized two-arm study under [unit.md](unit.md), with
executable packets in the worktree output root's `primary-release-01/` directory:
`primary-A-release.json`, followed by `primary-B-release.json`. A shared receipt
binds both fresh packets to the same clean execution source after this records-only
commit, unchanged accepted code, protocol, runtime, inputs, anchor, qualification
evidence, and worker `01a101ac-9ec8-7862-bdb6-38b9cd154673` (GPT-6.1-Sol/xhigh).
Old qualification/proposal receipts remain immutable. The worker must verify
the published packets and exact SHA256 values before invoking either arm.

Run A then B serially, one invocation each, on all 8 GPUs. Both independently
restart the original anchor with fresh AdamW state; neither resumes qualification
weights or optimizer. Each completes 16 continuous updates, greedy versions
0..16, checkpoints 0/1/4/8/16, and its automatic final readback. The B start is
conditional on A's successful terminal verification and process settlement,
without another lead acknowledgment. Keep tracked source and records unchanged
through both terminal source checks; an interim A report belongs in outputs and
direct transport. Then run the maintained saved-artifact A/B comparison once,
without extra acquisition, forward, checkpoint reload, or repeated readback.

Combined logical work remains 1188 requests (612 greedy, 576 sampled), at most
3,663,792 generated actions, 576 positive, up to 576 geometry and 288 duplicate
replays, and 10 checkpoint exports. The six-action diagnostic and fresh native
reload are qualification-only. No retry, warmup, extra arm, dose extension,
retuning, checkpoint selection, or quality early stopping is authorized.

Use 43,200 seconds per arm as an operational observation estimate, not a cap,
kill trigger, retry authority, or guaranteed upper bound. Qualification owner
wall was 1078 seconds. The slowest-rank matched-shape projection is 7089 seconds
per arm; if all 99 requests on the largest rank reach the horizon at the observed
capped rate, acquisition alone is 31,222 seconds. Maximum measured allocated CUDA
memory was 9.16 GB (reserved 12.04 GB), main RSS 13,617,268 KiB. Greedy/reload
reached 3084 actions, while samples reached only 343: full-horizon sampled
backward, evolving event supply, parser costs, contention and finalizer RSS
remain uncertain. These are disclosed operational risks, not extra qualification
queries or claims that all future shapes were measured.

The same worker owns each invocation through terminal artifacts, cleanup and
direct return. Timeouts resume observation of the same live invocation.
Nonfinite values, corrupted identities/positions, unsupported policy/media,
OOM or distributed failure require a prompt failure report and settlement;
preserve partial evidence and hold dependent B. In-scope diagnosis and repair
may proceed after settlement, but changed source or replacement native execution
requires a new exact lead packet and new output. The lead retains final consumer
and scientific acceptance. Stop after the declared comparison and report.
