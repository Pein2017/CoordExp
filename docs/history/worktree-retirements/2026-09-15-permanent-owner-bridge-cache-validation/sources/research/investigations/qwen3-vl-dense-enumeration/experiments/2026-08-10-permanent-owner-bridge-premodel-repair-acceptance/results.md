# Permanent owner bridge pre-model repair acceptance

Date: 2026-08-10 UTC

## Decision

The pre-model production orchestration repair is mechanically accepted at commit
`7d6dd733f736f9f3003761c9404d866872e08a66`. The exact W1 and W8 smokes,
checkpoint inspection, and fresh HF lifecycle smokes pass at that commit.

This record does **not** authorize a production recovery attempt. The original
singleton claim remains consumed. A production recovery remains HOLD until the
user explicitly authorizes one append-only, parent-linked successor attempt.

## Failure being repaired

The first production activation at commit `7f399a2f04bbea615e66c42531381c10bfb42fdb`
failed before `run.json`, model load, or optimizer creation. Ranks 1-7 waited in
the post-process-group pre-model status broadcast while rank 0 performed a
production-sized full cache scan. The original claim, activation, eight worker
admissions, binding failure, and logs remain unchanged in the canonical ledger.

The repair moves exact production cache attestation before Accelerator/process
group construction. It consumes the guard intent, re-derives the four live
cache identities, inspects manifests and determinants only, and requires an
error-symmetric, live-worker-bound W8 filesystem quorum. It then reconciles the
actual Accelerator topology back to the admitted rank.

## Static and test evidence

- Focused final repair suites: `203 passed`.
- Canonical repository pytest scope: `2029 passed`.
- Ruff check and format check: pass.
- OpenSpec strict validation: pass.
- Git diff check: pass.
- Deterministic CPU algorithm battery: `23/23 passed`, clean commit, source-set
  SHA-256 `a35e1c016a4d44de8b581b955da543d23738d7432cc0e50806293d6d737fee9c`.
- Durable battery receipt: `algorithm-receipt.json`, SHA-256
  `e5b0fcc5d07c0bffcf893ea3cb6a7448268458b20397934e53e57bdb44edf7a2`.

Independent fixed-commit reviews:

- Sol xhigh contract audit: PASS; no P0/P1/P2 code findings.
- Opus xhigh fixed-commit review: code-path PASS. It retains one execution-risk
  boundary: the canonical production pipeline cannot be rehearsed end-to-end
  against a disposable ledger without adding a bypass-like production seam.
  That seam was deliberately not added. The real recovery remains the first
  exact canonical pipeline execution.

## Real W8 pre-model rehearsal

The disposable rehearsal used a synthetic claim and strict rewritten intent
under `/tmp`; it never wrote the canonical production ledger and used no GPU,
model, or training execution. The real lifecycle was:

`setsid torchrun W8 -> activation receipt -> admit_stage1_worker -> default
production pre-model resolver -> eight durable slots -> CPU Gloo init/barrier/destroy`.

Observed evidence:

- activation publisher, torchrun launcher, and all ranks exited 0;
- worker admission quorum: 8/8, slot skew `1.172593s`;
- pre-model quorum: 8/8 `ok`, slot skew `3.913480s`;
- resolver duration: `90.01-90.19s` per rank;
- exactly 32 manifest inspections: four per rank, all `level=manifest`;
- payload-wide loader calls before process-group initialization: 0;
- real CPU Gloo init, barrier, and destroy completed on every rank;
- shared plan digest `806adfb1642e4c47db4733f66b93c471b6aa53644578504c7372926eb66e0de4`;
- shared pre-model quorum fingerprint
  `c81bafbb8cb442ebbebc8482253eaed7dd874355cedac7f440cdc5167cd947c1`.

Durable receipt: `premodel-w8-rehearsal-receipt.json`, SHA-256
`a9cccc6679398116f8e5a881db8f0f975836f5c0a412091de588d09c962ba059`.

The 3.91-second observed finish spread uses 3.3% of the 120-second quorum
budget. No timeout change is justified by this receipt.

## W8/A3 smoke

Run directory:

`outputs/smoke/coordexp_swift_owner_bridge/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1-20260810T213438Z`

- config fingerprint: `ca563a0a5fc36328b7767c45707631a340357f1bb58459d509865843d555e629`;
- status completed; world size 8; completed steps 8/8; consumed packs 24;
- 16 train/eval logging rows; every train row finite/applied;
- 12 owner-bridge events; four checkpoint events;
- first-step choreography receipt: eight rank summaries, digest
  `f0fed8b904401fe4731d39a9499feddf0ff8e2d1db50c328e99a7994d6702c9e`;
- final step-8 bridge payload fingerprint
  `38b1465ced741c8c3e5ffe75ad90273a0c0c57e53c02b463be1d4eaad1748103`;
- strict `inspect_checkpoint_composition` reinspection passed against that
  external event fingerprint.

## W1/A24 shadow smoke

Run directory:

`outputs/smoke/coordexp_swift_owner_bridge_single/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_single_gpu_ebs24_train1_val1-20260810T214353Z`

- config fingerprint: `a641b422790ce4bec2716e8b78914d219bdc80616b09203fa6616947731e734d`;
- status completed; world size 1; completed steps 4/4; consumed slots 96;
- eight train/eval logging rows; all finite/applied;
- 12 owner-bridge events; four checkpoint events;
- final step-4 bridge payload fingerprint
  `d258eaa0c2da770bb81c61e3c51fc618bf51882fc3302940cab1c62ff716fa63`;
- strict `inspect_checkpoint_composition` reinspection passed.

## Fresh HF lifecycle smokes

Both inference configs were materialized from the current smoke checkpoints,
then loaded in fresh processes and run on the two-image smoke fixture.

W1:

- config: `configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-w1-fresh-7d6dd73.yaml`;
- resolved config fingerprint
  `9675a018529a7c8ae5aea5c83c70e978f6a1d2c22747a68b8a3e8b2e2f6e2d16`;
- composition fingerprint
  `5cff08ace5a5617c7be54b152d8d942945fc084c7a47e6a4a10765a5d0850c26`;
- output root
  `outputs/smoke/coordexp_swift_owner_bridge_w1_hf/owner-bridge-stage1-w1-fresh-7d6dd73`;
- lifecycle rows 2/2 valid and stopped, zero terminal errors or dropped errors;
- lifecycle SHA-256
  `da441932ecb2226acb6f1fac48f1abb7ce7a52c33075174aea8b8bbab56a8e22`.

W8:

- config: `configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-w8-fresh-7d6dd73.yaml`;
- resolved config fingerprint
  `450f7b179fb9422f193355070976a39eaf821e1459489d11bbc3f220936f9e7b`;
- composition fingerprint
  `2e0ad145c2e08f893c0c2f0c12c4f48572103ba637c04c6105e9f0c58f62df5d`;
- output root
  `outputs/smoke/coordexp_swift_owner_bridge_w8_hf/owner-bridge-stage1-w8-fresh-7d6dd73`;
- lifecycle rows 2/2 valid and stopped, zero terminal errors or dropped errors;
- lifecycle SHA-256
  `56e7ac8f41ee6a5df49c5bc510fffd56b676c30f19a1f24bc12f1669d2e20f70`.

Both runs bind native-greedy fingerprint
`7beaf80cceb72281857b844ea8a627e9a046c50d57823123c2b9036581db1265`.

## Claim boundary and stop rule

This evidence establishes mechanical cache identity, pre-PG lifecycle,
distributed smoke execution, finite/applied optimizer updates, checkpoint
composition, and two-row native-greedy HF lifecycle behavior. It does not
establish model quality, benchmark eligibility, generalization, causal benefit,
or independent per-request image isolation beyond the batch-scoped lifecycle
contract.

The original production claim must not be deleted, moved, overwritten, or
reused. A recovery may proceed only after explicit user authorization for one
append-only parent-linked successor attempt. If that attempt cannot bind a run
or produce the first finite/applied artifact heartbeat, stop; do not create a
third activation.
