# Paired engineering assignment: exact-history batched crossed continuation

Read `unit.md` in this directory first. It owns the overall question, scope,
decoding, scientific distinctions, GPU budget and benchmark rules. This brief is
byte-identical for both candidates; only your native agent identity differs.

## Ownership and isolation

Worktree: `/data/CoordExp/.worktrees/self-rollout-behavior`, baseline HEAD
`f8c5f1f5100159db62dd7ff4090555dff63b99d8`.
Your candidate key is the final component of your native agent task name.
You may write only `candidates/<candidate_key>/` under this unit, and the matching
external output directory under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/engineering/<candidate_key>/`.
Do not read any other candidate directory or competitor messages/results.
No edits to shared core, configs, unit/data, other worktrees or checkpoints;
no commits, subagents, tool/model configuration changes, or external services.
Read-only reuse/import of existing project code is encouraged.

## Deliverable

Implement the smallest reusable experiment-local runner for one recipient
checkpoint per process. It must load the actual original Source or Rweak64,
consume exact native prompt + common generated prefix + designated row/EOS,
then greedily continue with the remaining whole-trajectory budget. Efficient
real multi-image batching and artifact-consumer correctness are essential.
Root will schedule independent processes across eight GPUs, merge raw artifacts,
own the final scientific reducer and choose one implementation for formal use.
You do not schedule the full experiment or own the scientific conclusion.

Provide `run.py` with this CLI (additional explicit diagnostic flags are okay):
- `--manifest PATH`
- `--recipient source|rweak`
- `--mode qualify|cross`
- `--batch-size N`
- `--case-ids ID[,ID...]` (optional subset of frozen manifest IDs)
- `--output-dir PATH`
- `--max-wall-seconds N`
- `--execute` (without it, validate/plan only; never load a model or use CUDA).

Manifest location, once root freezes it:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json`.
It has schema `row_cross_manifest_v1`, sources, policy, selection,
qualification_case_ids, cases. Cases contain row_id/stratum, raw image/input
record and provenance, common_prefix_token_ids, actions.source/rweak (kind
row|eos, token_ids, text), diagonals.source/rweak (exact token IDs, native
parsed record, owner matches, stop), and remaining budgets. Data preparation
is independently owned and still running. Root will send the same frozen schema
and hash to both candidates before any real run. You may begin source discovery
and implementation now; do not invent a replacement manifest or assume unknown
key spellings are authoritative. Clarifications are shared with both candidates.

## Source bindings and reusable surfaces

Original paired run roots (each contains `configs/resolved.yaml`,
`run_manifest.json`, raw outputs, `pred_token_trace.jsonl`, `image_plan.jsonl`):
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/evaluation/Source/holdout/Source-holdout512-native-v1/`
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/evaluation/Rweak/holdout/Rweak-holdout512-native-v1/`

Original receipts specify **FP32 / SDPA, patch_embed_linearization enabled,
left padding, greedy RP1.0, cap3084**. Preserve the exact base, adapter,
selected embeddings, media geometry, prompt and Qwen stop semantics. Do not
substitute the newer RP1.10 A16 config, BF16, merged weights or another model.
Original Rweak code/config owner is read-only sibling
`/data/CoordExp/.worktrees/coco-gt-correction-portfolio` (HEAD
`a4bac643cb75734d9553eeb1d8393df73e7db380`). Discover its actual composition
loader from the original resolved YAML; do not treat a nonstandard checkpoint
as a plain adapter without verifying its owning load contract.

Relevant existing code for discovery, not a mandate to compose all of it:
`src/inference/hf_backend.py` (HF session, native batching, exact-history
materialization, Qwen position handling), `src/inference/parsing.py`, native
matching/evaluation utilities, and prior complete-row research runners under
`scripts/research/` and the September8 self-rollout unit. Inspect exact symbols;
do not copy a large old framework or alter shared code to make the probe fit.

## Semantics and real acceptance

1. Native multimodal input must match original image/prompt/tokenizer/embedding
   identities. Append literal generated IDs without decode/re-tokenization.
   Construct recipient KV normally, never transplant a donor cache.
2. EOS actions terminate explicitly; do not continue after them. For row actions,
   enforce each request's own remaining budget: `3084 - len(prefix + action)`.
   Padding and batching must not grant extra tokens or impose a shorter cap.
3. Qualify same-arm replay against the original diagonal, and single versus
   batched outputs on the common root-selected variable-prefix fixtures. Cover
   recipient loading, prompt/media/position identity, row/EOS handling, raw-token
   persistence, fresh CPU reload and native parser/matcher consumption. Do not
   hide a failed replay by dropping the case or relaxing exact scientific identity.
4. Enable useful batching, bucket lengths/shapes where beneficial, reuse the
   loaded model and avoid unused full-vocabulary score/hidden-state payloads.
   These are execution-only optimizations. Measure throughput, memory, forwards,
   initialization/decode time and padding overhead; do not assert efficiency from
   batch size alone. Root will choose the measured production B and GPU layout.
5. Required `rows.jsonl` record fields: case_id (the manifest row_id), recipient,
   action_source, cell (`00`,`01`,`10`,`11`), mode, manifest_sha256,
   prefix_token_ids, action_token_ids, suffix_token_ids, generated_token_ids,
   raw_decode_text, decode_stop_reason, generated_token_count, remaining_budget,
   and `parsed` in the native raw detection-record shape. Include provenance or
   additional counters where useful; all generated IDs include the natural stop
   token if emitted, but never padding. `generated_token_ids` must equal
   prefix+action+suffix exactly. Distinguish forced terminal from sampled EOS.
6. Persist `receipt.json` binding source/config/checkpoint/embedding/code/input
   identities, CLI, actual runtime/device/B, complete requested/returned IDs,
   model/image forwards, stop/cap counts, timing, memory and success/error status.
   A partially written run is not success. Failure artifacts/logs stay retained;
   do not overwrite successful or failed raw evidence on retries.

Add focused CPU tests with at least one meaningful token/identity/budget failure
and a variable-prefix batching edge. The real consumer path, not synthetic tests
alone, owns acceptance. Existing original raw diagonals provide the golden
behavior; a partial technical failure is not a scientific negative.

## First return and GPU coordination

First return: CPU-tested candidate, exact qualification commands, material
source/schema questions, and estimated forwards/batches/budget. **No CUDA/model
load until root sends a concrete GPU/time grant.** You may run CPU tests and
inspect existing artifacts now. Root retains one scheduling owner and aggregate
4 GPUh accounting. At most four shared fixture IDs are used for qualification;
repeat only the minimal same-arm/cross and B1/B>1 executions needed for acceptance.
Then, on grant, own the real qualification and any root-approved bounded repair.
Do not run the formal32-case panel yourself or continue after the package stop.

Report candidate / HOLD / NEEDS_CONTEXT truthfully; never self-label lead-accepted.
Record implementation, tests, qualification, corrections and wait intervals for
the all-in acceptance-cost comparison. Root—not first completion—chooses winner.
