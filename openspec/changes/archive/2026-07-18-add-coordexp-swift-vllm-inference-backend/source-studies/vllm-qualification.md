# vLLM Qualification Source Study

## Scope

Qualify the installed inference boundary before implementation claims support.
The active environment currently provides vLLM 0.14.1, Transformers 4.57.1,
PEFT 0.17.1, Torch 2.9.1, and qwen-vl-utils 0.0.14.

## Static Findings

- vLLM registers Qwen3-VL as a multimodal MRoPE generative model.
- `SamplingParams.logprobs` returns selected-token likelihood evidence. Prompt
  logprobs describe teacher-forced prefill evidence and are not a valid
  same-executed-state raw channel for generated tokens after token zero.
- Engine `logprobs_mode` defaults to raw likelihood and must be explicitly set
  to `processed_logprobs` to preserve current CoordExp score semantics.
- vLLM 0.14.1 and current preview source reject PEFT `use_dora: true`.
- vLLM has no loader for CoordExp selected-token embedding delta payloads.
- Offline `LLM` owns model and multimodal processing; the current HF tensor
  `DecodeRequest.model_inputs` cannot be reused as a backend-neutral request.
- The current outer worker contract already isolates one parent-visible GPU as
  logical `cuda:0`; initial vLLM execution should use TP=1 and DP=1.
- Stock vLLM exposes only one engine-level logprob mode and one generated
  logprob output channel. The accepted no-vendor-patch mechanism uses a
  processed authoritative pass followed by a fresh raw-logprob engine. A
  CoordExp `AdapterLogitsProcessor` forces the accepted sequence after raw
  log-softmax is captured, preserving the raw distribution while reproducing
  the exact incremental decode prefix.
- The canonical single-image fixture has 14 unexpanded input ids with one
  image placeholder and 1,027 executed ids with 1,014 image placeholders.
  Initial generation and forced replay must submit the unexpanded form plus
  media and compare vLLM-returned ids against the expanded form; replaying
  expanded ids with media would expand the first placeholder a second time.

## Raw-Likelihood Correction

The initial qualification receipt proved prompt-logprob alignment and source
identity, but a later materialized-HF/vLLM numeric probe falsified the stronger
assumption that multimodal prompt prefill is numerically equivalent to the
incremental continuation state. Existing policy likelihood and evaluator
scores remain valid. Prompt-prefill raw values are superseded evidence only;
Wave 4 requires a fresh executed forced-decode receipt before raw vLLM support
is accepted.

## Reproducible Installed-Source Handles

The initial static study used the `ms` environment and these exact package
versions: vLLM 0.14.1, Transformers 4.57.1, PEFT 0.17.1, Torch 2.9.1, and
qwen-vl-utils 0.0.14.

- Qwen3-VL multimodal registration and prompt replacement:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl.py`,
  SHA-256 `c2c87e05040b719d614c3564703e9fca3820dc61fbe29aadce5d1039dd602e0e`.
- Logprob mode definition:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/vllm/config/model.py`,
  SHA-256 `bb8a629254b756f030ccb1808f72df0a9e537e22ae364521044993448d960c9c`.
- Prompt/sample logprob position containers:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/vllm/logprobs.py`,
  SHA-256 `b875096f2490274d8e169fa482bd2007f6579dfb809ba4831f01d9fab45096a2`.
- DoRA rejection:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/vllm/lora/peft_helper.py`,
  SHA-256 `acb6c22e3f94b235c50f6c170f5f21380b733ffdfcf45c1ca68cde22dce046f9`.

These static handles are reproducible evidence, not runtime qualification. The
executed receipt MUST bind them again and records drift as a failed match.

## Required Executed Probes

1. One base-only Qwen3-VL image with `do_resize=False`.
2. Exact local/backend prompt ids including image placeholders.
3. Exact special-token-preserving generated ids/text and `<|im_end|>` stop.
4. Processed chosen-token logprobs under repetition penalty 1.10.
5. Forced incremental raw-logprob replay aligned to every generated token,
   including stop, at every supported `max_num_seqs` value. Prompt-prefill
   likelihood remains historical diagnostic evidence only.
6. Worker CUDA binding, child-process mode, close behavior, and no orphan child.
7. Exhaustive source-base snapshot identity plus every exercised runtime source
   owner for multimodal projection, likelihood, engine shutdown, and
   distributed cleanup.
8. Independently reopened generation/replay image bytes and RGB pixels, with
   exact placeholder ranges derived from returned prompt ids.

## Version Policy

0.14.1 is the initial candidate, not accepted runtime evidence until the probes
pass. Upgrade or downgrade is permitted only after recording the concrete
failure, candidate dependency changes, and rerunning every probe.

Canonical passed runtime receipts live under
`src/inference/qualification_receipts/`, outside the active change so archiving
cannot disable the backend. A receipt MUST record
`status: passed`, the probe script SHA-256, fixture/model/tokenizer hashes,
installed package and source hashes, CUDA binding, engine process mode and
arguments, exact prompt/generated ids, stop evidence, processed likelihoods,
raw-replay alignment, and post-close child-process evidence.

## Executed Attempt History

- Attempt 1 used vLLM's utilization-based automatic KV-cache sizing on a
  shared A100. Model loading completed, but vLLM aborted during memory
  profiling because another process released about 1.4 GiB between its initial
  and final free-memory snapshots. The preserved receipt is
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-1-failed.json`.
- The retry uses an explicit 1 GiB KV-cache allocation for the one-sequence,
  2,048-token probe. vLLM still runs its model profile/compile pass, but skips
  the racy free-memory subtraction. This setting is execution provenance, not
  a relaxation of prompt, likelihood, stop, CUDA, or cleanup gates.
- Attempt 2 passed model loading, profiling, CUDA graph capture, and generation,
  then correctly failed the stop gate because the untuned base model exhausted
  a 256-token prose window without emitting `<|im_end|>`. The preserved receipt
  is `source-studies/receipts/vllm-0.14.1-qualification-attempt-2-failed.json`.
  The canonical retry widens the stop window to 768 inside the same 2,048-token
  context; it does not accept a length stop as equivalent to `<|im_end|>`.
- Attempt 3 reached and retained the `<|im_end|>` token id, but demonstrated
  that vLLM 0.14.1 omits a token-id stop from native `CompletionOutput.text`
  even with `skip_special_tokens=False`. The preserved receipt is
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-3-failed.json`.
  Backend normalization therefore treats generated ids as authoritative,
  reconstructs raw special-token-preserving text with the shared tokenizer,
  and verifies native text equals the decoded prefix before `<|im_end|>`.
- Attempt 4 passed prompt, generation, stop-token, policy-likelihood, raw-replay,
  and process cleanup checks, but the first in-process snapshot still retained
  CUDA allocations through the caller's outer `LLM` reference. The provisional
  receipt is
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-4-provisional.json`.
  Final qualification additionally drops that outer reference, collects it,
  empties the CUDA cache, and requires zero allocated and reserved bytes before
  process exit.
- Attempt 5 proved that vLLM 0.14.1 retains compile/model CUDA tensors until
  interpreter exit even after explicit in-process engine and distributed-group
  shutdown. The failed receipt is
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-5-failed.json`.
  Because canonical inference already uses fresh rank-local workers, the final
  probe now launches the engine in a child worker and requires the parent to
  observe worker exit and no surviving owned process group before publishing
  the qualification receipt. Parent-observed pre/post global GPU memory remains
  diagnostic rather than an equality gate on shared GPUs.
- Attempt 6 passed that narrow runtime probe, but second-round review found its
  positive file allowlist omitted live tokenizer/processor/template assets and
  its image evidence did not bind the submitted in-memory payload. It is
  preserved as
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-6-superseded-partial-identity.json`
  and is not qualification authority.
- Attempt 7 replaced the model allowlist with an exhaustive 19-file snapshot
  manifest, recorded 24 named installed/runtime source owners, independently hashed
  source bytes and RGB pixels immediately before generation and replay, and
  derived the exact returned image-pad range `[4, 1018)`. A third review found
  that lazily imported execution owners were still outside that named set. The
  receipt is preserved as
  `source-studies/receipts/vllm-0.14.1-qualification-attempt-7-superseded-partial-runtime-source-manifest.json`.
- Attempt 8 records every loaded source module from vLLM, Transformers, PEFT,
  and qwen-vl-utils after generation and replay. Its 853-file manifest includes
  asserted owners for the uniprocess executor, GPU model runner, model loader,
  sampler/logits, Qwen3-VL, and multimodal processing. This is the canonical
  processed-runtime, base-family, prompt-projection, and cleanup receipt; it is
  not forced-replay qualification authority.

## Rejected Shortcuts

- Treating vLLM raw default logprobs as existing evaluator scores.
- Treating engine startup as Qwen multimodal parity evidence.
- Calling native LoRA loading equivalent to DoRA.
- Claiming an HF-local image grid was executed by vLLM.
- Using mock vLLM responses as the runtime acceptance gate.

## Accepted Qualification

vLLM 0.14.1 is accepted for the scoped processed offline Qwen3-VL runtime by
`src/inference/qualification_receipts/vllm-0.14.1-qualification.json`. The passed base
receipt
binds probe SHA-256
`e6fa2cabeb0d69fcec14538aa7d39c69d79dbac9a5a2b65dfcf4de9d8c0beb7f`,
the real fixture, exhaustive 19-file source snapshot, 853 loaded package-source
files plus required execution owners, one logical A100, uniprocess engine mode, exact 14-to-1,027
prompt expansion, returned image-pad range `[4, 1018)`, identical generation
and replay source/RGB image hashes, 329 generated tokens ending in retained
`<|im_end|>`, 329 processed policy logprobs, historical prompt-prefill
likelihood evidence, and
post-worker cleanup back to the 3 MiB baseline with no surviving PID or child
process.

Generated-token `raw_model_logprob` authority is separate. The current BF16
forced-decode receipts are
`src/inference/qualification_receipts/vllm-0.14.1-forced-replay1.json` and
`src/inference/qualification_receipts/vllm-0.14.1-concurrency4.json`, covering the only
supported per-device replay concurrencies, one and four. Each binds the current
forced logits processor source, exact prompt and generated-token identities,
stop evidence, raw replay row hashes, source-base fingerprint, probe/config
sources, and the base qualification receipt.

Strict FP32 parity mode is independently bound by
`vllm-0.14.1-fp32-qualification.json`,
`vllm-0.14.1-fp32-forced-replay1.json`, and
`vllm-0.14.1-fp32-concurrency4.json` in the same stable directory. BF16 remains
the supported throughput mode; its evidence is not promoted to strict
cross-backend parity evidence.

The package-level qualification remains intentionally narrow. The separate
`src/inference/qualification_receipts/vllm-0.14.1-application-sources.json` receipt binds
the exact 23 CoordExp execution owners used by the application runtime. The
validator requires that complete path set, rejects missing or unexpected
owners, and rehashes each file before a vLLM session opens. This keeps package
qualification and application-code qualification explicit instead of treating
one as evidence for the other.

Later-wave evidence now covers the application adapter and composed execution
model. `vllm-0.14.1-fp32-distributed-acceptance.json` binds fresh one-rank and
two-rank dual-likelihood smokes plus an eight-rank val200 run through evaluator
consumption. `hf-vllm-val200-step4887-fp32.json` binds the matched current-code
dynamic-HF and materialized-vLLM FP32 val200 runs. HF remains a first-class
dynamic backend; materialized HF is only the execution-model composition
oracle, while vLLM consumes the immutable materialized snapshot.
