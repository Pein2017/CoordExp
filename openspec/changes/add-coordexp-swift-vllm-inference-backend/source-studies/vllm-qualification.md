# vLLM Qualification Source Study

## Scope

Qualify the installed inference boundary before implementation claims support.
The active environment currently provides vLLM 0.14.1, Transformers 4.57.1,
PEFT 0.17.1, Torch 2.9.1, and qwen-vl-utils 0.0.14.

## Static Findings

- vLLM registers Qwen3-VL as a multimodal MRoPE generative model.
- `SamplingParams.logprobs` returns selected-token likelihood evidence;
  prompt logprobs can provide teacher-forced continuation likelihoods.
- Engine `logprobs_mode` defaults to raw likelihood and must be explicitly set
  to `processed_logprobs` to preserve current CoordExp score semantics.
- vLLM 0.14.1 and current preview source reject PEFT `use_dora: true`.
- vLLM has no loader for CoordExp selected-token embedding delta payloads.
- Offline `LLM` owns model and multimodal processing; the current HF tensor
  `DecodeRequest.model_inputs` cannot be reused as a backend-neutral request.
- The current outer worker contract already isolates one parent-visible GPU as
  logical `cuda:0`; initial vLLM execution should use TP=1 and DP=1.
- vLLM raw replay requires a prompt-logprob request over prompt plus generated
  ids and may require one ignored generated token because zero-token generation
  is unsupported.
- The canonical single-image fixture has 14 unexpanded input ids with one
  image placeholder and 1,027 executed ids with 1,014 image placeholders.
  Initial generation and replay must submit the unexpanded form plus media and
  compare vLLM-returned ids against the expanded form; replaying expanded ids
  with media would expand the first placeholder a second time.

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
5. Prompt-logprob replay aligned to every generated token, including stop.
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

The canonical passed receipt for a qualified candidate lives at
`source-studies/receipts/vllm-<version>-qualification.json`. It MUST record
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
  observe worker exit, no surviving descendants, and GPU memory returned to
  the pre-worker baseline before publishing the qualification receipt.
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
  sampler/logits, Qwen3-VL, and multimodal processing. This is the current
  canonical receipt.

## Rejected Shortcuts

- Treating vLLM raw default logprobs as existing evaluator scores.
- Treating engine startup as Qwen multimodal parity evidence.
- Calling native LoRA loading equivalent to DoRA.
- Claiming an HF-local image grid was executed by vLLM.
- Using mock vLLM responses as the runtime acceptance gate.

## Accepted Qualification

vLLM 0.14.1 is accepted for the scoped offline Qwen3-VL backend by
`source-studies/receipts/vllm-0.14.1-qualification.json`. The passed receipt
binds probe SHA-256
`3f2994af75b0efc7a771cbcb1017e7bcbc2ed8a1257e3748ea9227bf3de4fc5b`,
the real fixture, exhaustive 19-file source snapshot, 853 loaded package-source
files plus required execution owners, one logical A100, uniprocess engine mode, exact 14-to-1,027
prompt expansion, returned image-pad range `[4, 1018)`, identical generation
and replay source/RGB image hashes, 329 generated tokens ending in retained
`<|im_end|>`, 329 processed policy logprobs, 329 raw replay logprobs, and
post-worker cleanup back to the 3 MiB baseline with no surviving PID or child
process.

This qualification is intentionally narrow. It does not yet prove the
CoordExp backend adapter, composed execution model, multi-rank runtime, or
matched val200 acceptance; those remain later wave gates.
