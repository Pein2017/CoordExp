# Lead ruling: exact native prefix interface

Status: implementation authorized; native execution remains pending the concrete
CPU candidate. This is a lead decision under the user's autonomous research
grant, not a request for user approval.

The worker may extend `src/qwen/vllm_rollout.py` and its existing focused tests
to support exact token-prefix continuation and one-token native score capture.
Keep `generate` defaults and all current callers' behavior unchanged. A
constructor `max_logprobs` option may default to the installed value20; this
unit alone requests-1. Installed vLLM0.29.0+cu129 source confirms native support
for `TokensPrompt`, sample `logprobs=-1` and engine `max_logprobs=-1`.

Reuse existing image validation, DoRA snapshot identity, device receipts,
engine lifecycle, raw-logprob mode and suffix termination checks. The supplied
input is original unexpanded chat token IDs plus exact generated-prefix IDs;
the processed native input must equal the independently bound expanded original
prompt plus that prefix. Never decode/re-encode or manually expand the generated
prefix. Preserve media/grid identity and checkpoint/norm settings.

Input-token/media/producer mismatch is a fail-closed execution error. By
contrast, a fresh emitted token differing from the historical saved token is
recorded fidelity evidence; do not assert it away, overwrite it or retry until
it matches. Native unforced versus same-token sham disagreement limits the
affected causal comparison under the frozen protocol.

Full-vocabulary scores are requested only for at most four one-token site
measurements. Suffix continuations return no full-vocabulary score trace. Count
these scoring requests/tokens in addition to the at most16 suffix requests.
Persist token-ID mapping, native backend/raw-score mode and normalization scope.
Native log-probability differences equal logit margins at a shared prefix;
call them log probabilities in raw artifacts. Prove complete vocabulary support
before reporting full-vocabulary mass/complements, reject NaN/+inf and missing
support, and preserve legitimate -inf zero-probability entries explicitly.

Required CPU evidence focuses on actual caller behavior: legacy TextPrompt
generation unchanged; TokensPrompt/image-placeholder expansion and exact
prefix equality; injected-token budget/causal score alignment; snapshot and
input mismatch rejection; score payload limited to one-token calls; malformed
or incomplete full-score data rejected. Reuse existing fixtures and tests,
with mutation/sensitivity evidence for load-bearing guards. CPU success does
not establish native seam fidelity; the first released native slice must do so.

The worker owns these added shared runtime/test surfaces for this package and
may include them in its scoped source commit. Report any further compatibility,
scientific-selection or runtime-scope change to the lead before proceeding.
