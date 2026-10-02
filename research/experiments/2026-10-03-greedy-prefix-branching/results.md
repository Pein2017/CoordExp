# CPU candidate: native coordinate branching

CPU preparation is complete; scientific evidence remains unmeasured. No model,
GPU, optimizer or HF replay call was made. The exact native packet awaits lead
review/release, under [lead ruling 01](lead-ruling-01.md). The worker is live
verified `gpt-6.1-sol/high`, thread `01a0fe03-4ce4-7490-b937-5285451b0b11`.

## Certified saved cases

All offsets are zero-based within the sealed fresh OFF generated tokens. Every
selected event is a complete row, prediction-selected before any GT accounting;
all four sites are eligible, distinct, x1 decisions. Legal x1 bins are 0..998.
Each image also has one malformed/censored span, retained as an observed burden
rather than used to replace a declared case.

| Image | Event | Row order | Token offset | Causal logits position | Emitted bin / token | Greedy budget | Supplied budget |
|---|---|---:|---:|---:|---|---:|---:|
|7511|first impossible coordinate|69|626|1945|999 / 152669|2458|2457|
|7511|first complete literal repeat|22|203|1522|630 / 152300|2881|2880|
|351017|first impossible coordinate|161|1507|2868|999 / 152669|1577|1576|
|351017|first complete literal repeat|3|31|1392|0 / 151670|3053|3052|

[CPU contract](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/cpu-02/contract.json)
and [site ledger](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/cpu-02/cpu-sites.json)
retain full prefix identities, row/slot data, original raw identities, bindings,
source acceptance locators and input reports. The label snapshot has all 18
images / 570 labels; the selected images have 44 and 49 labels respectively.
The site consumer reports each image's actual denominator IDs. It does not use
570 as a denominator for a selected-case efficacy rate.

The checkpoint is the frozen original-LR balanced constant endpoint16 at
`lower-lr-balanced-01/constant/native-observation-16-02/checkpoint-16`. All five
checkpoint payload files, identity.json, original input/policy/label bindings,
fresh OFF raw/producer identities and accepted norm-comparison receipts were
verified. The materialized adapter/delta payload is 88,601,909 bytes. This does
not alter or relabel historical evidence.

## Interface and measured CPU checks

`probes.greedy_prefix_branching` reuses maintained native input/media assembly,
strict parser/token mapping, full-label geometry regions, annotation matching,
DoRA resident lifecycle and clean Git source qualification. The shared extension
adds `generate_exact` while preserving legacy `generate`; constructor
`max_logprobs=20` preserves the old setting. This unit proposes `max_logprobs=-1`.

Installed vLLM0.29.0+cu129 exposes `TokensPrompt`, sample `logprobs=-1` and engine
`max_logprobs=-1`. The original unexpanded chat has 349 tokens for both images;
image-placeholder expansion yields bound original prompts of 1320 and 1362
IDs, with 972 and 1014 visual tokens. CPU native assembly exactly reproduced
prompt/media/grid identity. The actual installed token replacement primitive
exactly reproduced expanded original prompt plus literal generated prefix at
all four sites. The shared test exercises this primitive through the actual
`_generate_exact` caller. Native model execution at this seam remains unmeasured.

The bound vocabulary is token IDs 0..152669; coordinate IDs are 151670..152669
in bin order. One-token native scores must cover every bound ID and normalize
to probability one within 1e-4. Raw `-inf` is retained as JSON string `"-inf"`;
NaN, +inf and missing support fail. Scores are native raw log probabilities;
same-prefix differences are logit gaps. Only four scoring requests may return
full-vocabulary payloads. Suffixes return no full score payload.

Final check command:

```bash
python -m pytest -q tests/probes/test_greedy_prefix_branching.py tests/qwen/test_vllm_rollout.py
```

Exit0, **18 passed** in 26.27 seconds; raw stdout/stderr is
[cpu-tests-05.log](../../../outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/cpu-tests-05.log).
The checks cover actual saved sites, first impossible-coordinate causality,
same-token accounting, total budget, exact processed tokens/media, legacy
TextPrompt behavior, native vocabulary/score support, stale snapshot rejection,
source/checkpoint/input bindings and owner separation. The actual producer and
final JSON reader are exercised with a CPU engine double. Historical/sham drift
becomes HOLD and skips alternatives without retry. A false later-free owner
claim is rejected even after re-signing transport artifact hashes.

Preparation exits0/0 for cpu-01/cpu-02; compile and scoped whitespace checks
exit0. Failed check logs 01 and04 remain: the first exposed a short-output test
fixture through `trim_suffix`; the latter's expected error needed updating when
the added score-argmax guard rejected the corruption before normalization.
No genuine runtime failure, silent retry or native result is implied.

## Concrete release and terminal consumer

Qualify the clean committed source once:

```bash
python -m probes.greedy_prefix_branching qualify \
  --contract outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/cpu-02/contract.json \
  --output outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/qualified-contract-01.json
```

The lead freezes an immutable release copy with `native_released=true`, keeping
all scientific fields and source binding. The precise release SHA is then
required by both native entry and readback. If lead-owned source records change
Git HEAD, requalify the clean new commit; no dirty-source bypass exists.

```bash
timeout --signal=TERM --kill-after=30s 1200s python -m probes.greedy_prefix_branching run \
  --contract RELEASE_CONTRACT --contract-sha256 RELEASE_SHA256 \
  --output outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/native-01 \
  > outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/native-01.log 2>&1
python -m probes.greedy_prefix_branching readback \
  --contract RELEASE_CONTRACT --contract-sha256 RELEASE_SHA256 \
  --output outputs/research/physical-fn-recovery/2026-10-03/greedy-prefix-branching-01/native-01
```

Finite bound: one GPU/rank, one sequence at a time, four one-token score requests
plus at most 16 suffix requests; 39,868 newly generated native tokens including
four score tokens. The 12 supplied intervention/sham tokens are counted within
the original ceiling, giving at most 49,344 logical original-output tokens over
16 branches. Maximum processed context4446; maximum scoring prefix2869.
At most20 request prefills and39,848 decode steps (39,868 total generation
forwards before internal engine warmup); no HF forwards or optimizer steps.
KV cache2GiB; historical singleton norm-OFF allocator peak was approximately
6.72GiB. Expected native allocation8-12GiB, process RSS under16GiB, output under
64MiB and wall about5-15 minutes are estimates, not acceptance facts. External
wall limit1200s plus cleanup30s is finite; actual counters, timing/device
receipts and allocator peaks are retained. Shared stress occupancy is not an
idle-GPU gate.

For each site, capture native scores, ordinary greedy full suffix and same-saved
coordinate sham. Compare full token sequences and stop reasons with sealed
historical suffixes, and check scoring/unforced first-token agreement. Any
fidelity disagreement is retained as `HOLD_native_fidelity`; do not run that
site's alternatives or retry controls. Continue other frozen sites. Input,
media, checkpoint, source or score-support failure stops the package, preserving
raw evidence. Otherwise choose the top two distinct structurally legal native
coordinate tokens excluding the saved emitted token, tie-break by token ID,
and continue ordinary greedy after each one-token intervention. GT is absent
from this selector.

Artifacts: per-site `scores.json`, branch `greedy/sham/alternative1/alternative2.json`
and hash-bound `complete.json`; unit `complete.json` contains real request/token
counters and engine receipts. `readback.json` independently recomputes consumer
invariants, metrics and fidelity. Branches contain new producer identities,
exact prefixes, causal offsets, budgets, current-row completion, full native
legal/per-owner/union masses and margins, owner-completable prefix traces,
whole and later-free gains/losses/retention, assisted-row transitions and output
burdens. Independently matched subsets are explicitly nonadditive. Unknown and
annotation-unmatched rows remain neutral. No empty-history/population efficacy,
best-branch promotion or next-unit scheduling is authorized to this worker.
