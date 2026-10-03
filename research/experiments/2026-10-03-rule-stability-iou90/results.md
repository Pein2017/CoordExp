# Implementation preparation: no new model result

User decisions are fixed in [the protocol](unit.md):18-image full-label positives,
class-agnostic strict IoU>0.9 overlap events, model-learned behavior, two16-update
arms and coverage observations without a fixed retention threshold. The requested
implementation worker is GPT-6.1-Sol/xhigh. Native execution is not yet released.

Read-only preparation inspected all570 boxes: no degenerate/out-of-range box or
exact within-image box collision;11,036 distinct within-image GT pairs,7,092
same-category and3,944 cross-category. Maximum IoU is0.7878787879, from image14038
book annotations-132/-130; cross-category maximum is0.6503208066. These are input
diagnostics, not prediction-error measurements or proof that0.9 is identity-safe.

Thirteen of18 images overlap the historical val200 image list. This is an exposed
training-internal study. The stronger anchor's historical median-normalized val200
mAP48.0328% selects the starting recipe, not a fresh native result for this unit.

Current status: worker CPU implementation and technical evidence are recorded
below; lead acceptance and native release remain separate in state/rulings.
There is no new research-model inference or training result.


## Worker CPU candidate and evidence boundary

Worker `01a101ac-9ec8-7862-bdb6-38b9cd154673` verified actual
`gpt-6.1-sol`/`xhigh`, canonical cwd, and the direct return route. ENTRY transport
exited0 and was submitted once; it is not acceptance. Receipt:
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/transport/entry-01a101ac-20261003.json`.

The task-local implementation uses the maintained full570 renderer/object-owned
atoms, row-then-image positive reduction, parser/matching/category-credit contract,
seeded generation and untied DoRA/delta loaders/exporters. The only shared source
change is an optional `logits_processor` argument in `src/qwen/generation.py`.
Resident vLLM hardcodes greedy in its maintained request path, so this candidate
uses a resident native HF model per rank for both acquisition channels and replay;
there is no extra head, vLLM sampling assumption, refresh/export per update, or
inference guard.

Geometry follows the lead's appended partial-prefix clarification: certify actual
unambiguous slot actions, preserve already-certified sites after later malformation,
and retain unknown-start/empty-end dispositions. Duplicate events still require
complete valid boxes. Literal sampled token IDs remain authoritative: valid
noncanonical lexical segmentation (`[64,65] -> 'ab'`, versus canonical `[370]`)
does not fail a generated-text retokenization gate.

Median normalization sums BF16 base output rows and FP32 output deltas in FP32,
computes row norms/lower median/factors in FP64, and scales coordinate scores in
FP64 before casting back. HF generation promotes raw scores to FP32 before the
new processor; replay uses that same promotion and graph-connected current-weight
factors. Raw and policy likelihoods remain separate. B retains each cached/replay
action's raw/policy differences rather than rewriting behavior scores. No native
numeric/cache parity or native gradient/memory claim has been measured.

The static maintained8-rank LPT layout is
`[1584,14439,2685] / [4134,477415] / [7116,16228] / [13348,5001] /
[13923,7511] / [14038,10707] / [309264,417044] / [351017,6040,2299]`.
Every local image contribution backpropagates divided by18; one SUM gradient
collective per update yields the image mean despite unequal2/3-image ranks.
Language/input-delta/output-delta AdamW groups use the frozen rates and persist
continuous optimizer state. Checkpoint readback verifies tying, safetensor
shape/dtype, metadata, parameter roles, optimizer tensors/steps and inventory.

### Bounded CPU checks

All paths below are relative to the unit output root
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/`.

| Evidence | Command result and scope |
|---|---|
| `data-contract-green-03.log` | exit0,18passed; all570 real owner atoms, original prompts/grids/media, EOS exclusion, row mean, unequal-rank reduction and category-credit consumer |
| `cpu-policy/green-01.log` | exit0,11passed; differentiable factors, precision/cast semantics, full-support fake generation and separate likelihood channels |
| `cpu-policy/green-02.log` | exit0,21passed/2deselected; affected generation caller paths with substituted computation |
| `cpu-objectives-01/check-04-literal.log` | exit0,22passed;9objective +13maintained token-text checks, strict threshold/cross-category/one-event-per-row, causal shifts, EOS/cap, detached advantages, partial-prefix geometry and literal IDs |
| `compare-arms-green-01.log` | exit0,11passed/18deselected; all17 B-minus-A rows, own-baseline transitions, denominator/annotation identity, no re-scoring or input mutation |
| `cpu-lifecycle-01/checkpoint-consistency-red-01.log` | expected exit1,2failed; a normally sealed checkpoint with contradictory delta metadata or payload was incorrectly accepted |
| `cpu-lifecycle-01/checkpoint-consistency-green-01.log` |8passed after correction; compound pytest/compile command exit0; metadata/payload/parameter/optimizer consistency now enforced |
| `cpu-lifecycle-01/native-boundary-red-02.log` | expected CLI exit1; unreleased qualification-shaped config rejected before output creation/device/model work |

`cpu-policy/hf-callers.log` has an unresolved unrelated existing-caller failure:
exit1,18passed/3failed. The three
`test_position_id_derivation_resolves_first_rope_owner_down_model_chain[0/1/2]`
fixtures expose `_RopeOwner.get_rope_index` without `mm_token_type_ids`, while
unchanged `src/qwen/native.py` supplies that argument. These failures do not
traverse the new generation seam. This candidate does not claim the complete HF
session suite passes and does not change that separate surface.

The first actual8-process entry/write/readback/offline-consumer slice,
`cpu-smoke-B-01`, exited0:54synthetic acquisitions, checkpoints0/1, all18/570,
41.587s owner wall,34.893s torchrun wall, peak rank RSS1,034,940KiB,
11,269,195artifact bytes. That early lifecycle used scalar model/loss stand-ins,
so its evidence is limited to lifecycle/serialization/consumer seams. The stronger
checkpoint consumer was exercised in `cpu-smoke-B-02`, exit0,43.582s owner wall,
36.505s torchrun wall and peak rank RSS1,028,408KiB.

`cpu-smoke-B-03` uses the actual NativeEngine objective/replay/backward caller,
real full570 positive atoms and full152670-vocabulary losses/normalization;
only model scores/decisions and model parameter dimensions are synthetic. All8
ranks exited0 and fresh checkpoint/readback/consumer completed. Work:36greedy,
18sampled,18positive and18duplicate replays; zero geometry replays because the
synthetic greedy outputs contain no illegal sites. Owner wall137.762s, max rank
114.058s, peak rank RSS3,685,012KiB. Counterexample tensor checks cover nonempty
geometry separately. Raw terminal/readback/process-exit receipts preserve exact
statuses and source manifests; these synthetic metrics are not scientific results.

### CPU model-forward boundary exception

The original command was:

```bash
python -m pytest -q tests/probes/test_rule_stability_artifacts.py tests/probes/test_rule_stability_data.py tests/probes/test_rule_stability_policy.py tests/qwen/test_generation.py -k 'not generate_continuations_matches_real_model and not generate_continuations_seeded_real_model' > outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/cpu-lifecycle-01/combined-01.log 2>&1
```

It reported exit0,47passed/2warnings in8.97s, but the exclusion names did not match
the actual two node IDs:

- `tests/qwen/test_generation.py::test_real_generation_mixin_mixed_budgets_exact_histories_and_disabled_traces`
- `tests/qwen/test_generation.py::test_seeded_policy_uses_fresh_config_and_exact_fixed_batch_replay`

Both constructed a tiny CPU GPT2 fixture (vocabulary24, hidden8, one layer) and
zeroed every parameter after construction. No pretrained/research checkpoint was
loaded and no GPU was allocated. The first configured mixed budgets[2,4,0,0]
then separate2/4-action calls; the second configured budget5 and invoked generation
twice. The configured total suffix-action bound is22, not the initially reported
<=4. Actual action vectors/call counts were not retained in the pytest log and
were not reconstructed by rerunning. Existing Git identity at that check was
`8b7a52db99fd3043e73b8d264fa3319f2d1dd9b6`, with uncommitted candidate changes;
no clean source receipt was captured for the combined exception run.

This execution is labeled **boundary exception**, does not satisfy the
CPU-no-model-forward gate, and was not repeated. Direct report receipt:
`transport/cpu-check-boundary-exception-01.json`. The lead acknowledged it and
required exact exclusion controls without granting native release. Collection-only
`python -m pytest --collect-only -q tests/qwen/test_generation.py -k 'not real_generation_mixin and not seeded_policy'`
exited0 and collected9/11 tests with both prohibited IDs absent; preserved in
`cpu-lifecycle-01/generation-selection-collection-01.log`. Direct inspection of
those9 selected definitions found terminal/no-compute, fake `generate` tensors,
and suffix validation, without another GPT2 construction/forward dependency.
No affected legacy tests were rerun to manufacture a clean combined receipt.

No research model/native GPU invocation has been launched. Native loading,
forwards, NCCL, cached/replay numerics, checkpoint behavioral reload and scale
memory/timing remain unproven until an exact lead packet. Worker candidate is
separate from lead scientific/consumer acceptance.


### Final CPU arm consumer and proposed native work

`cpu-smoke-A-01` exited0 with the same actual full-vocabulary objective caller:
36greedy +18sampled acquisitions,18positive replays,0duplicate replays,
0geometry replays; checkpoints0/1 and all8 rank terminal receipts passed fresh
readback. Owner wall139.935s; max rank115.176s; peak rank RSS3,670,172KiB;
11,448,013artifact bytes. B03 retained12,757,213artifact bytes. These are separate
CPU lifecycle measurements with synthetic decisions, not A/B learnability results.

The actual comparison CLI exited0 and published `cpu-arm-comparison-01.json`,
with both versions0/1, endpoint1, each arm's own baseline transitions and
`scientific_evidence=false`. It consumed unchanged completed metrics; it did not
repeat either acquisition/readback. Both final CPU configs bind the same source
file identities and original inputs. Native primary uses all17 versions and
endpoint16, as tested by the bounded comparator.

Clean source qualification and unreleased proposal owner: `candidate-preparation-01/`.
The exact preparation exit, final commit/tree and proposal-file digests are retained
in that folder and the final direct-return receipt; this record does not rewrite
historical CPU source identities. Its
`candidate.json` binds the clean source commit/tree/closure, original label and
manifest bytes, anchor aggregate+manifest identity, runtime and rank layout.
`qualification-B-proposal.json` and `primary-A/B-proposal.json` remain
`released=false`; generating them is not a release. The lead creates exact
release packets from these bodies and owns source rebinding if Git/source or
protocol identity changes. Commands below name those future release files:

```bash
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01/qualification-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-qualification-B-01
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01/primary-A-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01
python -m probes.rule_stability compare --a-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01 --a-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01/primary-A-release.json --b-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01 --b-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-arm-comparison-01.json
```

Qualification is one B update on all18, then terminal greedy, CPU readback, and a
fresh8-rank load of checkpoint1+optimizer with one further greedy per image and
an offline behavioral comparison against saved version1. Acquisition work:
36greedy +18sampled +18fresh-reload greedy =72requests, at most222,048new actions;
18positive +up to18geometry +18duplicate replays. It binds real loading,
NCCL/SUM/optimizer, policy-score traces versus differentiable replay, checkpoint
serialization and fresh native reload. Numeric differences/event supply remain
observations; an absent branch signal is recorded, not assigned a scientific null.
The proposed3600s qualification observation deadline is an initial analytic
observation estimate, not a compute cap, kill timer or relaunch grant.

Primary runs A then B, each independently restarted from the anchor for16
continuous updates. Work totals:612greedy +576sampled =1188requests,
<=3,663,792acquisition actions/decoder forwards;576positive replays,
<=576geometry replays,288sampled replays in B,10checkpoint exports. Each rank
has2/3images, at most3; maximum prompt1372 and replay context4456 at horizon3084.
Per arm,18native processor materializations +18positive encodings occur once;
vision/model replay is recomputed, without assuming reusable vision/backward
state. There are no resident-vLLM refreshes or per-update exports. One flattened
FP32 gradient SUM per update precedes global clip1 and AdamW.

A FP32[3084,152670] score slab is1,883,337,120bytes. Raw/processed generation
histories and differentiable replay/probability slabs can coexist; qualified
native peaks must be measured. Output norm factors use about1000x2048 FP32
effective rows plus FP64 norm inputs. Conservative disk estimates are3GB for10
checkpoints including optimizer and1.5GB for raw traces/diagnostics; these are
analytic estimates, not measured native upper guarantees. Logs retain startup,
per-request acquisition, per-branch replay/backward, checkpoint inventory,
max-rank wall/RSS/CUDA memory, owner/readback/cleanup costs.

Primary observation deadlines are deliberately unset in unreleased proposals.
After qualification, the lead should bind them using the slowest rank's measured
per-action acquisition rate (extrapolated to3084), positive/geometry/duplicate
replay costs and actual load/export/finalizer costs, with explicit uncertainty
for later-length and event-supply changes. This is an operational observation
boundary and never a quality stop or permission to retry.

The persistent worker owns the released torchrun parent and its descendants,
actual exit/terminal/readback records, in-scope diagnosis and cleanup of only
confirmed descendants. Torchrun settles sibling ranks on rank failure; parent
records the actual nonzero exit and preserves all partial artifacts. No automatic
relaunch, warmup, duplicate successful readback or packet extension is permitted.
A timeout means observe/reconcile that same invocation. OOM/nonfinite/identity/
position/policy/distributed failure is technical failure, never a scientific null.
No native invocation has been released or started by this candidate.


### Source continuity and finalizer measurements

The production-shaped A01/B03 slices share source commit
`47f3a3cd27df04fd302633e04f0f29024e2aa8b4`, source-diff SHA256
`faa1b890f1a2d3f083cfde61def9f62cd16b02ac74485e8d168068944d02e828`
and156 source-file hashes, retained in each `cpu-smoke-*-config.json`. Their
protocol SHA256 was
`191e5d06ae2153d904a441617062274a1fa810f10a0aea873b11631086b6fbb0`.
The later lead clarification in `7951b7e52` and task-local marker/escape repair
are separately qualified below. Earlier successful acquisition/readback work
is preserved and was not rerun to relabel those source identities.

B03 torchrun cost131.611s; post-rank readback/source-check interval6.151s.
A01 torchrun cost133.841s; post-rank interval6.094s. These intervals include
checkpoint inventory, offline metrics, publication and source verification;
they do not isolate CPU readback RSS or a native finalizer estimate. Both runs
retain180 files. Process-exit and terminal receipts record actual exit0 and
settled rank processes. Native finalizer memory/timing and the additional fresh
checkpoint reload cost are still unmeasured qualification targets.

Frozen input bindings: full-label bytes
`1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792`;
original manifest bytes
`51ad01b2cf599087e4d93b269a0abb9d5c6be83625e9fe9c5f8541731bd962ad`;
anchor manifest file SHA256
`ffc3ac18edb5d284e8cad905326eddda13fd4506c90e099cb40b5f85fe1353b5`
and anchor aggregate digest
`3b168b98f23f5e42b00b6aa7ad8ca5438767bcb8c05f4cc4ce97087d800e0403`.
The aggregate digest and manifest-file digest have different meanings. Full
base/anchor byte qualification occurs once at each released execution boundary,
with terminal size/mtime checks, rather than per rank or per implementation fix.


### Final candidate surfaces and release discipline

Owned changed surfaces are `probes/rule_stability/{__init__,__main__,artifacts,
consumer,data,objectives,policy,runner}.py`, four
`tests/probes/test_rule_stability_{artifacts,data,objectives,policy}.py` files,
the opt-in generation seam, `probes/README.md`, and this technical report.
Lead-owned unit/state/index/catalog and shared checkpoint assets are preserved.
The final source-qualified candidate/proposals are generated after an explicit
owned-path local commit. No push, reset, clean or lifecycle unlock is performed.

Prepare command (CPU artifact qualification; no model construction/forward):

```bash
python -m probes.rule_stability prepare --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-01
```

Its exact argv, actual exit and wall/RSS/artifact cost are retained in
`candidate-preparation-01/preparation-exit.json`; the bounded log is
`candidate-preparation-01.log`. The clean source envelope in `candidate.json`
is the authoritative final commit/tree and self-including source closure,
with no remaining source diff. Proposals bind that source, the current lead
protocol, original input bytes, anchor and installed runtime.

Any subsequent Git commit, dirty record/source edit, runtime/input/protocol
change invalidates these clean-source packets until the lead binds a fresh
exact source-qualified release. During a released invocation, keep versioned
research-record edits out of the bound checkout until terminal verification.
A release requires `released=true`, the exact declared mode/arm/output, and
an operational observation deadline. Proposal files remain false. Initial
qualification does not authorize primary by transport completion alone.
On a failed invocation, preserve logs and partial outputs; do not reuse an
output path or retry from a timeout. The lead releases any repaired-source
replacement invocation through a fresh exact packet.


### Accepted decoded-text and actual-action clarification

The lead accepted optionB in `7951b7e521fb6f1e85939f3745a7ea4e6d57754f`.
The maintained decoded-text parser defines complete valid boxes. Ordinary token
pieces spelling wrappers/coordinates remain duplicate-eligible; canonical special
IDs are not an additional validity gate. True full-decode/saved-text corruption
still fails. Original actions and likelihoods are retained without retokenizing.

`trajectory_analysis` now maps each event to the earliest original causal action
that completes its accepted row's closing marker. Ordinary token1784 decodes to
`><`, so the completion endpoint may lie inside that action's text. Later suffixes
leave the completed event unchanged. Multiple row events at one action retain
multiplicity in reward-to-go. Per-row action-family fields and separate
`action_family_burdens`/dispositions distinguish ordinary spellings from decoded
validity, geometry and coverage; they do not silently relabel rows as malformed.
All per-image acquisition analyses remain inspectable at every greedy version.

At an unambiguous literal coordinate slot, emitted EOS/wrapper/non-coordinate
actions receive a type-error G site, then later slot certification stops until a
new unambiguous frame. A capped prefix without a next action has no invented site.
Genuine coordinate999 remains an illegal start and preserves encountered empty-end
dispositions. Ordinary wrappers may establish a frame when their completion ends
at an actual action boundary; a crossing box-start action makes that frame
unavailable rather than supplying a repaired prefix.

Preserved falsifications: `cpu-objectives-01/marker-escape-counterexample.json`
contains three valid ordinary-spelling histories rejected by the prior alignment
gate; `geometry-boundary-counterexamples.json` records skipped actual EOS/close
and guessed slot advancement. The interim `check-05-ruling-b.log` exited1 on a
stale test expectation that ignored an emitted wrapper; that expectation was
corrected to the newly explicit actual type-error contract.

Final explicit selection:

```bash
python -m pytest --collect-only -q tests/probes/test_rule_stability_objectives.py
python -m pytest -q tests/probes/test_rule_stability_objectives.py
```

Collection exited0,14nodes, with both `real_generation_mixin` and `seeded_policy`
absent (`cpu-objectives-01/collect-06-ruling-b.log`). Fixture inspection: this file
uses maintained `frontend(load_model=False)` and tensor/tokenizer operations;
`tests/conftest.py` only binds local paths/modules and has no model fixture.
Execution exited0,14passed/3warnings in7.65s (`check-06-ruling-b.log`). It covers
the required before/after closing action, boundary-crossing action and stable
suffix, ordinary coordinate validity plus separate type/family burden, capped
coord10 versus actual EOS, ambiguity stop, shared-action reward multiplicity,
nonadditive Unicode byte pieces and true full-decode corruption. Compilation
and source diff checks exited0. No successful tiny-model fixture, native job or
previous acquisition/readback was repeated for this repair.

Tokenizer-only full-horizon cost check exited0; raw records/analyses/costs are
retained in `cpu-objectives-01/mapping-ruling-b-cost.json` (248,812bytes). For
T=3084, canonical mapping took0.022293s with1full+2frame decodes/6,167ID visits;
ordinary-close fallback took1.289298s with1full+3084prefix+2frame decodes/4,763,227
ID visits. Peak process RSS was1,044,738,048bytes,196,608bytes above the already
loaded frontend peak. These are CPU tokenizer measurements, not model/runtime
qualification. With F certified frame candidates, canonical visits are bounded
by T+F*T; fallback adds at most Tprefix decodes and T*(T+1)/2ID visits. Only one
decoded prefix and integer metadata are stored, rather than all prefix strings.
The observed fallback cost is included in native artifact/finalizer uncertainty.

The candidate is worker-qualified CPU implementation, awaiting lead final consumer
acceptance and exact native release. Original boundary exception and unrelated
three HF fixture failures remain explicitly recorded above. No research-model
load, GPU allocation, native acquisition, training or model forward is released.


## Post-review full-support repair: preparation held

The lead's candidate review found two concrete reachable-action blockers.
Candidate `9c1b3d11d04026dd93133f0d5f4912a0be62aa1d` and
`candidate-preparation-01/` remain immutable historical evidence; source changes
invalidate its exact proposals. Native remains unreleased. The separate visual
image-token replay design is lead/Astra-owned and pending a concrete brief.
No corrected preparation or native/model/GPU work proceeds before that repair.

The PAD case is repaired at the task-local `NativeEngine.generate` caller by
setting maintained `allow_pad_tokens=True`. Shared defaults remain unchanged.
The tokenizer's PAD151643 differs from requested EOS151645; sampled PAD before
EOS/budget is an actual action, while suffix padding after EOS/budget is removed.
No mask, new stop rule or action remapping is introduced. Maintained
`padded_histories` derives its attention mask from supplied row length, so an
actual PAD in a literal replay history receives attention1.

Affected explicit node:
`tests/probes/test_rule_stability_policy.py::test_native_caller_preserves_sampled_pad_and_eos_actions`.
Collection-only exited0,1node, with both prohibited native-generation fixture
names absent (`cpu-pad-actions-01/collection.log`). Fixture inspection confirms
a plain fake computation callback, CPU score tensors, substituted CUDA autocast
and false CUDA availability; no model construction/load/forward. It calls the
actual `NativeEngine.generate` and maintained continuation/likelihood consumer,
then `exact_history_inputs` with a fake position-index operation.

RED: explicit node execution exited1,1failed/2warnings in6.46s, with the original
`hf_backend.unexpected_pad_token` rejection (`cpu-pad-actions-01/red.log`).
GREEN after the one-line caller opt-in: exit0,1passed/2warnings in7.06s
(`cpu-pad-actions-01/green.log`). The fake emitted payload is
[151643,151645,151643,151643]; the assertions retain actual [151643,151645],
stop `im_end`, exactly two finite raw/policy likelihoods, original replay IDs,
attention [1,1,1] for prompt/PAD/EOS and gradients for both sampled actions.
Separate assertions preserve a genuine PAD at budget exhaustion, remove batch
padding beyond that budget and reject non-padding content after actual EOS.
Earlier lifecycle/readback/model-fixture checks were not repeated.


## Corrected CPU candidate after full-support review

The lead/Astra implementation brief resolves the visual replay design hold.
Root committed its state/ruling record in `891ce1677a1976339516d21ee6b195f9d86389ed`.
The repaired source and new proposals belong to `candidate-preparation-02/`;
its `candidate.json` and `preparation-exit.json` bind the corrected clean source
commit/tree/closure and actual preparation cost. Preparation-01 and its candidate
remain historical and cannot authorize this changed source. Native remains
unreleased. The lead retains final acceptance and release ownership.

Maintained `exact_history_inputs(..., prompt_only_media=True)` is an explicit
singleton-image opt-in. It verifies the unchanged original prompt prefix,
single image grid and image modality count, reuses original prompt modality
types when supplied, and appends zero types for every generated action. Explicit
types reach `derive_position_ids`/the bound `get_rope_index`; legacy defaults
remain unchanged. Actual generated image/video/delimiter/PAD/EOS IDs and causal
likelihood positions are retained. This adds no actual video-input capability.

`probes/rule_stability/replay.py` scopes only the real rope owner's
`get_placeholder_mask` around one replay. The original bound method receives
the validated prompt ID/embedding slices and unchanged feature objects; false
suffix masks extend its outputs to the full history. Original feature-count
checks, scatter, visual positions, DeepStack injection and gradient graphs remain
owned by the maintained forward. The exact prior instance/class attribute is
restored in `finally`, including after validation/forward exceptions. Normal
replay does not synchronize diagnostic mask counts.

CPU evidence uses installed Qwen boundary methods on a plain fake owner and
synthetic tensors; it never constructs a GPT2/Qwen model or invokes a real model
forward. The saved baseline oracle (`cpu-media-replay-01/baseline.log`, exit0)
catches the legacy second-grid `StopIteration` and extra-image feature mismatch.
Collection-only checks exclude both prohibited model fixture names. The initial
affected replay selection exited0,21passed/2warnings in6.82s (`green-01.log`);
after the optional diagnostic receipt/None-mask change, only its two affected
nodes ran, exit0,2passed/20deselected in6.25s (`green-02.log`). The current file
has22 nodes. Assertions cover original/suffix IDs and modality types, image/video
and vision-delimiter suffixes, grid/prefix/type failures, prompt feature-count
failure, success/exception restoration, prompt scatter/DeepStack shapes, and
nonzero synthetic feature/DeepStack/causal action gradients. Supplied prompt
types are not recomputed from token IDs.

The qualification-only diagnostic calls the actual generation entry and corrected
replay seams, with model computation substituted in its CPU check. The ordered
processors are median normalization, then six-action selection. Selection records
each selected unforced normalized conditional likelihood before forcing; saved
raw conditionals also remain separate. Forced-selection likelihoods are zero in
the fake and are never policy likelihoods or `L_dup` inputs. Original six actions
are PAD151643, image151655, video151656, vision-start151652, vision-end151653,
EOS151645, resolved from the bound tokenizer/config. Exactly one request and one
six-position replay run on rank0/image1584/anchor0 before qualification training.
The receipt is teacher-forced technical evidence, with no backward, optimizer
contribution, acquisition trajectory or scientific row/coverage metric.

Exact new selections (each collection exited0 and excluded
`real_generation_mixin`/`seeded_policy`; fixture dependencies are fake callbacks,
plain CPU tensor math and path-only `tests/conftest.py`):

```bash
python -m pytest --collect-only -q tests/probes/test_rule_stability_policy.py::test_qualification_six_action_diagnostic_uses_unforced_scores_without_training
python -m pytest -q tests/probes/test_rule_stability_policy.py::test_qualification_six_action_diagnostic_uses_unforced_scores_without_training
python -m pytest --collect-only -q tests/probes/test_rule_stability_artifacts.py::test_release_keeps_technical_work_in_qualification_only tests/probes/test_rule_stability_policy.py::test_technical_processor_order_and_causal_prefix_have_teeth
python -m pytest -q tests/probes/test_rule_stability_artifacts.py::test_release_keeps_technical_work_in_qualification_only tests/probes/test_rule_stability_policy.py::test_technical_processor_order_and_causal_prefix_have_teeth
```

`cpu-technical-suffix-01/collection.log` has1node; `check-01.log` exited0,
1passed/2warnings in6.43s. It checks exact six IDs, PAD attention, original prompt
image mask, false suffix masks, method restoration, six causal score positions,
separate unforced scores and zero training/backward counters. Fake raw/policy
numeric gaps are0 because the substituted scores agree; this is no native numeric
claim. `selection-02.log` has3nodes; `check-02.log` exited0,3passed/2warnings
in6.92s. Counterfactual processor order changes the retained unforced likelihood,
altered cached prefixes fail, and packet validation rejects additional diagnostic
requests or a primary diagnostic before output creation.

Revised qualification work is36G+18S+18fresh-reloadG+1technical request =73requests,
at most222,054 generated actions. It retains18P, at most18G geometry replays,
18D and adds exactly one six-position full-history diagnostic replay. Primary
work remains1188requests/at most3,663,792 actions,576P/at most576G/288D,
17greedy versions and10checkpoint exports. Rank layout, materialization bounds,
score-slab/disk estimates and cleanup/retry rules above remain applicable.
The extra diagnostic uses existing resident rank0 inputs/model; six local
score captures and one mask receipt have bounded storage. Native cached handling,
raw/median score differences, finite positions/media, CUDA/NCCL peaks, runtime,
checkpoint behavioral reload and finalizer costs remain qualification targets.
Cached/full numeric differences are retained observations; count/position/media
corruption, unsupported handling or nonfinite values fail qualification.

Replacement future commands use the same invocation/output ownership with fresh
source-bound release files (these files do not exist until the lead releases them):

```bash
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/qualification-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-qualification-B-01
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/primary-A-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01
python -m probes.rule_stability compare --a-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01 --a-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/primary-A-release.json --b-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01 --b-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-arm-comparison-01.json
```

No prior successful lifecycle/readback or real model fixture was repeated. The
47-pass boundary exception, exact mistaken exclusion command, two tiny CPU GPT2
node IDs and configured22-action bound remain recorded above; actual vectors/call
counts were not retained, and that execution does not satisfy the CPU no-model-
forward gate. Earlier unrelated three HF fixture failures also remain recorded.
No native/model/GPU invocation is started by this corrected CPU preparation.
Narrow compilation of the nine changed Python files and `git diff --check`
exited0; `cpu-candidate-repair-01/static-checks.json` retains paths, status and
0.054786s check duration. No complete suite, acquisition or model fixture reran.
