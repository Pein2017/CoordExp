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


## Released native qualification: completed, pending lead acceptance

The lead accepted implementation `cbbce5e588daeb5a9f3bad7422be3abe0e5e9fdc` and
released one B qualification in `22ea4e1175b8e11fd16b9d8d062eb49769a6207b`
(tree`8654e9ab03a7ec8e182176206080dde3456ae8bf`). All157 source-file contents
were unchanged by that records-only rebind. Exact release SHA256:
`fabc492e6fd3e56b04619b451932d649cd7942cdcf5b4774ba1a234d5564503e`.
The command below ran once; no primary packet, retry, warmup, extra acquisition
or repeated successful readback was executed:

```bash
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/candidate-preparation-02/qualification-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-qualification-B-01
```

Owner wrapperPID2950598, nativePID2950599, torchrunPID2951102, initial
rank0..7 PIDs2951463..2951470, exec session90297. Owner/main torchrun/fresh-reload
torchrun and terminal all exited0; all16 rank completion receipts are present.
The entry performed its one full base/anchor qualification on11 payload files.
Terminal source/stat verification completed before this results update. All
owned processes settled, with no manual kill or replacement; a final `/proc`
check found none. Fresh-reload PIDs were not retained by the runner and were not
reconstructed; its awaited exit0 and eight completion receipts are retained.

Evidence under the unit output root: `native-qualification-B-01/` contains
invocation/process-exit/terminal/readback, rank entry/assignment/composition,
version0/1 acquisition and analyses, update1, checkpoints0/1, and fresh-reload
receipts. Adjacent `native-qualification-B-01-owner.log`, `*-owner-start.json`,
`*-owner-exit.json`, `*-session.json`, `*-launched.json`, `*-cleanup.json` and
`*-worker-summary.json` retain ownership, actual exits, structured observations
and cost/projection details. Direct launch receipt: `transport/native-launched-01.json`.
The summary only reads published metadata; it does not rerun a model or consumer.
Two one-off projections exited1 (an acquisition-summary file treated as a
trajectory, then `full:ID` positive owners compared directly to integer IDs).
Corrected projections exited0 without source/native changes. All570 unique
positive owner IDs then matched the full570 evaluator IDs, with5331 target atoms.

Actual work:36G+18S+18fresh-reloadG+1technical =73generation requests,
19,008generated actions (12,672initial +6technical +6,330reload),18P,
3geometry replays,18D and1six-position diagnostic replay. Both greedy versions
and checkpoints0/1 passed maintained readback; optimizer continuity/roles,
independent delta schema and inventories passed. All eight fresh checkpoint1
loads retained optimizer update1. Reload original IDs and stop reasons matched
saved version1 on18/18 images, including the capped output.

The diagnostic retained exact six original IDs, finite raw/unforced-normalized
likelihoods,1024prompt image positions, zero suffix image/video positions and
restored method binding. Its teacher-forced selection contributed no loss or
backward. Generation1.627131s, replay0.196786s; maximum absolute cached/full
conditional log-likelihood gap0.385347 for both raw/policy, mean0.131742.
Ordinary sampled replay maximum raw gap0.858914, policy0.857063; image-mean
absolute gaps0.027050/0.027068. Full per-action values are preserved in
`rank-*/update-1.json` and the diagnostic. Numeric differences are observations;
no policy/media/position/identity/nonfinite failure occurred.

All18 positive branches and three active geometry branches had nonzero gradients.
Duplicate credit had160nonzero advantage positions on image14038: sampled events0
versus greedy events5 at original actions139/148/166/175/193. Advantages ranged
3/3084..5/3084; its duplicate loss0.487500 and language/input/output gradient
L2 per-image-over18 were0.053455/0.034559/0.039435. Other17D gradients were zero.
Global preclip norm2.538394 agreed across ranks, followed by clip1 and one
continuous AdamW update. This supplies a branch signal, not policy-efficacy proof.

| Greedy observation | Version0 | Version1 |
|---|---:|---:|
| Matched full570 annotation IDs |255|246|
| Annotation-relative FN / FP |315 /98|324 /219|
| F1 |0.552546|0.475362|
| Valid rows / duplicate events |353 /5|465 /119|
| Geometry-invalid rows / malformed outputs |4 /0|226 /1|
| Generated actions / caps |3342 /0|6330 /1|

Version1 gained17 IDs, lost26 and retained229; complete per-image IDs/burdens are
in `metrics.json` and the summary. Image351017 changed326actions/17matches to
3084actions/1match, with113duplicate events and221invalid rows. Its greedy
request took315.376439s and stopped at budget; fresh reload reproduced it.
These are one-update B qualification observations, not the finite16-update
contrast, generalization or physical-FN evidence. No quality gate was added.

Owner wall1078.114521s; launch-to-terminal1066.823509s. Entry import/source/payload
interval10.194447s includes the one hash but does not isolate hash time.
Main torchrun671.301107s; initial readback-publication interval7.360626s;
fresh reload process interval386.468685s; reload readback-publication0.492042s;
terminal stat/source-publication1.196103s. Slowest main rank7 took649.731622s,
with325.365972s skew; its startup-to-composition21.975919s and measured
G0/S0/G1 acquisition sums67.531368/61.794711/351.189500s. Joint branch
replay/loss/backward/gradient-measurement sums across images were20.411907s P,
2.385449s G and13.367146s D; these parallel sums are not wall time, and separate
backward timing was not retained. Checkpoint first-file-write-to-seal spans
0.352030s/0.668057s exclude export work before the first file. Checkpoints
contain88,606,106/266,045,742bytes; native output216files/372,890,528bytes.
Separate finalizer RSS was not retained and was not reconstructed by a rerun.

Maximum main rank RSS13,617,268KiB; reload/owner-child peak13,621,020KiB.
Maximum CUDA allocated9,159,713,792bytes, reserved12,035,555,328bytes;
reload allocated9,062,720,000bytes. Peak sampled length343; a full3084-action
differentiable sampled replay was not acquired. The greedy/reload horizon was
exercised; future full-horizon training memory and event supply remain uncertain.

Proposed primary observation estimate:43,200s per arm (12hours), for lead binding,
never a cap/kill/retry/quality gate. With static ranks, version1-like greedy costs,
version0-like sample lengths/branch costs and measured startup/export/finalizer,
the slowest-rank projection is7088.895249s/arm (about1.97hours). If all99requests
on a three-image rank reach3084 at the observed capped rate, acquisition alone
is31,222.267509s/arm (8.67hours), before larger replay/event/export costs. The
12hour estimate allows uncertainty but is not a guaranteed upper bound; later
lengths, full-horizon sampled gradients, parsing, shared contention and memory
can change. The lead owns native acceptance and exact primary packets. Primary
remains unreleased; earlier CPU fixture exception and unrelated HF failures remain
preserved above.


## Released finite primary A/B: completed, pending lead acceptance

Both released native commands and the saved comparison exited0. Each arm
completed16continuous updates, all18images/full570owners at every positive
update, greedy versions0..16, and checkpoints0/1/4/8/16. Both automatic
readbacks completed. No quality gate, retry, warmup, extra acquisition,
qualification diagnostic, primary reload or repeated readback was added.
B independently restarted the original anchor with fresh AdamW after A's
terminal/source check and confirmed process settlement.

At endpoint16, B has187fewer strict-IoU>.9 duplicate events and115fewer
invalid rows than A, with2more matched annotations. Both trajectories worsen
duplication, category disagreement, censoring and valid-row F1 relative to
the shared baseline. This is a technically completed training-internal
contrast, not scientific acceptance, generalization or physical-FN proof.
Lead ruling03 owns qualification acceptance and primary release; the lead
still owns acceptance and interpretation of this completed package.

### Exact execution and evidence identity

Execution commit249b9b09762457d2f786ab2fbc0c7de677cb5d3f,
treed9964f7969cf6d1e790e7260362452d294d5db2e, clean and unchanged through
both terminal source checks and the comparison. All157bound source contents
equal qualified implementationcbbce5e588daeb5a9f3bad7422be3abe0e5e9fdc.
Only this owned report is changed after those boundaries.

Canonical cwd `/data/CoordExp/.worktrees/research-probes`; bare python selected
ms: Python3.12.11, torch2.13.0+cu129, transformers5.17.0, peft0.21.1,
safetensors0.8.0, flash-attn2.8.4. Output prefix below is
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/`.

- A packet `primary-release-01/primary-A-release.json`: SHA256
  dbd986a588137e1af5625c6141ef90564083e38b2bfd8fc5a177abe754d5b146.
- B packet `primary-release-01/primary-B-release.json`: SHA256
  cbd73df8565a180d47ada462356a0b75f5022b80de7b447ec7b9acec669647a4.
- Original full-label JSON: SHA256
  1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792;
  original input manifest51ad01b2cf599087e4d93b269a0abb9d5c6be83625e9fe9c5f8541731bd962ad.
- Original anchor `/data/CoordExp/outputs/shared/checkpoints/untied-axis001-step2444/payload`: aggregate
  3b168b98f23f5e42b00b6aa7ad8ca5438767bcb8c05f4cc4ce97087d800e0403,
  manifestffc3ac18edb5d284e8cad905326eddda13fd4506c90e099cb40b5f85fe1353b5.
- Frozen protocol unit.md SHA256
  dc5ecdefdbc68be0299c8d8c608d407219ee647db6c982b35a6058b68823976b.
  Each native entry performed its one11-file base/anchor qualification; no
  repeated payload hash was performed for reporting.
- Saved comparison `primary-comparison-01.json`: SHA256
  80ceff9e627381c9b730006f152b119da6d7703dc788ba419dac13cc287e7d36,
  exit0,8.564092s, child RSS1,083,940KiB.

Actual maintained commands (each once, serial A then B then compare):

```sh
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/primary-release-01/primary-A-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01
python -m probes.rule_stability native-run --config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/primary-release-01/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01
python -m probes.rule_stability compare --a-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01 --a-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/primary-release-01/primary-A-release.json --b-run outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01 --b-config outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/primary-release-01/primary-B-release.json --output outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/primary-comparison-01.json
```

Raw identities and exits: each native directory retains invocation.json,
process-exit.json, terminal.json, readback.json, metrics.json, ranks.log,
8rank complete/entry/assignment receipts,16updates per rank and every
original acquisition/analysis. Adjacent `native-primary-{A,B}-01-owner.log`
and owner-start/owner-exit/session/launched/cleanup JSON preserve process
ownership. Each owner/native/torchrun exit0; per-rank complete status is
retained rather than invented separate per-rank shell exit receipts.

A readback SHA256ea8e888a07b20e97730abd76b285154155bbc38bc512e24021846dc8aa70e526;
metrics2669d978f9ffb8a0903c6b38bbde18ce59b9300012712b6d61ce4c00d0f85eea.
B readback77f384ed8bbac29f1818e7694ec55b0997632d13f7e8abe46a00a1e262950139;
metrics8ddba7e2dfbcf6786cb4daeb73433e9b92ad9c519698b6f5ff198ef84a382a06.
`primary-worker-summary-01.json` contains structured work, per-version,
branch, numeric, checkpoint and per-rank resource data.
`primary-numeric-details-01.json` and `primary-sample-burdens-01.json` retain
selected numeric cases and all sampled burden totals without new model work.

### Full trajectory and annotation-relative endpoint

All18baseline original action-ID sequences and stop reasons are identical
between arms. Coverage uses the maintained class-agnostic cardinality-first
one-to-one IoU>=.5 match followed by exact description; duplicate events
separately use strict class-agnostic IoU>.9. The denominator remains570.
Full retained/gained/lost annotation identity pairs, per-image metrics and
adjacent/own-baseline transitions for every version are retained in each
metrics.json and the saved comparison. Endpoint16 is prescribed.

| Version | A matched | B matched | A duplicates | B duplicates | A invalid | B invalid | A/B caps |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 255 | 255 | 5 | 5 | 4 | 4 | 0/0 |
| 1 | 242 | 236 | 169 | 151 | 168 | 152 | 1/1 |
| 2 | 226 | 220 | 334 | 216 | 309 | 454 | 2/2 |
| 3 | 256 | 231 | 439 | 190 | 203 | 530 | 2/2 |
| 4 | 237 | 236 | 529 | 372 | 176 | 609 | 2/3 |
| 5 | 266 | 237 | 358 | 481 | 10 | 471 | 1/3 |
| 6 | 271 | 271 | 353 | 253 | 42 | 193 | 1/2 |
| 7 | 275 | 264 | 367 | 338 | 7 | 53 | 1/1 |
| 8 | 282 | 272 | 370 | 355 | 8 | 54 | 1/1 |
| 9 | 256 | 272 | 574 | 381 | 105 | 54 | 2/1 |
| 10 | 280 | 240 | 358 | 369 | 3 | 288 | 1/2 |
| 11 | 287 | 267 | 353 | 402 | 1 | 19 | 1/1 |
| 12 | 280 | 276 | 347 | 380 | 1 | 23 | 1/1 |
| 13 | 291 | 282 | 347 | 368 | 0 | 21 | 1/1 |
| 14 | 284 | 280 | 347 | 588 | 0 | 105 | 1/3 |
| 15 | 278 | 283 | 428 | 338 | 1 | 8 | 1/1 |
| 16 | 286 | 288 | 562 | 375 | 125 | 10 | 2/1 |

| Endpoint burden | Shared baseline | A16 | B16 | B minus A |
|---|---:|---:|---:|---:|
| Matched | 255 | 286 | 288 | 2 |
| Annotation FN | 315 | 284 | 282 | -2 |
| Annotation-relative FP | 98 | 817 | 624 | -193 |
| Valid-row F1 | 0.552546 | 0.341901 | 0.388664 | 0.046763 |
| Valid rows | 353 | 1103 | 912 | -191 |
| Nonduplicate valid rows | 348 | 541 | 537 | -4 |
| Duplicate row events | 5 | 562 | 375 | -187 |
| Invalid geometry rows | 4 | 125 | 10 | -115 |
| Category disagreements | 1 | 6 | 5 | -1 |
| Malformed outputs | 0 | 2 | 1 | -1 |
| Actions | 3342 | 11516 | 8788 | -2728 |
| EOS outputs | 18 | 16 | 17 | 1 |
| Budget-censored outputs | 0 | 2 | 1 | -1 |
| Longest duplicate burst | 2 | 153 | 174 | 21 |

Own-baseline matched owners: A retained223/gained63/lost32;
B retained230/gained58/lost25. A endpoint change is+31matched,
+557duplicate events, +121invalid rows and+8174actions; B is+33matched,
+370duplicate events, +6invalid rows and+5446actions. B's longest duplicate
burst174 exceeds A's153. Image351017 remains capped at3084 and has only
1/49matched annotations in both arms; its duplicate events are295A/297B.
No aggregate improvement is claimed to establish owner identity or physical
recovery. Endpoint marker/action-family escape counts are zero in both arms;
all per-version family burdens and empty-end dispositions remain retained
separately from decoded-text geometry/validity. Across all17greedy versions
there are390A/1030B empty-legal-end dispositions; these counts are not
substitutes for invalid-row totals or invented repaired histories.

### Training signal and numeric limitations

Every update contains each of570image/annotation owners exactly once,
5331positive atoms, and all8rank optimizer step counts agree with1..16.
A positive/G active image-update gradients:288/37; B positive/G/D:288/60/94.
All active branches supply language, input-delta and output-delta gradients.
Unweighted image-average branch losses across the dose: A P1.561316/G.225998;
B P1.599359/G.386771/D5.356654 (G receives the frozen.1weight).
Global preclip L2 spans1.061265..4.667797A and2.538250..10.837978B;
all32updates invoke the prescribed clip1. Role-specific magnitudes are
retained in the structured summary and raw update receipts.

B's288sample histories contain2duplicate events, versus5187greedy-baseline
events across training versions0..15. Its72835actual-action advantages
contain27794positive/183negative/44858zero values;94image-update replays
are active. Reconstructing inclusive reward-to-go on original completion
action positions gives exactly the saved FP32 advantages (max difference0),
including actual EOS, without length normalization. Events occurred at
rank0/image1584/version3/action143 and rank5/image10707/version13/action195.
This records signal supply; it does not prove a scientific effect.

B cached-generation versus full replay: mean absolute raw/policy action
logprob gaps.029025/.029021; image-mean gaps.028373/.028379. Global maxima
1.609905raw/1.586478policy occur at update6/rank0/image14439/action98,
token151786, with advantage0. Maximum active policy gap.953699 occurs at
update11/rank6/image417044/action75/token152641, advantage.011348898.
All original action identities/likelihood lengths and retained values are
finite. Holding the saved advantages fixed, the largest behavior-versus-
native-replay loss scalar difference is-.338759: image351017/update10,
96.701906behavior versus97.040665replay; mean signed difference across
288images is-.005955. These scalar observations do not bound gradient
error or establish negligible bias. The accepted description remains the
same intended median-normalized policy with numerically approximate replay,
not exact cached-policy gradients. No new tolerance was introduced.

A/B samples all terminated with actual EOS:288each, no sample cap/empty.
Their valid rows total7231/7282, invalid rows345/446, malformed outputs17/16,
duplicate events0/2. Peak sample lengths839/1082 remain below3084: a
full-horizon sampled differentiable backward is still unmeasured. No extra
probe was run to fill that gap. The qualification's six-special-ID technical
diagnostic and fresh checkpoint/optimizer reload remain separate prior
evidence; neither was repeated or injected into primary trajectories.

### Actual work, costs and settlement

Combined1188requests =612G+576S;454567actual generated actions, below
the released3663792bound;576P/97Ggeometry/288D replays and10exports.
Static ranks are unchanged: r0[1584,14439,2685], r1[4134,477415],
r2[7116,16228], r3[13348,5001], r4[13923,7511], r5[14038,10707],
r6[309264,417044], r7[351017,6040,2299]. Each three-image rank has99
generation requests per arm; each two-image rank has66, with image/18
contributions and FP32 SUM. Rank complete/call receipts retain exact layout.

| Measured cost | A | B |
|---|---:|---:|
| Owner wall seconds | 7528.446815 | 7505.694313 |
| Native terminal seconds | 7516.603007 | 7494.232360 |
| Torchrun seconds | 7500.628052 | 7478.818489 |
| Slowest rank seconds | 7480.887927 | 7460.478561 |
| Rank skew seconds | 316.897749 | 311.125279 |
| Entry import/source/runtime/payload seconds | 10.762395 | 10.411897 |
| Rank exit to readback publication seconds | 14.941247 | 14.309208 |
| Readback to terminal source/stat seconds | 1.036087 | 1.104093 |
| Maximum RSS KiB | 13619756 | 13615956 |
| Maximum CUDA allocated bytes | 9161646080 | 9161646080 |
| Maximum CUDA reserved bytes | 12037652480 | 22884122624 |
| Artifact files | 1530 | 1530 |
| Artifact bytes | 1445817097 | 1453133913 |

Total primary owner wall15034.141128s (about4.18hours). Slowest rank7
for both arms. Rank entry-to-composition publication20.657286..24.995404sA
and21.949172..26.100310sB includes model/component startup; hash time is
not isolated from entry import/source/runtime/payload. Summed acquisition
seconds across parallel ranks: A G14716.037216/S7240.008334;
B G16643.143272/S7356.693407. Joint replay/loss/backward/gradient sums
A P314.824605/G31.225784; B P310.605668/G56.185607/D191.031249.
These parallel sums are not wall time; separate replay/forward/backward,
optimizer time and finalizer RSS were not retained and were not rerun.

Each checkpoint retains7payload files and590parameter entries:
588language/1input-delta/1output-delta, with continuous AdamW update
metadata0/1/4/8/16 and readback schema/optimizer validation. cp0 payload
88606106bytes; each trained payload266045742bytes. First payload publication
to seal spans.276024A/.272023B at0 and.660055..672055A/
.664056..672056B thereafter, excluding prewrite extraction. No primary
fresh model reload was authorized; qualification supplied that boundary.
The43200s observation was never treated as a kill/retry/quality gate.

A owner session24275: wrapper3148390/native3148405/torchrun3150648,
ranks0..7PIDs3151658..3151665. B session58916: wrapper3485472/
native3485475/torchrun3486418, ranks3486446..3486453. All known owned
PIDs are absent, no matching native module processes remain, and no signal
was sent. Comparison session78168/owner3563822/child3563825 exited0.
Final holder receipt `primary-holder-release-01.json` records settled
comparison and remaining writers; the worker releases this source holder
after the report commit and direct return. Lead remains the record and
integration/retirement decision owner; no next experiment is scheduled here.

Post-run metadata inspection01 exited1 because denominator identities are
image/annotation pairs rather than scalar IDs; corrected metadata02 exited0
in5.354033s, without native/consumer repetition. B's initial rank-PID
inspection similarly excluded the torchrun parent after its missing RANK
environment caused exit1; corrected inspection exited0. Both raw incidents
are retained. The historical47-pass tiny CPU GPT2 boundary exception, exact
wrong exclusions/two affected nodes, selection-control evidence and unrelated
HF caller failures above remain preserved, with no successful fixture rerun.
The optional wake skill reported stale daemon heartbeat; no monitor/daemon
change or separate Codex runner was created.
