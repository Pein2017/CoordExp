---
doc_id: docs.research-probe-infra-base
layer: docs
doc_type: workflow
status: canonical
domain: research
summary: Direct research execution and optional integrity capabilities at their existing owners.
tags: [research, probes, artifacts, inference]
updated: 2026-09-09
---

# Research Probe Infrastructure Base

Maintain shared mechanics directly in `research-probes`. Experimental profiles
live in ordinary `probes/<direction>/` packages. The
[branch/worktree policy](BRANCH_AND_WORKTREE_POLICY.md) owns lifecycle;
capability selection never performs worktree or branch operations.

## Ordinary execution

Start with the model/data operations the experiment needs and a small explicit
configuration. Record the actual configuration, code revision and dirty status,
input/model locations, output location and seed where relevant. Dirty status is
not a promise of exact replay: save effective source/configuration when a result
will be cited or replayed.

No journal, admission dossier, per-source hash chain or completely clean Git
worktree is required merely to run an exploratory producer. A profile is a
local scientific configuration, not a global runtime class or registry.

| Need | Existing owner | Caller keeps explicit |
| --- | --- | --- |
| Direction-local V1 inference profile | `src.config.inference.load_research_infer_config` | Scientific values and actual native generation policy; this loader does not enable debug or relax value validation |
| Selected prompt and annotated-target planning | `src.inference.inputs.plan_examples` | Ordered rows, resolved profile, target length; pixel materialization remains explicit |
| Qwen processor/tokenizer/model loading | `src.qwen.runtime_loading.QwenLoadOptions`, `load_qwen_components_from_options` | Device, model lifetime, train/eval mode and selected checkpoint |
| Deterministic scored inference | `src.inference.runtime` and `src.inference.backend` | Input/policy, output interpretation and claims |
| Exact multimodal history/replay | `src.qwen.native.prepare_native_inputs`, `prepare_replay` | Literal token IDs, images, model mode/device; invalid shapes and token histories fail |
| Budgeted native continuation | `src.qwen.generation.generate_continuations`, `NativeGenerationPolicy` | Per-request budgets, seed/batch order, policy and optional traces |
| Named layer capture | `src.qwen.inspection.CaptureInputs`, `CaptureHiddenRows` | Selected sites and intervention formulas; context exit removes hooks |
| Aligned differentiable scores | `src.losses.token_scores.aligned_token_logprobs` | Causal alignment, masks, reductions, credit and distributed factors |
| Global annotated-owner assignment | `src.eval.assignment.global_matches` | Category/threshold policy; cardinality then quantized IoU, distinct from greedy visualization |
| Packed training | `src.training`, `src.runtime`, `src.supervision`, `src.packing` | Selected training config, loss and synchronization contract |
| Simple validated result publication | `src.artifacts.publish_json_exclusive` | Payload meaning and an absent final output path |

Native research operations do not need to manufacture packed sequences or a
strict HF evidence session. Their maintained package examples document exact
history, scoring, continuation and capture usage. The stable research behavior
is owned by the [infra-base contract](../openspec/specs/coordexp-infras-research-probe-infra-base/spec.md).

## Small input inspection and shared/profile boundaries

Inspect arbitrary selected rows using an existing profile without adopting the
Source256 population or loading model weights:

```bash
python -m probes.dora_owner_learning.inspect \
  --input tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl \
  --count 2 --seed 7 --target-max-length 12000 \
  --output /tmp/probe-input-inspection.json
```

The output path must be new. `--ids ID1 ID2` preserves the requested order;
`--count N --seed S` samples without replacement. Unknown/duplicate IDs,
oversampling and malformed sources fail. `--config` selects an existing valid
profile; `--input` explicitly identifies the inspected data without editing that
saved profile. The result records both the configured source and actual selected
source. Omit `--target-max-length` for generation-only inspection.

Generation prompt IDs and annotated targets are separate views. Target positions
refer to the full encoded sequence, including image expansion; EOS and ignored
post-EOS tokens retain their existing span types. Changing target order may leave
generation prompt IDs unchanged. The existing strict `preflight` entries retain
their original source, cohort and predecessor-evidence gates.

For direct Python composition, load one processor-only frontend and select the
rows once. `plan_examples(rows, config=config, components=frontend.qwen,
target_max_length=12000)` returns ordinary values with `rendered`, `image`,
`prompt`, `request` and `target` fields. Request-only consumers do not allocate
pixel tensors. Use the existing `prepare_native_inputs` explicitly to materialize
the request(s); reuse that result for subsequent replay. Batch-one replay uses a
batch-one native input mapping, rather than slicing a multi-image pixel payload
as if its first dimension were the example dimension.

```python
from src.qwen.native import prepare_native_inputs, prepare_replay
from src.losses import aligned_token_logprobs

# `plan` is one selected annotated plan; `model` and its lifetime belong to the profile.
native = prepare_native_inputs(frontend.qwen.processor, (plan.request,),
                               device="cpu", record_media_identity=True)
action_ids = tuple(token for span in plan.target.supervised_token_spans
                   for token in span.token_ids)
replay = prepare_replay(model, native.inputs,
                       prompt_token_ids=native.prompt_token_ids[0],
                       continuation_token_ids=action_ids)
logits = replay.aligned_logits(model(**replay.inputs).logits)
token_logprobs = aligned_token_logprobs(logits, replay.target_ids)
```

The compact Source256 profile validates that those supervised spans are
contiguous. A profile with a different supervision layout must retain its own
mask/history logic. Token scoring deliberately selects no reduction or optimizer.
Use the existing Source256 `TrajectoryScorer`, owner-outcome scorer or a
direction-owned objective according to the question. The fixed Source256
language-DoRA binding is shared within its DORA runtime; its 588 tensors and
18,006,016 scalars are a profile contract. Other small surfaces remain explicit
selections through the public adapter API or existing Human13 magnitude surface.

| Shared mechanics | Profile-owned meaning |
| --- | --- |
| Render/image-plan reuse, prompt construction, exact encoding and native replay | Cohort, ordering choice, target length, intervention and generation policy |
| Existing parameter objects and a fixed profile's repeated binding | Trainable surface choice, optimizer, loss/mask/reduction, DDP synchronization |
| Existing `owner_change` / `aggregate_scores` and branch read functions | Comparator arms, natural versus forced interpretation, denominators and claims |

Read saved branches with `probes.dora_owner_learning.branch_bridge.reduction`
or the applicable existing selective-preservation reducer. These already retain
owner gains/losses and natural/conditional distinctions; missing sampler traces
remain missing. Input inspection is not a new trajectory format or trainer.
Current execution identities include changed shared dependencies; historical
receipts retain their original effective sources. Measured CPU preparation
improvements do not imply faster model forwards or better research outcomes.

The [upgrade acceptance example](../openspec/changes/upgrade-research-probe-workflow/acceptance.md#real-model-acceptance)
preserves an executed selection, exact replay, profile-owned update and paired
diagnostics, plus a repeatable CPU command for reading its saved results.

For one immutable result, use the leaf directly:

```python
from src.artifacts import load_canonical_json, publish_json_exclusive

publish_json_exclusive(result_path, payload)
assert load_canonical_json(result_path) == payload
```

An occupied final path is an error; this API does not overwrite existing bytes.
Mutable intermediate files remain explicitly caller-owned. Do not add another
result writer merely to avoid knowing this distinction.

## Select strict capabilities only when needed

| Need | Public owner | What it protects | What it does not decide |
| --- | --- | --- | --- |
| Durable work across process attempts | `src.artifacts.ExecutionEvidenceJournal` | Identity, records, attempts and terminal completeness across interruption | Work-item meaning, retry policy, scheduling or scientific success |
| Declared CPU-to-model mechanics gate | `src.artifacts.ResearchProbeAdmission` and its binding operations | Exact selected bindings, reserved outputs and two-stage mechanics admission | Launch authorization, cohort, objective, outcome or continuation decision |
| Stable cached/exported inference execution | Existing inference context/model-export owners | Worker transport, cache identity, merge/fold and persisted artifacts | A requirement that ordinary dynamic HF startup use the same machinery |

These optional APIs keep their strict contracts. Ordinary producer simplicity
does not weaken collision checks, exact recovery identity, finite values,
geometry/token alignment, gradient scaling or persisted compatibility. The
artifact facade remains lazy so JSON-only callers do not initialize models.

See the [journal contract](../openspec/specs/coordexp-infras-execution-evidence-journal/spec.md),
[admission contract](../openspec/specs/coordexp-infras-research-probe-admission/spec.md),
[inference runtime](../openspec/specs/coordexp-infras-infer-config-runtime/spec.md)
and [inference pipeline](../openspec/specs/coordexp-infras-infer-pipeline/spec.md)
for the selected API and failure behavior. Do not copy the full typed admission
surface into a producer that only needs to save one result.

## Sharing and acceptance

Share demonstrated repeated operations at their concept owner. Keep scientific
thresholds, reductions, interventions and stop rules in direction code. Match
assignment, greedy visualization and official COCO scoring are distinct
contracts even when all use IoU. Differentiable replay and inference-only
evidence likewise have different gradient and output requirements.

Use a real retained caller and the smallest decisive test. Historical files
alone do not justify a general framework, and experiments need not be running
to justify reuse. No global coordinator, scheduler, optimizer transaction layer
or hook/plugin registry follows from sharing tensor or model operations.

Leaf publication can be checked with a temporary CPU result and occupied-path
counterexample. Recovery uses an actual interrupted/reopened producer.
Model-facing changes need the applicable native/adapter consumer checks;
helper tests alone do not establish real-model parity. Record that evidence
boundary rather than adding ceremonial receipts.

## Maintained direction entries

- [DORA owner learning](../probes/dora_owner_learning/README.md): Source256 preparation, sampling and CE/RLOO; separate coordinate/full-action scoring.
- [Source/Rweak row crossing](../probes/source_rweak_row_cross/README.md): frozen manifest preparation, native continuation and offline assignment/reduction.
- [Human13](../probes/human13/README.md): output-QP and magnitude finite-panel profiles.
- [Logit lens](../probes/logit_lens/README.md): base, causal, radius/direction and natural continuation profiles.

Each README supplies real inputs and the cheapest CPU entry. Run `python -m pytest -q probes` for their joint regression suite; saved-input row-cross checks require the documented manifest and preserved original-code root. The [acceptance record](../openspec/changes/restructure-research-probe-development/acceptance.md) separates these checks from the one-case Source model smoke.

The remaining `scripts/research` closure supports existing optional admission/evidence consumers, coverage comparison and research navigation checks. Its historical producers and dedicated tests are listed in the [retirement disposition](../openspec/changes/restructure-research-probe-development/retired-files.md); new direction work starts in the four packages above or a new ordinary direction package.
