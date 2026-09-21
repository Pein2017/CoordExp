---
doc_id: docs.research-probe-infra-base
layer: docs
doc_type: workflow
status: canonical
domain: research
summary: Direct research execution and optional integrity capabilities at their existing owners.
tags: [research, probes, artifacts, inference]
updated: 2026-09-17
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
| Frozen-case native request reconstruction | `src.inference.bound_requests.build_bound_native_requests` | The already-bound input row/image plan and its identity checks; this is not fresh replanning |
| Qwen processor/tokenizer/model loading | `src.qwen.runtime_loading.QwenLoadOptions`, `load_qwen_components_from_options` | Device, model lifetime, train/eval mode and selected checkpoint |
| Deterministic scored inference | `src.inference.runtime` and `src.inference.backend` | Input/policy, output interpretation and claims |
| Exact multimodal history/replay | `src.qwen.native.prepare_native_inputs`, `prepare_replay` | Literal token IDs, images, model mode/device; invalid shapes and token histories fail |
| Compact batched replay alignment | `src.qwen.native.select_compact_replay_logits` | Batch construction, target masks, reduction and optimizer meaning |
| Budgeted native continuation | `src.qwen.generation.generate_continuations`, `NativeGenerationPolicy` | Per-request budgets, seed/batch order, policy and optional traces |
| Named layer capture | `src.qwen.inspection.CaptureInputs`, `CaptureHiddenRows` | Selected sites and intervention formulas; context exit removes hooks |
| Aligned differentiable scores | `src.losses.token_scores.aligned_token_logprobs` | Causal alignment, masks, reductions, credit and distributed factors |
| Global annotated-owner assignment | `src.eval.assignment.global_matches` | Category/threshold policy; cardinality then quantized IoU, distinct from greedy visualization |
| Native decode to standard detection row | `src.eval.native_rows.native_detection_record` | What is a trusted owner, category policy and scientific scoring |
| Qwen decoder activation checkpointing | `src.qwen.checkpointing` | Whether it is enabled, expected decoder depth, parity criterion and memory/speed tradeoff |
| Adapter-only serialization | `src.adapters.dora.save_dora_adapter_payload` | Trainable-surface choice, optimizer state and checkpoint-selection semantics |
| Deterministic tensor/layout identity | `src.runtime.model_state` | Which tensors constitute a scientific state or acceptance criterion |
| Owned-child completion wait | `src.runtime.process_completion` | Which jobs may launch/retry, GPU allocation, scientific deadline and continuation policy |
| Owned-child spawn, absolute wait and termination | `src.runtime.owned_process` | Explicit command/cwd/environment, resource assignment and recovery policy; only newly owned child handles |
| Singleton native materialization combination | `src.qwen.native.combine_singleton_native_inputs` | Literal ordered prompts and compatible media fields; no research route or mask selection |
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

The [upgrade acceptance example](../openspec/changes/archive/2026-09-12-upgrade-research-probe-workflow/acceptance.md#real-model-acceptance)
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

## Coding-agent rules for maintained probes

These rules are intentionally small and enforceable. They protect the research
semantics from being hidden inside convenience abstractions while preventing a
new direction from copying an old experiment's machinery.

1. **Keep scientific meaning direction-local.** Cohorts, teacher construction,
   masks, owner/admission semantics, loss numerators and denominators, credit,
   update schedules, acceptance thresholds and stop rules stay in
   `probes/<direction>/`. Similar loops are not evidence that these meanings are
   interchangeable.
2. **Give repeated mechanics a concept owner.** When two maintained directions
   need the same operation with the same contract, reuse or add the smallest
   owner under `src/`. A maintained direction must not import a reusable
   execution helper from another direction merely because that experiment
   implemented it first. Prefer a small primitive over a new base class,
   registry, trainer framework or plugin surface.
3. **Separate execution checkout from evidence locations.** Python/module source,
   subprocess `cwd` and executable helper paths derive from the current checkout.
   Never hard-code a sibling worktree as the execution root. Absolute dataset,
   checkpoint, immutable artifact and provenance paths may remain explicit when
   they are part of the scientific binding.
4. **Preserve evidence, not obsolete executable compatibility.** Closed producers
   may be recovered by their recorded Git/source snapshots. Do not keep live
   aliases or duplicate implementations solely so an old command still executes
   from today's tree.
5. **Put maintained tests beside the direction.** New direction regression tests
   live in `probes/<direction>/tests/` and must be discovered by
   `python -m pytest -q probes`. `tests/research/` is not the home for new
   direction-local tests. Shared primitives additionally receive tests at their
   `tests/<owner>/` surface when a direction test does not directly exercise the
   contract.
6. **Mechanical refactors may not silently change scientific invariants.** Byte
   encoding/hash identity, token history, loss denominator, gradient collective,
   owner population, decode policy and stop semantics require an explicit
   migration plus a parity or counterexample test when changed.
7. **Performance claims require measured scope plus parity.** Record the exact
   workload, device count, wall/step metric, memory and behavior-equivalence
   criterion. A faster configuration that misses its declared parity gate is a
   candidate for investigation, not a new default.
8. **Do not generalize on aesthetics.** A new global abstraction needs at least
   two current consumers with the same demonstrated contract and evidence that a
   smaller shared primitive or local composition would not remove the repeated
   work. Historical similarity alone is insufficient.

## Maintained direction entries

- [DORA owner learning](../probes/dora_owner_learning/README.md): Source256 preparation, sampling and CE/RLOO; separate coordinate/full-action scoring.
- [Source/Rweak row crossing](../probes/source_rweak_row_cross/README.md): frozen manifest preparation, native continuation and offline assignment/reduction.
- [Human13](../probes/human13/README.md): output-QP and magnitude finite-panel profiles.
- [Logit lens](../probes/logit_lens/README.md): base, causal, radius/direction and natural continuation profiles.
- `parallel_owner_research`: conditional-credit/composition and owner-preservation lanes.
- `native_owner_scale`: scaled owner supply, state probes and independent evaluation.
- `owner_successor_scale`: credible-successor supply, conditional credit and replay throughput.
- [Training-set completion](../probes/training_set_completion/README.md): acquisition, reviewed teacher construction, CE training, natural readback and physical evaluation; shared replay/distributed owners and historical-source migration boundaries.

Maintained direction tests live beside their package and are discovered by
`python -m pytest -q probes`. Saved-input row-cross checks additionally require
the documented manifest and preserved original-code root. The [acceptance record](../openspec/changes/restructure-research-probe-development/acceptance.md) separates these checks from the one-case Source model smoke.

The remaining `scripts/research` closure supports existing optional admission/evidence consumers, coverage comparison and research navigation checks. Its historical producers and dedicated tests are listed in the [retirement disposition](../openspec/changes/restructure-research-probe-development/retired-files.md); new direction work starts in the four packages above or a new ordinary direction package.

## Source and identity owners after the September 21 migration

See [Output storage policy](OUTPUT_STORAGE_POLICY.md) before creating source captures or run-local files. Shared literal native-input identity lives in `src/qwen/input_identity.py`; completion JSON/path operations live in `probes/training_set_completion/artifacts.py`; saved row accounting lives in `probes/training_set_completion/row_scoring.py`. DoRA loaded-composition checks live in `probes/dora_owner_learning/composition.py`.

`src/artifacts/source_provenance.py` captures source evidence outside outputs. `src/artifacts/source_archive.py` explicitly recovers expected historical bytes without executing them or satisfying current-source launch gates. Ordinary maintained imports replace the old output-directory loaders. Scientifically distinct recipes and the untied payload implementation retain separate owners rather than being collapsed into a universal trainer.
