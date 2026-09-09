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
| Qwen processor/tokenizer/model loading | `src.qwen.runtime_loading.QwenLoadOptions`, `load_qwen_components_from_options` | Device, model lifetime, train/eval mode and selected checkpoint |
| Deterministic scored inference | `src.inference.runtime` and `src.inference.backend` | Input/policy, output interpretation and claims |
| Packed training | `src.training`, `src.runtime`, `src.supervision`, `src.packing` | Selected training config, loss and synchronization contract |
| Simple validated result publication | `src.artifacts.publish_json_exclusive` | Payload meaning and an absent final output path |

Native research operations do not need to manufacture packed sequences or a
strict HF evidence session. Their maintained package examples document exact
history, scoring, continuation and capture usage. The stable research behavior
is owned by the [infra-base contract](../openspec/specs/coordexp-swift-research-probe-infra-base/spec.md).

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

See the [journal contract](../openspec/specs/coordexp-swift-execution-evidence-journal/spec.md),
[admission contract](../openspec/specs/coordexp-swift-research-probe-admission/spec.md),
[inference runtime](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md)
and [inference pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md)
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
