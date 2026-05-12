---
doc_id: docs.agent-engineering-constitution
layer: docs
doc_type: agent-guide
status: draft
domain: repo
summary: Lightweight engineering principles and workflow checklist for future Codex agents working in CoordExp.
updated: 2026-05-04
---

# Agent Engineering Constitution

Purpose: give future Codex agents a shared engineering posture for CoordExp.
This is a lightweight constitution, not a rigid lawbook. Prefer these
principles unless a task, experiment, or compatibility constraint gives a clear
reason to deviate.

Scope: applies across code, configs, docs, metrics, evaluation, artifacts, and
research workflow records in CoordExp.

## 1. Core Posture

CoordExp is a research stack, not just an application. Good changes preserve
the ability to reproduce, inspect, compare, and explain results later.

Default posture:

- Make the codebase easy to route before making it clever.
- Preserve executable truth in the repo.
- Keep stable contracts strict and compatibility behavior explicit.
- Avoid legacy support by default; preserve old configs, deprecated APIs, or
  historical code paths only when a task explicitly requires it.
- Prefer small, source-owned changes over broad entrypoint edits.
- State uncertainty and scope instead of inventing certainty.
- Treat benchmark claims as invalid unless scope, config, checkpoint, artifact
  root, and metric surface are clear.
- Keep future agents in mind: a good change should be easier to inspect than
  the change it replaces.

## 2. Authority And Ownership

Every shared concept should have one owner. Other modules may adapt it,
validate it, import it, or re-export it, but should not redefine it.

Use this decision rule:

- If two modules need the same valid values, create or reuse a shared owner.
- If two modules need different behavior for the same value, keep the value
  owned centrally and put local policy near each consumer.
- If a value appears in config, metrics, docs, and artifacts, define the
  canonical spelling once and treat other spellings as aliases.
- If a module is a facade, adapter, compatibility wrapper, or shim, do not add
  new source behavior there unless the task is specifically about that adapter.

Common owner categories:

- Schema owner: validates authored YAML or JSON.
- Runtime owner: resolves support policy, invalid combinations, and wiring.
- Template owner: defines strict rendering and parsing behavior.
- Compatibility owner: reads legacy artifacts, preserves old imports, or repairs
  generated text.
- Metric owner: defines identity, reducer, denominator, unit, aliases, and
  comparability surface.
- Artifact owner: defines durable file names, manifest fields, and provenance
  shape.

## 3. Strict Contracts Versus Compatibility

Strict behavior and compatibility behavior must stay visibly separate.

Use strict paths for:

- Training labels.
- Stable evaluation metrics.
- Parser behavior that affects benchmark comparability.
- Config schemas for current supported surfaces.
- Artifact and manifest contracts.
- Geometry, coordinate, and object-ordering invariants.

Use compatibility paths for:

- Legacy imports.
- Old artifact readers.
- Generated-text salvage or diagnostic parsing.
- Transitional shims between old runtime plumbing and new schemas.
- Migration helpers that should not define new canonical behavior.

Decision rule:

- If behavior can change a result, label, metric, or artifact claim, keep it
  strict and fail fast.
- If behavior exists to avoid breaking old callers, label it as compatibility
  and keep it boring.
- Diagnostic behavior is useful, but it must never masquerade as canonical
  evaluation.

Legacy support default:

- This is a personal research repo, not a public library.
- Do not preserve backward compatibility for old configs, deprecated APIs, or
  historical code paths unless the user, a current artifact contract, or an
  active reproducibility requirement explicitly asks for it.
- Prefer concise current code over compatibility layers for designs that have
  clearly been superseded.
- When removing legacy support, preserve reproducibility by keeping artifacts,
  progress notes, docs, or commit history sufficient to understand old results.
- Compatibility is opt-in; current design clarity is the default.

## 4. Schema, Type, And Container Design

Prefer explicit containers for concepts that cross module boundaries.

Use typed immutable containers when:

- The shape matters to more than one module.
- Validation is meaningful.
- Future agents need to inspect fields quickly.
- The object represents runtime policy, metric identity, prepared examples,
  parser options, artifact metadata, or dataset-side sidecars.

Use dictionaries when:

- Data is at an IO boundary.
- Data is loaded from or written to JSON/YAML.
- The shape is intentionally loose metadata.
- The object is local, short-lived, and not part of a contract.

Container design criteria:

- Name the concept, not the current implementation.
- Keep fields cohesive.
- Avoid catch-all fields unless the boundary is intentionally loose.
- Normalize external data into typed objects early.
- Serialize typed objects back to dictionaries only at artifact or logging
  boundaries.
- Prefer frozen or immutable records for cross-boundary contracts.

Avoid:

- Passing anonymous dictionaries through many modules.
- Containers that mix unrelated concerns.
- Adding a field because it is convenient for one caller but meaningless to the
  concept.
- Treating a compatibility shim object as a source of truth.

## 5. Config Knob Organization

Config sections should answer "who owns this behavior?"

Default section meanings:

- Data config describes inputs, image roots, sampling limits, object ordering,
  and dataset contract.
- Prompt config describes prompt variants and prompt composition.
- Template config describes rendered assistant shape and strict parse contract.
- Objective config describes loss semantics, target construction, and research
  objective variants.
- Packing config describes packing, cache, and sample layout behavior.
- Evaluation config describes parser expectations, metric surfaces, and eval
  comparability.
- Validation config describes fail-fast checks.
- Debug config describes temporary smoke or inspection overrides.
- Framework/runtime config describes model, launcher, training args, deepspeed,
  tuning, quantization, and infrastructure integration.

Decision rule for new knobs:

- If it changes labels or loss semantics, it belongs near objective, template,
  or data.
- If it changes only execution infrastructure, it belongs in runtime/framework
  sections.
- If it changes metric comparability, it belongs near evaluation or metric
  surface config.
- If it is temporary, put it under debug and make it hard to confuse with a
  stable research knob.
- If the behavior should simply be correct, do not add a knob.

Config naming criteria:

- Prefer explicit string modes when more than two modes are plausible.
- Prefer clear booleans only for stable, truly binary behavior.
- Reject unknown keys on strict current schemas.
- Reject obsolete keys on latest schemas.
- Avoid reusing old names for new semantics.
- Keep aliases at boundaries, not throughout the runtime.

## 6. Naming And Vocabulary

Names should encode the layer they belong to.

Recommended vocabulary:

- Use `format` for serialized text shape.
- Use `template_id` for strict render and parse contracts.
- Use `mode` for runtime behavior selected by code.
- Use `variant` for research objective or prompt families.
- Use `parser_mode` for strict, auto-detect, salvage, or diagnostic parsing.
- Use `metric_surface` for comparability boundaries.
- Use `artifact_surface` or explicit suffix names for output variants.
- Use `legacy`, `compat`, `obsolete`, or `deprecated` only when the behavior is
  truly non-canonical.

Decision rule:

- If a future agent could confuse a value with a format, template, objective,
  runtime mode, and parser mode, rename it or document the layer.
- If a name appears in a metric key, config value, artifact name, and doc, make
  its canonical spelling explicit.
- Prefer boring, searchable names over clever abbreviations.
- Keep research abbreviations only when they are already part of the method
  vocabulary, and define them near the owner.

## 7. Module Boundaries And Reusable Abstractions

A module should own a concept, not a pile of convenience helpers.

Healthy module boundary signals:

- The module has a clear reason to exist.
- Its public names share a concept.
- It can be explained in one sentence.
- It reduces pressure on a large entrypoint or facade.
- It has a natural targeted verification surface.

Unhealthy module boundary signals:

- It is named after a vague implementation bucket.
- It mixes config parsing, metric emission, artifact IO, and model logic.
- It exists only because one function became long.
- It becomes the second place where a contract is defined.
- New agents would need to open it and the old owner to understand one concept.

Abstraction decision rule:

- First use: implement directly with clear names.
- Second use: consider extraction if the repeated logic is nontrivial.
- Third use: extract a shared abstraction because divergence is likely.
- Immediate extraction is justified for safety-critical contracts such as
  geometry, parser modes, metric denominators, sidecar keys, artifact names, and
  schema validation.

Avoid abstraction theater:

- Do not create registries before there are real extension points.
- Do not create inheritance hierarchies for one implementation.
- Do not hide objective-specific behavior behind generic layers that make
  debugging harder.
- Do not turn research branches into frameworks before their contracts stabilize.

## 8. Duplication Policy

Some duplication is cheaper than premature abstraction. Contract duplication is
dangerous.

Acceptable duplication:

- Short local code that is easier to read than abstract.
- Test setup that keeps cases independent.
- Transitional compatibility code with a clear deletion path.
- Similar orchestration steps for surfaces with different semantics.

Risky duplication:

- Repeated template IDs, parser modes, objective variants, metric keys, artifact
  names, or sidecar keys.
- Repeated geometry logic.
- Repeated config validation.
- Repeated denominator or reducer logic for metrics.
- Repeated fail-fast support policy in entrypoints and runtime modules.

Decision rule:

- Duplicate implementation only when semantics are local.
- Centralize vocabulary and safety contracts.
- If duplicated code must stay, add a comment naming why it is intentionally
  separate.

## 9. Metrics And Evaluation Semantics

Metrics are claims machinery. They must carry enough meaning to be interpreted
later.

Before adding or changing a metric, decide:

- Numerator.
- Denominator.
- Reducer.
- Unit.
- Semantic role.
- Token role, object scope, coordinate surface, parser mode, or metric surface
  when relevant.
- Whether it is canonical or diagnostic-only.
- Whether it needs a legacy alias.
- Whether it is comparable across runs.

Metric principles:

- Start from semantic identity, not flat string spelling.
- Publish flat keys only after identity and denominator are clear.
- Register legacy aliases deliberately.
- Do not manually duplicate alias values in multiple places.
- Avoid metric names that hide their denominator.
- Label partial, proxy, tiny, `val200`, `limit=200`, raw-text, coord-token, and
  full-val scopes explicitly.

Evaluation principles:

- Keep strict metrics separate from diagnostic salvage.
- Preserve parser mode and metric surface in any comparable result.
- Treat duplicate-control, confidence, COCO, LVIS, f1-ish, and proxy views as
  distinct surfaces unless a source owner explicitly unifies them.
- Do not compare results across parser modes or bbox surfaces without saying so.
- Artifact completeness matters more than logs that merely say a pipeline
  completed.

## 10. Training, Geometry, And Runtime Safety

Training changes must preserve geometry, labels, sidecars, and artifact
interpretability.

Training principles:

- Keep entrypoints as orchestrators, not policy owners.
- Put surface-specific policy in surface-owned runtime or trainer modules.
- Preserve geometry and image alignment end to end.
- Do not silently resize or reorder coordinates.
- Keep labels, sidecars, model inputs, and logging metadata distinct.
- Fail fast on unsupported packing, cache, sidecar, or parser combinations.
- Update manifests and docs when behavior becomes stable or operator-facing.

Runtime decision rule:

- If an invalid combination could produce plausible but wrong metrics, reject it
  before training or evaluation starts.
- If support is partial, name the missing implementation layer in the error.
- If a runtime path is temporary, call it debug, experimental, or compatibility.

## 11. Comments, Docstrings, And Extension Points

Good comments protect boundaries. They do not narrate obvious code.

Use comments and docstrings to explain:

- Which module owns a concept.
- Whether behavior is strict, diagnostic, compatibility, or transitional.
- Which other surfaces must change together.
- Why a fail-fast guard exists.
- Why a compatibility shim exists.
- What a future extension must preserve.

Avoid comments that:

- Repeat the code.
- Promise validation that does not exist.
- Hide uncertainty.
- Describe historical behavior as current behavior.
- Turn a temporary workaround into a permanent contract by accident.

Useful extension-point language:

- "Add new modes here only after schema, runtime policy, metrics, and docs are
  updated together."
- "This facade preserves imports; source behavior belongs in owner modules."
- "Diagnostic parsing is not a canonical metric surface."
- "This shim maps new schema to legacy runtime plumbing; do not treat it as the
  canonical config source."

## 12. Documentation Principles

Docs should route agents to truth. They should not become a second
implementation.

Good docs:

- Say what the current stable behavior is.
- Link exact source owners.
- Explain comparability boundaries.
- Distinguish hypothesis, plan, result, interpretation, and stable contract.
- Name artifact roots, configs, checkpoints, metric files, and scope when
  reporting results.
- Stay short enough that future agents will read them.

Risky docs:

- Duplicate long lists of volatile constants.
- Make benchmark claims without scope.
- Preserve old experimental interpretation as current guidance.
- Hide whether evidence is proxy, tiny, partial, or full validation.
- Copy implementation checklists that belong in super-power plans.

Decision rule:

- Put durable current behavior in docs.
- Put detailed execution checklists in super-power plans.
- Put coarse progress and blockers in `progress/` when needed.
- Put research interpretation and durable decisions in `docs/` when appropriate.
- Keep executable truth in the repo.

## 13. Future-Agent Workflow Checklist

Use this checklist before nontrivial edits.

Orient:

- Identify the surface: data, config, template, objective, runtime, trainer,
  inference, eval, metrics, artifacts, docs, or workflow records.
- Read the relevant routing docs before broad source search.
- Locate the owner module for the concept.
- Identify whether the change is strict, compatibility, diagnostic, or
  temporary.
- Check whether a facade or entrypoint is only adapting an owner module.

Design:

- State the concept being changed in one sentence.
- Choose the canonical vocabulary.
- Decide which aliases are allowed.
- Decide which container or schema owns the shape.
- Decide which invalid combinations must fail fast.
- Decide which metrics, artifacts, docs, or manifests are affected.
- Choose the smallest viable change.

Implement:

- Edit the owner module first.
- Keep facades thin.
- Keep entrypoints orchestration-focused.
- Preserve old imports unless the task explicitly breaks compatibility.
- Keep strict and diagnostic paths separate.
- Avoid broad abstractions until repeated use justifies them.

Verify when verification is in scope:

- Prefer the narrowest realistic test or artifact check.
- Verify config parsing before launching expensive training.
- Verify geometry, parser, metric, or artifact contracts before broad suites.
- Report exact scope and command shape.
- Do not claim full validation from partial evidence.

Document:

- Update docs when user-facing behavior, config schema, artifact names, metrics,
  or recommended workflows change.
- Link source owners instead of restating large code details.
- Record exact metric scope for experiment results.
- Keep management records short, link-rich, and non-duplicative.

## 14. Pause And Escalate Triggers

Future agents should pause for alignment when:

- A change would alter metric comparability.
- A change would alter geometry or object ordering.
- A change would add a new stable config knob.
- A change would move behavior between strict and diagnostic paths.
- A change would make an old artifact unreadable.
- A change would require broad entrypoint conditionals.
- A change would introduce a new abstraction layer with unclear second use.
- A change would compare partial evidence against full validation.
- A dirty worktree contains unrelated changes in files the agent needs to edit.

Escalation should be concrete:

- Name the trade-off.
- Offer one recommended path.
- Keep the user unblocked.
- Do not turn uncertainty into silent behavior.

## 15. Good Patterns To Preserve

Preserve these codebase-wide patterns:

- Source-owned contracts with thin compatibility adapters.
- Strict parsing and training behavior separated from generated-text salvage.
- Frozen typed containers for cross-module internal contracts.
- Config schemas that reject unknown or obsolete keys on stable surfaces.
- Runtime modules that own support policy and fail-fast checks.
- Metric identity and alias bridges instead of scattered flat-key duplication.
- Evaluation modules split by concern behind stable public imports.
- Docs that act as routing maps and source-owner indexes.
- Small logical commits and isolated worktree edits during large refactors.

## 16. Risky Patterns To Avoid

Avoid these codebase-wide risks:

- Adding implementation logic to compatibility facades.
- Adding "just one more conditional" to a large entrypoint when a surface owner
  should exist.
- Creating config knobs as escape hatches for unclear behavior.
- Reusing legacy names for new semantics.
- Copying constants across config, metrics, docs, and artifacts.
- Mixing strict metrics with diagnostic salvage.
- Passing loose dictionaries across many modules.
- Creating generic abstractions before contracts stabilize.
- Treating proxy, tiny, partial, or `val200` evidence as full validation.
- Updating management records without updating executable repo truth.

## 17. North Star

The best CoordExp engineering change makes the next research question easier to
ask and safer to answer.

Aim for code where a future agent can quickly determine:

- What concept is being changed.
- Which module owns it.
- Which config field selects it.
- Which metric or artifact records it.
- Which compatibility path, if any, adapts it.
- Which narrow verification surface proves it still works.
- Which evidence scope supports any claim made about it.

If a change improves model behavior but makes these questions harder to answer,
it is not finished yet.
