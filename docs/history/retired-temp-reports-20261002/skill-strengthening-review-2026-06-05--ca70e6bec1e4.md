# CoordExp Skill Strengthening Review

Date: 2026-06-05
Scope: `.codex/skills/audit-review`, `.codex/skills/coordexp-codebase`, `.codex/skills/coordexp-research-context`, `.codex/skills/model-diagnosis`, `.codex/skills/model-innovation-risk-audit`, `.codex/skills/worktree-feature-loop`
Mode: read-only review of target skills plus generated proposal artifact. No target skill files were edited.

## Executive Summary

The six skills are already unusually strong on CoordExp-specific authority ordering, benchmark scope labels, artifact provenance, and user-preference preservation. The dominant improvement opportunity is not adding more generic expertise. It is making the skill ecosystem composable: clearer role boundaries, a common handoff packet, small task classifiers, and stronger stop/closure rules.

Recommended architecture:

- `worktree-feature-loop`: conductor for execution, isolation, verification gate, and cleanup delegation.
- `coordexp-codebase`: current-checkout router and minimal trace map.
- `coordexp-research-context`: historical/mechanism context builder and stale-evidence labeler.
- `model-innovation-risk-audit`: pre-launch trust gate for new mechanisms and silent train/eval/config/artifact mismatch.
- `model-diagnosis`: post-observation artifact-first symptom and root-cause debugger.
- `audit-review`: read-only severity-ranked correctness, reproducibility, governance, and eval-validity reviewer.

Highest-leverage quick wins:

1. Add a short "When To Use / When To Hand Off" selector to each skill.
2. Add a shared handoff packet template.
3. Add `model-diagnosis` artifact triage before metric interpretation.
4. Add `audit-review` audit-mode selector and review closure rule.
5. Add `coordexp-codebase` task classifier, minimal trace recipe, and verification matrix.
6. Add `coordexp-research-context` memory/session evidence protocol, stale-handle rule, and research stop rule.
7. Add `model-innovation-risk-audit` minimal contract diff and ConfigLoader/list-replacement footgun.
8. Add `worktree-feature-loop` preflight snapshot, reversible-change rule, and explicit delegation map.

## Evidence Base

Subagents:

- Audit & Governance Agent: `audit-review`, `model-innovation-risk-audit`.
- Research Coordination Agent: `coordexp-research-context`.
- Codebase Coordination Agent: `coordexp-codebase`.
- Model Evaluation Agent: `model-diagnosis`.
- Engineering Workflow Agent: `worktree-feature-loop`.
- Meta-Architecture Agent: cross-skill integration.

Local evidence handles:

- Current skill inventory: target `SKILL.md` files are 67-139 lines each; several have references, while `coordexp-codebase` and `coordexp-research-context` are mostly single-file playbooks.
- `MEMORY.md:1-35`: Stage-1 SoftCE/locality comparison preferences: hypothesis-neutral, slot-wise `x1/y1/x2/y2`, teacher-forced plus self-prefix, plot/table decision support.
- `MEMORY.md:37-97`: docs-first architecture audit, review-gated Stage-2 fix, selective staging, subagent fallback.
- `MEMORY.md:168-201`: skill audit hygiene: concise nontrivial skills, progressive disclosure, one-skill-at-a-time validation, YAML metadata validation.
- `MEMORY.md:497-553`: Stage-2 invalid prediction diagnosis: exact artifact root first, symptom taxonomy, sampled train rollout versus greedy eval split, aggregate `prepare_failures` before examples.
- `MEMORY.md:580-622`: Stage-2 config/loss/artifact contracts: `ConfigLoader.load_yaml_with_extends()` deep-merges dicts but replaces lists, docs/spec sync, eval artifact materialization.
- `rollout_summaries/2026-05-31T09-51-03-1bZN-coordexp_self_distillation_workflow_packaging.md:15-34`: shortlist-first packaging, subagent evidence lanes, avoid duplicate assets.
- `rollout_summaries/2026-06-04T08-53-40-QFQb-coordexp_ckpt3668_free_greedy_comparison_and_score_probe.md:17-36`: infer/eval helper traps, free rollout versus strict materialization, malformed row policy.
- `rollout_summaries/2026-06-04T08-53-40-QFQb-coordexp_ckpt3668_free_greedy_comparison_and_score_probe.md:63-100`: recall/leakiness and duplicate-burst diagnostics beyond AP.
- `rollout_summaries/2026-06-04T08-53-40-QFQb-coordexp_ckpt3668_free_greedy_comparison_and_score_probe.md:120-151`: scoring coverage probe, `score_desc` null no-op, provenance sidecars.
- `rollout_summaries/2026-06-01T08-31-00-arLC-coordexp_stage1_autoregressive_rollout_mechanism_and_coord_s.md:17-33`: mechanism exploration and probe correction.
- `rollout_summaries/2026-06-01T08-31-00-arLC-coordexp_stage1_autoregressive_rollout_mechanism_and_coord_s.md:74-87`: stop broad exploration once mechanism converges and propose algorithm direction.
- `rollout_summaries/2026-05-29T10-49-26-3AKi-anchor_grid_research_archived_worktree_cleanup.md:17-28`: research-first design and deprecate noncompetitive direction.
- `rollout_summaries/2026-05-29T10-49-26-3AKi-anchor_grid_research_archived_worktree_cleanup.md:50-58`: archive failed direction under `progress/` before worktree cleanup.

Caveat: several older `MEMORY.md` rollout-summary pointers are stale or absent from the current `rollout_summaries/` directory. Treat those registry entries as historical lower-confidence evidence unless the referenced summary is present.

## Current Capability Assessment

### `audit-review`

Strengths:

- Strong docs-first authority model.
- Findings-first severity format with evidence, impact, fix direction, and verification.
- Good read-only guardrails: no invented results, no benchmark scope mixing, no `progress/` as current truth, no Git by reflex.
- Useful audit-specific references for pipeline and report skeletons.

Weaknesses:

- Missing audit-mode selector: code/spec audit, artifact/run audit, launch gate, claim validity audit, implementation-vs-contract audit.
- OpenSpec governance lifecycle is only implicit.
- Review closure semantics are missing: timeout/disconnected reviewer is unresolved, not approval.
- Claim auditing is under-specified: exact claim, evidence scope, comparator, counterevidence, falsification.

Blind spots and failure modes:

- Overexploration after blocker-only review.
- Treating eval traces/artifact dumps as implementation details rather than stable eval-validity contracts.
- Missing docs/spec updates when schema, artifact names, metrics, or loss semantics move.

### `model-innovation-risk-audit`

Strengths:

- Correctly framed as pre-launch contract and provenance gate, not symptom debugger.
- Strong triangulation list across design intent, config, data, tokenizer, collator, loss, decode/eval, artifacts.
- Good risk taxonomy reference, especially sidecar/logit alignment, loss composition, metrics raw-vs-effective, and train/decode/eval parity.
- Stage-2 production gate already returns bounded verdicts.

Weaknesses:

- Needs a required minimal contract diff: intended, authored config, resolved config, runtime, artifacts, verdict.
- Key ConfigLoader/list-replacement and loss-footgun heuristics live in memory or references, not entrypoint path.
- Stage-2 verdict shape should generalize to other high-risk innovations.

Blind spots and failure modes:

- Smoke passes while runtime silently falls back to legacy behavior.
- Objective list patch silently drops inherited objectives.
- Metrics look normal while differentiable scalar differs from intended formula.

### `coordexp-codebase`

Strengths:

- Strong compact routing map and authority spine.
- Good high-signal source map across data, training, Stage-1, Stage-2, infer/eval, bootstrap, common.
- Encodes non-obvious guardrails: config-first, `do_resize=false`, bbox math via geometry, Stage-2 pipeline namespace split.

Weaknesses:

- More static map than operational onboarding workflow.
- Missing task classifier and "when to delegate" map.
- Verification guidance is too generic for repeated CoordExp patterns.
- Serena guidance is thin and should point to `serena-mcp-navigation`.

Blind spots and failure modes:

- Agent reads `progress/` as current behavior.
- Agent over-greps or scans all code before docs routing.
- Agent collapses artifact zones: `outputs`, `output_remote`, `public_data`, and provenance manifests.
- Agent stages local `.codex` or `.serena` state during code/config commits.

### `coordexp-research-context`

Strengths:

- Correct current-vs-historical split.
- Good context pack contract: current answer, handles, evidence, labels, risks, search seeds.
- Strong benchmark scope discipline and high-signal code handles.

Weaknesses:

- Missing memory/session-mining loop and stale-handle rule.
- Missing research stop rule: enough for algorithm design, one more probe, do not interpret yet, archive/pause.
- Benchmark interpretation is too generic for AP/AR/FN/duplicate/scoring/provenance needs.
- Negative-result archival before cleanup is not explicit.

Blind spots and failure modes:

- Historical memory from older checkout presented as current fact.
- AP-only conclusion misses recall leakage, duplicate burst, parser invalidity, or score no-op.
- Continues broad mechanism analysis after a small objective smoke is the correct next step.

### `model-diagnosis`

Strengths:

- Clear role boundary: symptom-to-root-cause debugger, not pre-launch audit.
- Good causal symptom classification: train/eval, teacher-forced/free-rollout, validity collapse, precision/recall shifts, provenance impossibility.
- Strong Stage-1 coordinate-locality recipe.

Weaknesses:

- Under-specified artifact-first execution.
- Missing standard symptom packet for AP/AR/FN, validity, duplication, score provenance, train/eval path separation.
- Missing failure-family aggregation for invalid outputs before sampling examples.
- Missing explicit decision vocabulary: artifact invalid, implementation bug, objective mismatch, decoding/parser mismatch, data issue, optimization issue, model limitation, inconclusive.

Blind spots and failure modes:

- Mistakes parser/materialization policy for model behavior.
- Treats a few invalid examples as representative without counting families.
- Claims `desc+bbox` scoring effects without checking `score_desc` coverage.

### `worktree-feature-loop`

Strengths:

- Correct default toward isolation for research, dirty trees, multi-file work, long runs, parallel agents, branch review.
- Good delegation to `using-git-worktrees`.
- CoordExp-specific gotchas around ignored roots, image roots, nested `mcp/codexUI`, and durable findings.
- Useful final status block.

Weaknesses:

- Missing dirty-state, branch, worktree, upstream preflight.
- "Small inplace edit" is not defined.
- Verification, commit grouping, sync semantics, and cleanup gates are too compact.
- Needs explicit conductor/delegation map.

Blind spots and failure modes:

- Under-isolating schema/artifact/metric changes.
- Deferring commit grouping until after large mixed changes accumulate.
- Removing worktree before archiving research evidence.
- Retrying push after interruption without explicit request.

## Benchmark Analysis

Against strong software engineering workflows, the skills are above average on evidence handles, narrow verification, docs-first routing, and dirty-worktree safety. They are weaker on lifecycle gates: intake classification, review closure, preflight snapshots, commit grouping, and "what proves done".

Against top-tier AI research workflows, the skills are strong on scope labels, artifact provenance, and hypothesis-neutral comparisons. Missing pieces are falsifiable evidence ledgers, baseline/ablation discipline, benchmark decomposition beyond aggregate AP, negative-result archiving, confidence/scoring coverage, and mechanism-to-intervention traceability.

Against scalable AI-agent operations, the skills are close but need a shared protocol. Without a common handoff packet, each skill can become a bespoke mini-world. The scalable design is compact entrypoints plus conditional references.

## Cross-Skill Operating Framework

Use this flow for CoordExp AI research and engineering:

1. Intake and isolation: `worktree-feature-loop` classifies request and chooses `worktree` or `inplace`.
2. Current truth: `coordexp-codebase` resolves docs/spec/config/code/artifact handles in the active checkout.
3. Historical meaning: `coordexp-research-context` adds memory/progress/rollout evidence only when lineage matters and labels stale or cross-checkout facts.
4. Pre-launch trust: `model-innovation-risk-audit` checks new objective/data/tokenizer/decode/eval/runtime changes with a minimal contract diff.
5. Post-launch diagnosis: `model-diagnosis` starts from exact artifact roots and produces symptom taxonomy plus likely root cause.
6. Independent review: `audit-review` returns severity-ranked correctness/reproducibility/eval-validity findings.
7. Finish: `worktree-feature-loop` delegates git/sync/cleanup and reports verification, artifacts, skipped checks, residual risks, and remaining dirty state.

Common handoff packet:

```text
intent:
scope: cwd, branch/worktree, checkpoint, dataset slice, config, artifact root, metric scope
current_handles: docs/specs/code/config keys proving current behavior
historical_handles: progress/memory/rollout evidence, labeled historical/stale when needed
artifact_handles: summaries, manifests, raw/scored JSONL, parser/drop counters, token traces, duplicate reports
risk_flags: schema drift, config fallback, eval validity, provenance mismatch, dirty state, missing baseline
decision_needed: implement / audit / diagnose / launch gate / rerun / archive / ask user
verification_path: exact targeted test, smoke, artifact check, or skipped reason
```

## Roadmap

### Quick Wins (<= 1 Day)

- Add small "When To Use / Hand Off" selectors to all six skills.
- Add a shared `references/handoff-packets.md` or skill-local equivalent.
- Add `model-diagnosis` artifact triage ladder.
- Add `audit-review` audit-mode selector and blocker-only stop rule.
- Add `coordexp-codebase` first-route classifier, minimal trace recipe, and verification matrix.
- Add `coordexp-research-context` memory/session evidence rule and stop rule.
- Add `model-innovation-risk-audit` minimal contract diff and ConfigLoader/list replacement warning.
- Add `worktree-feature-loop` preflight snapshot and reversible-change definition.

### Medium-Term Upgrades (<= 2 Weeks)

- Create shared references:
  - `references/handoff-packets.md`
  - `references/artifact-triage.md`
  - `references/contract-diff.md`
  - `references/benchmark-interpretation.md`
  - `references/verification-matrix.md`
- Move long file lists and metric breakdowns out of `SKILL.md` files.
- Add example workflows:
  - new Stage-2 objective launch gate;
  - checkpoint infer/eval metric drop diagnosis;
  - dirty worktree feature through audit, commit, sync, cleanup.
- Add review-closure rule to audit/risk skills.
- Add keep/drop/rerun/hold decision vocabulary to diagnosis and research context.
- Add a lightweight skill ecosystem index: owner, adjacent skills, references, anti-overlap warnings.

### Long-Term Investments

- Build small regression examples for skill routing: prompt -> expected skill set -> output skeleton.
- Add optional validator checks for reference links and YAML frontmatter across `.codex/skills`.
- Consider custom subagent templates only after repeated use proves stable.
- Keep memory updates explicit-only, preserving "no hidden agent memory stores".

## Proposed Revised Skill Drafts

These are proposed drafts, not applied edits. They intentionally keep entrypoints compact and point to references for long checklists.

### Proposed `.codex/skills/audit-review/SKILL.md`

```markdown
---
name: audit-review
description: "Use when producing a read-only CoordExp audit of code, configs, specs, artifacts, docs, progress notes, or OpenSpec changes for correctness, reproducibility, governance, and eval-validity risks."
---

# Audit Review

Produce read-only findings that help another implementer change CoordExp safely. Optimize for correctness, reproducibility, governance, pipeline integrity, and eval validity over style commentary.

## Role Boundary

Use this for severity-ranked audits. Do not implement fixes. If the user gives an abnormal model symptom, start with `model-diagnosis`; return here only for independent correctness or claim-validity review. If the user asks whether a new mechanism is contract-safe before launch, use `model-innovation-risk-audit`.

## Audit Mode Selector

Name the mode before searching broadly:

- `change/spec audit`: compare implementation, docs, stable specs, and active OpenSpec deltas.
- `artifact/run audit`: start from the exact artifact root; label scope and do not generalize beyond it.
- `launch gate`: decide `promote`, `hold`, `rerun gate`, or `needs user decision`.
- `claim validity audit`: identify claim, scope, baseline/ablation, metrics, counterevidence, and falsification gap.
- `implementation-vs-contract audit`: verify code/config/runtime/artifacts implement documented behavior.

If the user requests blocker-only review, stop after blocking findings, confirmed OK, and residual risks.

## Authority Model

Use current repo truth in this order:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when explicitly in scope
7. `progress/` only for history, diagnostics, benchmark evidence, or empirical failures

Use `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for routing. Treat `progress/audits/` and temporary notes as removable evidence, not durable behavior references.

## Governance Checks

For stable contracts or OpenSpec work:

- Check active-change state when it matters.
- Validate specs strictly when specs changed.
- Verify code/docs/spec sync when schema, artifact names, metrics, loss semantics, entrypoints, or recommended workflows move.
- Treat incomplete proposals as active or explicitly deprecated; do not let stale changes masquerade as current contract.
- Reviewer timeout or disconnection is unresolved, not approval.

## Output Contract

Lead with findings, ordered by severity:

- `P0`: likely invalidates correctness, reproducibility, or evaluation claims.
- `P1`: substantial risk to supported workflows, artifacts, metrics, or config contracts.
- `P2`: maintainability, clarity, or missing-coverage risks that could become failures.

Each finding needs evidence handle, impact, fix direction, and minimal verification.

Also include confirmed OK checks, open questions only when blocking, suggested next actions, and residual risks.

## Read-Only Guardrails

- Do not modify code, configs, docs, specs, or artifacts.
- Do not invent results; label hypotheses.
- Do not compare benchmark scopes without labels.
- Do not use `progress/` as current behavior when docs/specs cover the contract.
- Use Git inspection only when dirty state, PR/change diff, or user request makes it relevant.
- For Python, narrow with `rg` or `rtk grep`, then use Serena symbols.

## References

Open only when helpful:

- `references/report-template.md`: report skeleton.
- `references/grep-seeds.md`: high-signal search seeds.
- `references/pipeline-checklist.md`: end-to-end correctness and eval-validity checklist.
```

### Proposed `.codex/skills/model-innovation-risk-audit/SKILL.md`

```markdown
---
name: model-innovation-risk-audit
description: "Use when reviewing a planned or newly wired CoordExp model/objective/data/tokenizer/decode/eval/runtime change before trusting training or eval claims, especially for silent train/eval/config/artifact mismatch risk."
---

# Model Innovation Risk Audit

Default to read-only audit. This is a contract and provenance gate, not a symptom debugger.

## Role Boundary

Use when asking:

- Can we trust this new mechanism, config, data path, tokenizer/template, loss, decode path, eval path, or runtime integration?
- Could this silently train or evaluate a different contract than intended?
- Before launch or interpretation, are schema, materialized config, data, loss, decode/eval, metrics, and artifacts aligned?

If concrete behavior is already abnormal, start with `model-diagnosis`; return here only if diagnosis suggests silent config/runtime/eval drift.

## Minimal Contract Diff

For every innovation, produce:

- Intended contract: design/spec/plan.
- Authored config: YAML keys and inheritance chain.
- Resolved contract: materialized config and schema dataclass.
- Runtime contract: dataset/collator/trainer/loss/decode/eval objects actually used.
- Artifact contract: resolved config, manifests, parser/drop counters, metric keys.
- Evidence verdict: matched / mismatched / unproven.

Do not trust a smoke run if any row silently falls back to legacy behavior.

## Contract Triangulation

Compare intent across design, config schema, materialized config, data JSONL, geometry, ordering, image roots, tokenizer/template/stops, dataset/collator labels and masks, model forward/logits/slicing, loss weights and precision, metrics, decode/eval/parser behavior, and artifacts.

## Config And Loss Footguns

- Dicts may deep-merge while objective lists may replace wholesale; inspect final resolved lists.
- Reject legacy keys in semantic modes instead of warning or ignoring.
- Separate monitoring-only knobs from differentiable objective weights.
- Verify raw loss terms and effective weighted contributions are both logged.
- Check zero-weight targets, EOS/type-gate composition, duplicate multiplicity, and teacher-token membership.
- Use deterministic tiny-logit tests for scalar formulas before trusting training curves.

## Probes

Prefer narrow evidence: real tokenizer ids, resolved config, one encoded sample, one collated batch, tiny logits loss calculation, JSONL/image scan, artifact manifest check, targeted unit test, or smoke command.

Probe goal: prove or falsify a contract mismatch. Do not explain a metric regression from aggregate scores alone.

## Findings-First Report

Use:

```text
Findings
Minimal Contract Diff
Confirmed OK
Decision Questions
Patch Recommendations
Unit Tests And Diagnostics
Smoke Run Suggestions
Residual Risks
```

Finding severity: `P0`, `P1`, `P2`, `P3`. Do not inflate severity because a finding is interesting.

## Parallel Audit Split

When independent surfaces exist and subagents are available, split by objective/loss, dataset/collator, tokenizer/template/decode, config/runtime, artifact/eval, and tests/diagnostics.

## Launch/Promote Verdict

For high-risk model innovations, return one of: `promote`, `hold`, `rerun gate`, or `needs user decision`, plus the smallest verification that would change the verdict.

## References

Load only when needed:

- `references/risk-taxonomy.md`
- `references/subagent-prompts.md`
- `references/report-template.md`
```

### Proposed `.codex/skills/coordexp-codebase/SKILL.md`

```markdown
---
name: coordexp-codebase
description: "Use when navigating the CoordExp research codebase, locating current docs/specs/code entrypoints, or changing data, training, Stage-1, Stage-2, inference, evaluation, artifact, or provenance behavior."
---

# CoordExp Codebase Navigation

Use this as the current-checkout router. Keep it pointer-first: repo docs for durable truth, then the smallest code/config/artifact surface.

## First Route The Task

- Current behavior or entrypoint lookup: use this skill, then docs route.
- Historical rationale, benchmark provenance, or current-vs-old comparison: pair with `coordexp-research-context`.
- Infer/eval launch, repair, or metric-bearing artifact work: hand off to `coordexp-infer-eval-workflow`.
- `public_data` manifests/checksums/regeneration/cleanup: hand off to `coordexp-public-data-provenance`.
- Python symbol/call graph tracing: narrow here, then use `serena-mcp-navigation`.
- New objective/runtime/eval/data trust gate: use `model-innovation-risk-audit`.
- Abnormal model behavior: use `model-diagnosis`.
- Commit/sync/isolation: use `worktree-feature-loop` and git skills.

## Authority Model

Use current repo truth in this order:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_OVERVIEW.md`
3. `docs/IMPLEMENTATION_MAP.md`
4. relevant domain docs under `docs/`
5. `openspec/specs/` only for stable compatibility-sensitive contracts
6. `openspec/changes/<active-change>/` only when explicitly scoped
7. `progress/` only for history, diagnostics, benchmark evidence, or design derivation

Use `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, and `docs/ARTIFACTS.md` for routing.

## Minimal Trace Recipe

1. Name the exact user surface: config, script, artifact root, metric, or symbol.
2. Open the docs/catalog route for that surface.
3. Resolve config/schema ownership before code changes.
4. Trace one path end to end: JSONL/image -> config -> loader -> dataset/collator -> trainer/infer/eval -> artifact/metric.
5. For Python, use `rg`/`rtk grep` to narrow, then Serena symbols/references.
6. Stop when the smallest code/config/artifact surface answers the question.

## Verification Matrix

- Python implementation: `python -m py_compile <touched.py>` plus targeted `python -m pytest <test>`.
- YAML/config leaf: parse/resolve YAML and run smallest config/schema test.
- OpenSpec/stable contract: strict OpenSpec validation for changed specs/active changes.
- Infer/eval artifact: check `resolved_config.json`, `summary.json`, `metrics.json`, raw/scored artifact names, coordinate surface, and scope label.
- `public_data`: run the provenance-manifest test.
- Commit hygiene: inspect dirty state, stage only intentional paths, run `git diff --cached --check`.

## High-Signal Source Map

Keep the existing source map and task routing matrix from the current skill:

- `src/sft.py`, `src/config/`, `src/training_runtime/`
- `src/datasets/`, `src/detection/`, `src/data_collators/`
- `src/trainers/`, `src/infer/`, `src/eval/`
- `src/bootstrap/`, `src/common/`, `src/analysis/`

Preserve current task routing for data/geometry, Stage-1 baseline, Stage-1 set-continuation, Stage-1 compact recursive detection, Stage-2 two-channel, Stage-2 rollout-aligned, infer/eval, artifacts, manifests, and provenance.

## Edge Cases

- `stage2_ab.pipeline` and `rollout_matching.pipeline` are not interchangeable.
- Sorted Channel-B insertion can move FN objects into the prefix; token types and desc CE weights must stay aligned.
- Training geometry assumes `do_resize=false`; route bbox math through `src/datasets/geometry.py` unless editing detection serialization.
- `output_remote`/`outputs` are artifact zones; avoid whole-tree mirroring by default.
- `public_data` recovery is manifest/checksum/regeneration work, not routine Baidu sync.
- Local `.codex`/`.serena` memory/config files are usually not runtime feature files.

## Guardrails

Config-first. Offline-prepared single-dataset JSONL is the default training surface. Preserve image/geometry alignment. Do not edit upstream HF model files. Do not invent benchmark results or compare scopes without labels.
```

### Proposed `.codex/skills/coordexp-research-context/SKILL.md`

```markdown
---
name: coordexp-research-context
description: "Use when CoordExp work needs broad research context, design history, empirical evidence, benchmark provenance, diagnostics, or a current-vs-historical read before implementation or audit."
---

# CoordExp Research Context

Build a compact evidence pack that separates current contract, historical evidence, mechanism inference, and next research action.

## Authority And Scope

Current behavior comes from docs/specs/code in the active checkout. Historical evidence comes from `progress/`, artifacts, memories, and rollout summaries.

Always label cwd, branch/checkpoint if known, dataset slice, bbox format, decode surface, metric scope, and whether evidence is current, historical, partial, stale, or repaired.

## Evidence Mining Loop

1. Start from docs route for current behavior.
2. Search `.codex/memories/MEMORY.md` by task-specific keywords.
3. Open at most 1-3 relevant rollout summaries or memory/skill files.
4. If a memory handle is missing or stale, say so, use the registry entry as lower-confidence evidence, and search available summaries by date/keyword for a substitute.
5. Prefer artifact summaries/manifests over logs.

## Context Pack Output

```text
Question
Current Contract
Historical Evidence
Mechanism Read
Benchmark Or Probe Evidence
Counterevidence And Caveats
Decision: enough / one more probe / do not interpret yet / archive or pause / hand off
Search Seeds And Exact Handles
```

## Research Stop Rules

Stop broad exploration when competing explanations are narrowed to one dominant mechanism, remaining uncertainty maps to a small probe or objective smoke, or a direction is noncompetitive enough to archive with reopen criteria.

Continue only when artifact validity is unresolved, baseline/scope mismatch blocks interpretation, symptom taxonomy is not counted, or the proposed mechanism lacks discriminating evidence.

## Benchmark Interpretation Checklist

For score comparisons, checkpoint selection, or keep/drop decisions, report:

- checkpoint, dataset slice, limit/full scope, bbox format, prompt/template, decode surface, repetition penalty, temperature, GPU/shard shape;
- raw and guarded metrics separately;
- AP plus recall/FN/F1-ish when leakiness matters;
- parse/drop/materialization validity;
- duplicate burst totals and per-image tails;
- confidence/scoring provenance, including whether desc scores are non-null before claiming geom+desc fusion effects;
- whether a result is final, partial, repaired, or temporary probe evidence.

## Paused Or Failed Directions

When recommending pause/drop, preserve best baseline and candidate metrics with scope labels, implemented surfaces/artifacts, failure diagnosis, reopen criteria, and cleanup boundary before worktree cleanup when edits are in scope.

## Integration

- Hand off to `coordexp-codebase` for current code entrypoints or implementation paths.
- Hand off to `coordexp-infer-eval-workflow` for launch, repair, scoring, eval, Oracle-K, or proxy bundles.
- Hand off to `model-diagnosis` for abnormal metrics, invalid outputs, duplication, recall shifts, or train/eval divergence.
- Hand off to `model-innovation-risk-audit` before trusting new mechanisms.
- Hand off to `audit-review` for findings-first correctness/eval-validity audits.

## Existing Maps

Retain the current progress routing, high-signal code handles, benchmark/evidence rules, and `references/grep-seeds.md` pointer from the existing skill.
```

### Proposed `.codex/skills/model-diagnosis/SKILL.md`

```markdown
---
name: model-diagnosis
description: "Use when CoordExp model behavior is already abnormal or uncertain: metric drops, FP/FN shifts, invalid or malformed outputs, duplication bursts, length/stop/repetition changes, train/eval divergence, optimization instability, or launch-health symptoms."
---

# Model Diagnosis

Stay causal. This is a symptom-to-root-cause debugger, not a pre-launch contract audit.

## Role Boundary

Use when asking why metric, rollout behavior, loss, parse rate, prediction count, or qualitative output changed. Use `model-innovation-risk-audit` first for planned/new mechanisms before results exist.

Switch to `model-innovation-risk-audit` when evidence suggests silent train/eval/config/runtime mismatch, stale artifacts, wrong adapter, template drift, schema drift, metric ambiguity, or score knob no-op.

## 0. Artifact Triage First

Start from the exact artifact root named by the user.

- Identify run root, checkpoint, dataset slice, decode surface, template, bbox format, scope.
- Open durable summaries before logs: `resolved_config.json`, `summary.json`, `metrics*.json`, `run_metadata.json`, `pipeline_manifest.json`.
- For infer/eval, check raw/scored prediction JSONL, token traces, confidence summaries, duplicate guard reports, parser/drop counters, and provenance sidecars.
- For invalid outputs, aggregate `monitor_dumps/prepare_failures` before sampling examples.
- Decide whether malformed rows should abort, be guarded, or be post-processed according to benchmark contract and user intent.

## Symptom Delta Packet

Collect:

- main metrics: AP/AP50/AP75/AR, raw and guarded;
- FP/FN: predicted count, FN count, recall/F1-ish when relevant;
- validity: parse/drop/truncation/empty/malformed rates;
- repetition: duplicate suppression, p95/p99/max predictions per image, repeated class tails;
- train/eval path: sampled training rollout, greedy eval rollout, teacher-forced loss/logits, free rollout;
- scoring: score source, `score_desc` coverage, bbox-vs-desc fusion truth.

## Classify Before Explaining

Keep the current symptom taxonomy: train improves/eval drops, teacher-forced improves/free rollout worsens, validity collapse, recall down, precision down, both down, crowded-only failure, impossible provenance.

For invalid predictions, count failure families before examples: wrong arity, missing fields, unexpected keys, invalid coordinate slot, runaway punctuation/bracket tails, truncation, empty objects, max-token saturation.

## Mechanism Probes

Use tiny probes before expensive training when objective math, targets, tokenizer/template, sampling, packing, precision, optimizer groups, decoding, or eval semantics are implicated.

For coordinate objective decisions, preserve `x1/y1/x2/y2`, teacher-forced vs self-prefix, plots plus tables, variant identity, and scope labels.

## Decision Vocabulary

Return one of:

- `artifact invalid`
- `implementation bug likely`
- `objective mismatch`
- `decoding/parser mismatch`
- `data distribution issue`
- `optimization issue`
- `model limitation`
- `inconclusive-needs-probe`
- `keep`, `drop`, `hold pending paired baseline`, or `rerun due artifact/provenance mismatch`

## Output

```text
Diagnosis
Evidence
Probe Scope
Symptom Taxonomy
Likely Root Cause
Corrective Strategies
Verification
Confidence
```

Never relax parsers to hide malformed outputs unless the user explicitly changes the benchmark contract.
```

### Proposed `.codex/skills/worktree-feature-loop/SKILL.md`

```markdown
---
name: worktree-feature-loop
description: Use when starting CoordExp feature, fix, or research work where a dirty tree, parallel work, experiments, or long-running artifacts make isolation useful.
---

# Worktree Feature Loop

Orchestrate CoordExp feature/fix/research delivery from intake through isolation, implementation, verification, git hygiene, sync, cleanup, and handoff.

## 0. Classify Request

- `read_only_audit`: do not edit; use `audit-review`.
- `implementation`: isolate if dirty, multi-file, research, long-running, schema/artifact/metric-bearing, or review-bound.
- `commit_only`: do not change behavior; use git hygiene/sync loop.
- `merge_sync_cleanup`: verify branch/worktree state; use finishing/git skills.
- `experiment_or_smoke`: use worktree or absolute shared roots; use `full-pipeline-smoke` when train/infer/eval/artifacts must be proven.

## 1. Preflight Snapshot

Before edits or cleanup, inspect:

- root and branch;
- linked worktree versus main checkout;
- `git status --short --branch`;
- `git worktree list`;
- upstream/ahead/behind when sync or publication is in scope;
- nested repo guard for `mcp/codexUI`;
- data/model/output root availability for smoke or long runs.

Stop or ask on ambiguous scope, tracked secrets, destructive cleanup, publication not requested, high-cost training/smoke, or dirty files overlapping intended edits with unclear ownership.

## 2. Isolation Decision

Use a worktree for research, multi-file or multi-commit work, current-tree dirty state, long-running artifacts, parallel agents, schema/artifact/metric/default changes, and branch review.

Use inplace only for narrow edits with explicit verification, no stable contract changes, no long-running outputs, and no unrelated dirt in the same files.

CoordExp default worktree root: `.worktrees/`. Branch prefix: `codex/`.

## 3. Plan Surface

- `existing`: continue existing OpenSpec or super-power artifact.
- `new`: create only for stable compatibility-sensitive or multi-step work that needs it.
- `none`: implement directly and record acceptance checks in final/PR text.

## 4. Implementation Loop

Use `coordexp-codebase` for docs/code routing. Keep changes config-first, reversible, and scoped. Update docs/specs when stable defaults, schemas, entrypoints, artifacts, metrics, loss semantics, or workflows move.

## 5. Verification Gate

Name expected proof before checks. Use targeted tests first. Use `full-pipeline-smoke` for train/infer/eval/artifact changes. Report skipped checks with reason and residual risk.

## 6. Git/Sync Gate

For commits, delegate to git hygiene: group by intent, stage explicit paths, inspect cached diff, run staged-scope verification, commit, then repeat.

For sync, fetch and inspect ahead/behind before push. Push only when requested or clearly in scope.

## 7. Finish/Cleanup

Use finishing/development-branch workflow for merge/PR/keep/discard. Remove worktrees only after merge/discard, from the main root, with provenance check and no uncommitted work. Promote durable outputs before cleanup.

## CoordExp Gotchas

- Ignored data/model roots may be missing in worktrees; prefer symlinks or absolute shared roots over config churn.
- Preserve canonical image roots for infer/eval fanout.
- Never stage nested `mcp/codexUI` changes into the parent repo.
- Dirty files in other worktrees are expected and out of scope.
- If research produced durable findings, promote them to `progress/`, docs, or requested artifact locations before cleanup.

## Final Status Block

```text
task_slug:
request_class:
execution_mode:
worktree_path:
branch_name:
base_branch:
spec_mode:
changed_files_or_commits:
validation_ran:
validation_skipped:
artifact_roots:
sync_state:
merge_state:
cleanup_state:
remaining_dirty:
residual_risks:
```
```

## Shared Reference Architecture

Recommended new shared references:

- `references/handoff-packets.md`: common packet shape and skill-specific handoffs.
- `references/artifact-triage.md`: model/infer/eval artifact ladder and file roles.
- `references/contract-diff.md`: intended/authored/resolved/runtime/artifact diff shape plus config/loss footguns.
- `references/benchmark-interpretation.md`: AP/AR/FN, raw/guarded, parser/drop, duplicate, scoring coverage, scope labels.
- `references/verification-matrix.md`: targeted checks by Python/YAML/OpenSpec/infer/eval/public_data/git surfaces.

Avoid copying:

- full infer/eval launch recipes into `audit-review` or `model-diagnosis`;
- git/PAT mechanics into `worktree-feature-loop`;
- all repo maps into `coordexp-research-context`;
- OpenSpec author/archive mechanics into audit skills;
- Serena mechanics into `coordexp-codebase`.

## Validation Plan For Actual Skill Updates

When applying these revisions, run:

```bash
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/audit-review
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/coordexp-codebase
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/coordexp-research-context
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/model-diagnosis
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/model-innovation-risk-audit
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/worktree-feature-loop
python - <<'PY'
import pathlib, yaml
for path in pathlib.Path(".codex/skills").glob("*/agents/openai.yaml"):
    yaml.safe_load(path.read_text())
PY
git diff --check
```

## Deliberate Skips

- No actual target skill files were edited in this pass.
- No new custom subagent templates were created. This review used subagents, but repeated use should be proven before packaging them.
- No memory updates were written because the user did not explicitly request a memory update.
- No broad session-log mining was performed beyond memory registry and selected rollout summaries; this kept the review evidence-backed without turning it into an unbounded archaeology pass.
