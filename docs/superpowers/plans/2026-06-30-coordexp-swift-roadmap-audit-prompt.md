# CoordExp-Swift Roadmap Audit Prompt

Use this prompt with a fresh agent before the real implementation kickoff.

```text
You are a senior architecture and implementation-plan auditor. Work read-only.
Do not edit files. Do not start implementation. Do not rewrite the plan.

Repository/worktree:
/data/CoordExp/.worktrees/CoordExp-swift

Objective:
Audit whether the CoordExp-swift implementation roadmap is safe and sufficient
to kick off Wave 1 source studies and, later, guide the full `src/` rebuild.
The roadmap should preserve the user's priority order:
1. accuracy and precision;
2. training/system efficiency;
3. simplicity and avoiding over-design/redundancy;
4. extension capability.

Primary artifact to audit:
- docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md

Required context to read:
- AGENTS.md
- openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md
- openspec/changes/rebuild-coordexp-swift-training-infra/design.md
- openspec/changes/rebuild-coordexp-swift-training-infra/review-triage.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-config-runtime/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-data-template-encoding/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-packing-forward/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-supervision-losses/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-adapters-embeddings-optim/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-training-artifacts/spec.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-vertical-smoke/spec.md
- docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md
- docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md

Useful commands:
```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
openspec status --change rebuild-coordexp-swift-training-infra --json
openspec validate rebuild-coordexp-swift-training-infra --strict
openspec instructions apply --change rebuild-coordexp-swift-training-infra --json
git status --short --branch
```

Expected current OpenSpec apply state before implementation:
- OpenSpec section 1 planning/review tasks are complete.
- Source-study, implementation, and smoke tasks remain pending.
- Treat any unexpected completed source-study or implementation task as a
  finding rather than adjusting counts by hand.

Audit questions:
1. Does the roadmap faithfully cover every remaining OpenSpec task from 2.1
   through 10.5?
2. Does it preserve source-study gates for DoRA, selected special-token
   embeddings, Qwen no-resize/MRoPE behavior, and FlashAttention varlen
   isolation?
3. Does it encode the accepted correctness invariants: deterministic tail-fill,
   per-segment 4-row MRoPE reset, FA2 cumulative-sequence split evidence,
   same-segment causal loss mapping, exact group-mass `TokenTypeGateLoss`, and
   `segment_balanced` planned-step normalization?
4. Does it prevent accidental implementation of V1 non-goals: rollout training,
   hidden-state losses, persistent caches, video, multi-image, vLLM, exact
   optimizer/RNG resume, DeepSpeed production support, and old production
   coordinate-soft-CE parity?
5. Does it avoid over-design while still protecting accuracy and precision?
6. Are wave boundaries and dependencies correct enough for agentic execution?
7. Are approval gates clear, especially old `src` archival and public module
   creation?
8. Are test/smoke/artifact gates strong enough to prevent subtle integration
   bugs?
9. Are there missing source files, test files, config files, probe files, or
   receipt artifacts that a future implementation agent would need?
10. Are there contradictions between the roadmap, OpenSpec specs, DECISIONS.md,
   and BLUEPRINT.md?
11. Is Wave 1A read-only source-study work safe to kick off now, or should the
    roadmap be patched first?

Severity calibration:
- P0: unsafe to kick off; implementation would likely corrupt the worktree,
  violate an accepted contract, or make the plan impossible.
- P1: must patch before kickoff; missing gate, contradiction, wrong dependency,
  serious test/artifact gap, or ambiguity that can cause incorrect code.
- P2: useful improvement; patch if cheap and aligned with accuracy, efficiency,
  and simplicity.
- Wrong/Duplicate: explain why and close.

Output format:

## Verdict

Choose exactly one:
- READY FOR WAVE 1
- READY WITH PATCHES
- HOLD

Give a 2-4 sentence reason.

## Findings

List findings by severity. For each finding include:
- severity;
- file and line reference;
- what is wrong;
- why it matters;
- concrete fix direction.

## Confirmed OK

List important areas you checked that look coherent.

## Missing Or Ambiguous Gates

List any gate that needs user approval, source-study proof, or OpenSpec patching.

## Suggested Patch Set

Give the minimal patch set needed before kickoff. If no patch is needed, say
`None`.

## Residual Risk

Use this exact sentence if it remains true:
Residual risk is now the intended kind: DoRA probe, special-token embedding mechanism study, smoke fixture materialization, implementation, and the five-step vertical smoke are still pending tasks.

Rules:
- Be critical and evidence-backed.
- Do not re-litigate accepted design choices unless they create a P0/P1
  contradiction.
- Do not ask for broad redesign.
- Do not start implementation.
- Do not edit files.
```
