# CoordExp-Swift Long-Term Execution Charter

## Purpose

This charter governs the long-running rebuild of the CoordExp-swift training
infrastructure in `/data/CoordExp/.worktrees/CoordExp-swift`. It records the
user-approved execution permissions, stop gates, GPU policy, MCP/tool policy,
and completion criteria for the active goal.

The implementation roadmap remains
`docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md`.
The normative implementation contracts remain the OpenSpec change
`rebuild-coordexp-swift-training-infra`, `DECISIONS.md`, and `BLUEPRINT.md`.
As of 2026-07-02, the active phase is production relaunch readiness for the
completed training rebuild. The current follow-on roadmap is
`docs/superpowers/plans/2026-07-02-coordexp-swift-production-relaunch-roadmap.md`,
and the follow-on OpenSpec change is `prepare-coordexp-swift-production-relaunch`.

## Goal

Rebuild `src/` into a functional, review-gated, smoke-verified CoordExp-swift
V1 training infrastructure by executing the approved OpenSpec and Superpowers
roadmap wave by wave.

The goal is complete only when the new codebase can run the approved Qwen3-VL
DoRA supervised training smoke with real artifacts, metrics, eval.forward
outputs, checkpoint metadata, and `checkpoints/checkpoint-final.json`, and all
required review gates report no unresolved P0/P1 findings.

## Autonomy Granted

Codex may autonomously:

- execute the approved roadmap wave by wave;
- use `superpowers:subagent-driven-development` for implementation slices,
  source studies, probes, spec reviews, code-quality reviews, and artifact
  audits;
- use MCP tools, including CodeGraph and Serena, when they
  help inspect legacy code, upstream behavior, symbol relationships, or impact;
- patch docs, OpenSpec specs, tasks, and roadmap files when implementation
  evidence reveals a local inconsistency, missing invariant, or too-weak
  contract;
- create source-study docs, probe scripts, tests, fixtures, configs, new `src/`
  files, and runtime artifacts required by approved waves;
- run local tests, probes, dry runs, and smoke runs;
- use up to 8 GPUs for tests and smokes when available, subject to the GPU
  policy below;
- make small local engineering adjustments that preserve the approved
  architecture and improve correctness, simplicity, debuggability, or
  maintainability.

## Mandatory Stop Gates

Codex must stop and ask the user before:

- moving or archiving old `src/` into `reference/legacy_src/`;
- redefining DoRA away from PEFT `LoraConfig(use_dora=True)`;
- choosing the final special-token embedding mechanism if the source study
  leaves multiple plausible paths;
- changing the first adapter-enabled acceptance smoke away from DoRA;
- weakening no-padding, no-resize, FlashAttention, MRoPE, `segment_balanced`,
  gate-loss, or same-segment causal-shift contracts;
- introducing a new production runtime dependency;
- running a long expensive multi-GPU job beyond probe or smoke scope;
- claiming DeepSpeed production support;
- expanding V1 into rollout training, hidden-state losses,
  hidden-state/KV/runtime feature caches, video, multi-image, vLLM, exact
  resume, or old coordinate-soft-CE parity;
- making any change that achieves local functionality by drifting from the
  approved research intent.

Deterministic supervised packing-cache reuse and runtime fixes to that cache
are approved training infrastructure when governed by OpenSpec receipts.
Hidden-state, KV, and runtime feature caches remain gated.

## Drift Policy

Local adjustments are allowed without asking when they preserve the approved
contracts. Examples: internal helper renames, merging or splitting tiny modules,
moving a receipt writer to a better package, adding a missing test, improving an
error class, or patching a spec scenario to match already-approved semantics.

Contract adjustments require a docs/OpenSpec patch plus review before coding
through the change. Examples: an upstream Qwen helper behaves differently than
expected, PEFT parameter names require a different optimizer matcher, FA2 branch
evidence needs a different receipt shape, or the config schema needs one more
explicit field.

Research or architecture drift requires user approval. Examples: standard LoRA
replaces DoRA acceptance, padding is introduced, full base-model tuning is
added, loss semantics change, old objective parity is revived, or DoRA becomes
a novel research mechanism rather than an implementation substrate.

## GPU Policy

Codex may use up to 8 GPUs for tests and smokes, but must manage shared-machine
contention carefully.

- Check GPU occupancy before GPU-heavy work.
- Prefer the smallest GPU footprint that proves the current gate.
- Default source-study probes and early smokes to 1 GPU unless the behavior is
  distributed-specific.
- Record visible devices, GPU count, model dtype, attention implementation, and
  memory-relevant settings in smoke/probe receipts when applicable.
- Avoid launching jobs that consume all GPUs when active training jobs make
  memory contention likely.
- For DeepSpeed or multi-rank validation, do not claim production support until
  the explicit systems smoke passes.
- If a required smoke cannot fit without disrupting active runs, wait, reduce
  scope only if the reduced run still proves the gate, or ask the user.

## MCP And Subagent Policy

- Use CodeGraph for broad codebase exploration, legacy
  invariant inventory, symbol relationships, and impact analysis.
- Use Serena for precise Python symbol reads, references, diagnostics, and
  surgical edits after narrowing.
- Use raw shell for exact commands, JSON/YAML checks, tests, probes,
  `nvidia-smi`, and artifact inspection.
- Use web only when local installed source is insufficient or current upstream
  documentation/source is needed.
- Use `superpowers:subagent-driven-development` during implementation waves.
- For implementation slices, use implementer, spec-compliance reviewer, and
  code-quality reviewer roles.
- Do not dispatch parallel implementation agents that write overlapping files.
- Parallelize read-only source studies and audits when safe.

Wave 1A is read-only investigation. Wave 1B may write probe scripts under
approved probe paths. Production `src/` implementation starts only after the
relevant source-study gates, approval cards, and wave-specific plans are ready.

## Completion Criteria

The long-term goal is not complete until current evidence proves all of the
following:

- OpenSpec V1 rebuild tasks are complete with evidence;
- old `src/` was archived only after approval, and new `src/` is the active
  implementation;
- source studies and probes pass review for DoRA, special-token
  embeddings, Qwen no-resize/MRoPE/FA2, and legacy correctness invariants;
- new tests cover config, data/template/encoding, packing/forward,
  supervision/losses, adapters/embeddings/optimizer, training artifacts, and
  the vertical smoke contract;
- the five-planned-step Qwen3-VL DoRA smoke runs successfully;
- smoke artifacts include resolved configs, manifest, Qwen setup receipt, pack
  plan, loss plan, trainable-surface receipt, optimizer receipt, metric events,
  eval.forward summaries, checkpoint metadata, and
  `checkpoints/checkpoint-final.json`;
- metrics include weighted protected losses plus top-level `acc_top1` and
  `acc_top5`;
- no V1 non-goals leaked into implementation;
- final review finds no unresolved P0/P1 findings;
- final handoff explains what exists, how to run it, what was verified, and
  what remains future work.
