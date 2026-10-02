## Why

The root ignored output tree mixes branch-owned runs, handwritten reports and executable scripts with potentially shared checkpoints. Main still instructs launches from the root checkout; recent worker transport reuses the run directory for human messages. A 29-root inventory measures about 234 GiB, with live holders and immutable path-bound receipts, so indiscriminate directory moves would lose provenance or break consumers.

## What Changes

- Clarify the existing storage/worktree guides: owner-worktree outputs for branch runs; root outputs only for explicitly selected durable shared runtime assets. Separate maintained code/research records and disposable transport messages from runtime receipts.
- Establish a locked, detached main-owned linked worktree at the verified main commit for main-owned output storage; no branch checkout override, merge, launch or automatic lifecycle cleanup.
- Correct the lead-worker assignment/report routing at its existing skill owner, preserving the already-dirty transport reference and shared AGENTS unchanged.
- Move detailed inference/evaluation operator steps into the existing inference/evaluation skill; keep `docs/eval/` as a concise route, contract, and interpretation surface.
- Extract useful COCO/LVIS audit code and interpretation to their actual data/research owners; extract the research-owned HF compatibility probe. Move confirmed inactive, consumer-qualified runtime files individually with hashes and explicit producer provenance.
- Evaluate a selected checkpoint for explicit shared retention; never promote its entire benchmark run or overwrite a destination.
- Absorb all remaining human-authored research records into their existing research owners, preserving evidence status and provenance, then remove the redundant output copies.
- Remove obsolete checkpoint, rollout, inference, frozen-source, and script payloads only after exact active-writer and consumer checks; record source/script path, size and hash without retaining executable copies.
- Retire old Label Studio after exporting its two pending drafts; continue Gate A. The latest user ruling defers annotation relocation to the next public-data migration, so this round preserves its existing root runtime and published views.
- Qualify legacy content by an actual current consumer or named necessary reproduction before assigning a destination; a generic move into main-runs is not retention acceptance. Retire obsolete payloads after useful meaning is distilled and recorded. Preserve frozen receipt fields as provenance, update current consumers, and remove originals without old-path symlinks or historical binary/source copies.
- Keep a small Git-managed inventory summary and the exact large per-file machine receipt in the maintenance worktree outputs. Use this finite record, not a new governance registry.

## Capabilities

### New Capabilities

None. This is operator routing, bounded maintenance tooling and content migration.

### Modified Capabilities

None. `skip_specs: true` applies: no training/inference/checkpoint schema, scientific rule, data recovery guarantee or continuation gate is changed. Existing `coordexp-infras-training-artifacts` and Research Probes `research-probe-development` remain their behavioral owners. In particular, historical source bindings stay immutable and dirty/unqualified sources gain no permission to execute. Do not invent an output-governance capability to duplicate the current operator guides.

## Impact

Root main owns the OpenSpec change and global storage/main-worktree policy. The canonical Research Probes checkout owns its local policy, runtime probe and scientific interpretation. Main's public_data owns retained annotation-audit tooling; its existing untracked v2 exporter/tests are not edited. The shared lead-worker skill gets only message-versus-receipt placement guidance.

The 2026-10-02 user ruling supersedes the earlier permanent legacy-path holds.
Dataset labels, model tensors, loss/geometry/matching rules and Codex transcripts
are unchanged. Frozen source/script identities are metadata, not retained
executable archives. New training/inference writers reject root outputs; selected
shared assets remain valid inputs. Annotation relocation is explicitly deferred by the latest user ruling;
Gate A retains its existing live root. Old Label Studio is retired and its two
drafts are exported without publishing or importing them. Do not stop training jobs or ambiguous external log producers. No GPU
research, staging/commits/pushes, worktree removal, remote transfer or environment
change is authorized. Report actual remaining execution blockers; do not accept
old layout as a permanent final form.
