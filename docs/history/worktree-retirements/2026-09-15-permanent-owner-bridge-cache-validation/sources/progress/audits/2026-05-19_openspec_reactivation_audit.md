---
doc_id: progress.audits.openspec-reactivation-2026-05-19
layer: progress
doc_type: audit
status: active-audit
domain: governance
summary: OpenSpec reactivation status, validation cleanup, and active-change triage.
tags: [openspec, audit, governance, training, stage1, stage2]
updated: 2026-05-19
---

# OpenSpec Reactivation Audit (2026-05-19)

Scope: thin OpenSpec/codebase-architecture hygiene pass. This audit did not
run training, val200 inference, expensive experiments, or implement a
`trie_marginal` objective.

Bulky command outputs are under:

```text
temp/openspec_reactivation_20260519/
```

## Command Outcomes

| Command | Outcome |
|---|---|
| `rtk git status --short --branch` | Initial state was `main...origin/main` plus unrelated untracked `colleague_prompt.md`. |
| `openspec --version` | `1.3.1`. |
| `openspec list --changes --json` | Worked; reported 12 non-archived changes for the committed reactivation layer. A local uncommitted `add-stage1-trie-marginal-objective` draft may temporarily raise this to 13 during review. |
| `openspec list --specs --json` | Accepted the flag but emitted the text `Specs:` table, not JSON; initially listed 30 specs, and lists 31 after adding `stage1-latest-detection-objectives`. Treat this as a CLI-output quirk until the CLI is fixed or the command contract changes. |
| `openspec validate --all --strict --json --no-interactive` before cleanup | Failed: 28/42 passed, 14 failed. Failing items were 2 changes and 12 specs, mostly stale delta headers in main specs and missing parser-visible `SHALL`/`MUST` text. |
| `openspec validate --all --strict --json --no-interactive` after cleanup | Passed: 42/42 items, 12/12 changes, 30/30 specs before the new Stage-1 objective owner was added. |
| `openspec validate --all --strict --json --no-interactive` after Stage-1 objective contract update | Passed: 44/44 items, 13/13 changes, 31/31 specs when the local trie-marginal draft change is included. The committed reactivation layer remains strict-valid without that uncommitted draft. |
| `rg -n "openspec list --type\|openspec validate .*--type change\|openspec validate .*--type spec" docs openspec/changes --glob '!openspec/changes/archive/**'` after cleanup | No matches. |

No pytest was run because the edits were docs/OpenSpec/progress only and no
runtime Python, config parser, trainer, inference, or evaluator files changed.

## Validation Cleanup Applied

The cleanup was format-preserving rather than behavior-changing:

- converted stale `## ADDED Requirements` / `## MODIFIED Requirements` headers in
  stable `openspec/specs/` files into normal `## Purpose` + `## Requirements`
  sections;
- added explicit parser-visible `SHALL`/`MUST` sentences to requirements that
  already had normative meaning but failed strict validation;
- added one missing scenario to `teacher-forcing-unified-loss-registry` and one
  to `rollout-matching-sft`;
- normalized active-change validation examples away from `--type change`;
- clarified docs routing so `docs/` and runbooks remain current-behavior truth,
  while `openspec/specs/` is the stable contract layer.

## Contract Layer Added

The follow-up contract update used integration rather than one spec per module:

- added `openspec/specs/stage1-latest-detection-objectives/spec.md` as the
  stable capability owner for latest-schema Stage-1 compact recursive detection
  objectives;
- prepared a local
  `openspec/changes/add-stage1-trie-marginal-objective/` draft as a
  contract-only active change for future review, but did not require it for the
  committed reactivation layer;
- routed current Stage-1 latest compact detection configs in
  `docs/catalog.yaml` to the new stable objective contract;
- updated `docs/AGENT_INDEX.md` and `docs/training/README.md` so future agents
  know to integrate new objective variants into the capability spec through an
  active change.

## Active Change Matrix

| Change | Status | Recommended action |
|---|---|---|
| `add-adjacent-distributional-repulsion-loss` | `implemented-but-drifted` | Reconcile tasks/specs with implemented Stage-2 `coord_reg` adjacent knobs and metrics, then archive after focused config/coord-reg/Stage-2 tests. |
| `add-center-size-bbox-supervision` | `complete-unarchived` | Archive as legacy Stage-1 SFT `custom.bbox_geo.parameterization=center_size` support; do not promote it as latest compact recursive detection behavior. |
| `add-duplication-collapse-analysis-study` | `complete-unarchived` | Archive as historical/mechanism analysis evidence; it should not steer current training behavior. |
| `add-lvis-coco-proxy-supervision` | `blocked-needs-user` | Decide whether to finish the original Stage-2 metadata-weighting contract or split/archive current data/export/Stage-1 work and open a smaller Stage-2 follow-up. |
| `add-stage1-et-rmp-ce-objective` | `implemented-but-drifted` | Reconcile old set-continuation wording with the current latest-detection ET-RMP-CE comparator/ablation surfaces, then archive or rewrite into a latest-detection capability. |
| `add-stage1-set-continuation-training` | `stale-historical` | Treat as superseded/rejection-only unless deliberately resurrecting the older trainer family; a resurrection should be a fresh change, not silent reactivation. |
| `add-structured-experiment-metadata` | `complete-unarchived` | Archive after syncing the change-local `experiment-metadata` contract into stable specs if needed. |
| `adopt-cxcy-logw-logh-bbox-parameterization` | `implemented-but-drifted` | Reconcile with the offline bbox-format preprocessing contract, including `cxcywh`, then archive as absorbed/superseded. |
| `birth-first-stage2-channel-b` | `current-active` | Keep active only if the next Stage-2 decision round still wants this behavior; otherwise retire explicitly. No current implementation was found. |
| `refactor-offline-bbox-format-preprocessing` | `current-active` | Keep active and finish before archiving bbox-parameterization work; priorities are provenance/fail-fast validation, cache/run metadata, and approved smoke/retrain decisions. |
| `refactor-training-runtime-architecture` | `implemented-but-drifted` | Split remaining future mission/recipe orchestration from the completed runtime-plan slice, then archive/sync the completed setup-plan contract. |
| `support-norm1000-raw-text-stage1` | `complete-unarchived` | Archive after documenting or explicitly deferring the one remaining smoke-workflow task; keep it as legacy Stage-1 raw-text benchmark, not latest compact detection. |

## Recommended Buckets

Archive candidates after standard archive/sync checks:

- `add-center-size-bbox-supervision`
- `add-duplication-collapse-analysis-study`
- `add-structured-experiment-metadata`
- `support-norm1000-raw-text-stage1` after closing or deferring the remaining
  smoke-workflow task

Repair before archive:

- `add-adjacent-distributional-repulsion-loss`
- `add-stage1-et-rmp-ce-objective`
- `adopt-cxcy-logw-logh-bbox-parameterization`
- `refactor-training-runtime-architecture`

Keep active:

- `refactor-offline-bbox-format-preprocessing`
- `birth-first-stage2-channel-b`, only if still desired as a current Stage-2
  research contract

Requires user decision:

- `add-lvis-coco-proxy-supervision`
- `add-stage1-set-continuation-training` if anyone wants to resurrect the old
  trainer family instead of treating it as historical/superseded

## Trie-Marginal Readiness

The repo is now ready for future trie-marginal implementation work from an
aligned OpenSpec state. The stable capability owner exists at:

```text
openspec/specs/stage1-latest-detection-objectives/spec.md
```

and the recommended contract-only active change path is:

```text
openspec/changes/add-stage1-trie-marginal-objective/
```

Do not implement it as part of this reactivation pass. The change should stay
contract-focused first and cover:

- objective name, for example `trie_marginal` or
  `sampled_path_trie_marginal`;
- ambiguous trie nodes use valid-set/subtree marginal only;
- no balance loss at ambiguous nodes;
- no selected-child hard CE at ambiguous nodes;
- hard CE after branch commitment;
- same-description coordinate-onset ambiguity;
- candidate filtering after coordinate choices;
- EOS vs valid-continuation diagnostics;
- old ET-RMP-CE paths preserved as comparator baselines.

Residual risks:

- `openspec list --specs --json` still emits text despite the `--json` flag.
- Strict validation is now clean, but semantic drift remains in the
  `implemented-but-drifted` changes above.
- No archive command was run in this pass; completed changes remain
  non-archived by design.
