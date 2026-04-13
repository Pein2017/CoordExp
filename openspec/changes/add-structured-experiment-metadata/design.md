## Context

CoordExp already persists several training-side artifacts with distinct roles:

- `resolved_config.json` stores the exact authored config after typed loading
  and inheritance resolution.
- `effective_runtime.json` stores the executed runtime after launcher/bootstrap
  mutation.
- `pipeline_manifest.json` stores the resolved pipeline structure and checksum.
- `run_metadata.json` stores low-level provenance such as git state, upstream
  versions, launcher metadata, and cache metadata.

The missing piece is a structured, operator-authored experiment layer. Today
that intent is encoded indirectly in `training.run_name` and
`training.artifact_subdir`, which keeps paths unique but makes run reasoning
fragile and difficult to compare across ablations.

## Goals / Non-Goals

**Goals:**

- Add a first-class, config-first `experiment` section for human-authored run
  intent.
- Create a single run-level artifact that unifies soft experiment intent with
  hard runtime and provenance summaries.
- Preserve the current authority boundaries:
  - `resolved_config.json` for exact config
  - `effective_runtime.json` for executed runtime
  - `pipeline_manifest.json` for pipeline structure
  - `run_metadata.json` for low-level provenance
- Avoid new CLI flags and preserve strict config validation.
- Keep legacy configs runnable while allowing new runs to stop depending on
  semantic `run_name` parsing.

**Non-Goals:**

- Mass-migrating every existing training config in one slice.
- Auto-generating natural-language experiment narratives from config diffs.
- Changing infer/eval pipeline config contracts in the same slice.
- Replacing authoritative config/runtime artifacts with one monolithic file.

## Decisions

### 1. Add a top-level `experiment` config section

Decision:
- Introduce `experiment.*` as a typed top-level training config section.

Rationale:
- Experiment intent is neither training runtime nor dataset contract.
- `custom.extra` is intentionally an escape hatch, not a stable audit surface.
- A dedicated top-level section makes the intent explicit in leaf YAML and keeps
  strict unknown-key handling intact.

Alternatives considered:
- Put experiment metadata under `custom.extra`.
  Rejected because it would remain a residual bucket rather than a first-class
  contract.
- Put experiment metadata under `training.*`.
  Rejected because it would mix authored semantics with downstream trainer args.

### 2. Keep authored prose separate from exact runtime truth

Decision:
- The `experiment` section stores authored prose and lightweight grouping data.
- Exact machine truth remains in `resolved_config.json`,
  `effective_runtime.json`, and `pipeline_manifest.json`.

Rationale:
- Authored summaries are valuable for humans but can drift if treated as the
  exact source of truth.
- The repo already has authoritative machine artifacts for exact settings.
- Separation lets analysis tools compare runs using exact artifacts while still
  presenting the authored hypothesis and deviations.

Alternatives considered:
- Encode exact settings directly inside the authored experiment block.
  Rejected because it duplicates authoritative runtime/config artifacts and
  invites drift.

### 3. Emit `experiment_manifest.json` as the primary run-level overview artifact

Decision:
- Add `experiment_manifest.json` as the first artifact a human or analysis
  script should open when orienting to a run.
- Populate it with:
  - `identity`
  - `experiment` (authored soft metadata)
  - `runtime_summary`
  - `provenance_summary`
  - `artifacts`

Rationale:
- This gives one structured entrypoint for both soft and hard context without
  overloading any one existing artifact.
- It reduces the need to infer context from directory names.
- It remains scalable because deeper details still live in specialized files.

Alternatives considered:
- Expand `run_metadata.json` to hold everything.
  Rejected because `run_metadata.json` is already described and used as a
  provenance-oriented artifact.
- Stuff experiment metadata into `resolved_config.json` only.
  Rejected because the resolved config is authoritative but not optimized as an
  operator-facing run summary.

### 4. Preserve legacy compatibility by making `experiment` optional

Decision:
- `experiment` is optional in v1.
- When absent, the run still emits `experiment_manifest.json` with
  `experiment.authored` set to `null` and the hard-runtime/provenance sections
  populated.

Rationale:
- The repo contains many existing leaves and historical workflows.
- We can ship the new contract now without forcing a bulk config rewrite.
- New and touched configs can adopt the richer metadata immediately.

Alternatives considered:
- Require `experiment` for all training configs immediately.
  Rejected because it would create noisy repo-wide churn unrelated to runtime
  correctness.

## Risks / Trade-offs

- [Risk] Authored experiment prose can become stale relative to config changes.
  → Mitigation: keep exact machine truth in `resolved_config.json` and
  `effective_runtime.json`; position `experiment.*` as narrative context, not
  canonical runtime truth.

- [Risk] Another artifact can increase operator confusion.
  → Mitigation: document `experiment_manifest.json` as the “open this first”
  artifact and clearly define the role of each existing file.

- [Risk] Downstream tools may assume only `run_metadata.json` exists.
  → Mitigation: keep `run_metadata.json` in place for compatibility and update
  repo-owned consumers to recognize `experiment_manifest.json`.

- [Risk] The new top-level section could weaken strict config behavior if left
  loosely typed.
  → Mitigation: parse `experiment` through the same strict schema discipline
  used elsewhere in training config loading.

## Migration Plan

1. Add typed schema support for top-level `experiment`.
2. Emit `experiment_manifest.json` from the training bootstrap path.
3. Keep `run_metadata.json` as the low-level provenance artifact.
4. Update docs/tests and at least one representative config example to show the
   preferred authoring pattern.
5. Migrate more leaf configs incrementally as they are touched for future work.

Rollback:
- If the new manifest proves unhelpful, keep the schema support and stop
  treating `experiment_manifest.json` as primary; existing concern-specific
  artifacts remain valid.

## Open Questions

- Should infer/eval pipeline configs later gain a sibling `experiment` section
  so standalone offline runs can author the same narrative context?
- Do we want a future repo tool that indexes `experiment_manifest.json` across
  run directories for cross-run comparison?
