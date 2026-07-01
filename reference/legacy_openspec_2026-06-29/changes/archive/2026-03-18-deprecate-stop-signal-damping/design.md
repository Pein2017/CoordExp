## Context

`adaptive-stop-signal-damping` was implemented as an opt-in teacher-forcing
experiment, but empirical rollout behavior now shows it is toxic: it leads to
duplicate-heavy proposals and poor dense-scene behavior. The rollback must
prevent future accidental use while keeping the repo's historical record clear.

## Decisions

### 1. Deprecation is hard fail-fast, not silent disable

Any authored `token_ce.config.stop_signal_damping` block must raise an explicit
error telling the user to remove it. This prevents inert compatibility stubs
from keeping the feature discoverable or silently reviving it through old YAML.

### 2. Runtime stop-signal behavior is removed completely

The token CE runtime no longer computes stop-signal weighting, and the trainer
no longer emits `stop_signal_ce` atoms or `stop_signal/*` diagnostics. This
keeps the live training contract aligned with the deprecation.

### 3. Stable docs and checked-in YAMLs are scrubbed

Operator-facing docs and canonical profiles must stop advertising the
experiment. Dedicated smoke/reference configs created only for this feature are
removed.

### 4. The original change is archived as historical context

The original `adaptive-stop-signal-damping` change should be archived without
syncing specs into `openspec/specs/`. This preserves history while removing it
from the active change set.

## Risks / Trade-offs

- Existing configs that still author the deprecated block would fail to load if
  they are not scrubbed in the same patch.
- Leaving stale metrics/tests/docs behind would create the impression that the
  experiment is still supported.
- The semantic stop metadata helpers may remain as dead cleanup candidates; they
  are not required for the immediate safety barrier once the config/runtime path
  is removed.
