## Why

The inspected Pi sessions have healthy ordinary-request reuse but pooled cache hit below 90% because maintenance requests add approximately 250K uncached input each; Luna also has unexplained isolated zero-cache ordinary requests. Large raw JSONL/ledger dumps inflate new input, while current accounting and transport logs cannot distinguish these causes.

## What Changes

- Wave 1: add bounded, read-only diagnostic projections for Pi session journals and explicitly selected JSON/ledger fields, without changing raw evidence or built-in read semantics; split Web usage into ordinary inference, compaction, branch-summary, cache warming and other usage while preserving the existing total.
- Wave 2: add opt-in, bounded, content-free final-request diagnostics and usage-presence evidence, correlating ordinary Web inference and extension-owned compaction with their sessions and requests. Observe after request transformations, not merely an early context hook.
- Wave 3: carry session cache affinity into context-management compaction through the supported provider interface, qualify preserved and intentionally changed prefix regions, and retain opaque-checkpoint persistence, cancellation and continuation behavior. Cache hits remain backend observations, not a guaranteed 90% target.
- Use ordered batch acceptance: Wave 1 before Wave 2, Wave 2 before Wave 3; ask-advisor reviews the design before implementation.

## Capabilities

### New Capabilities

- `pi-bounded-artifact-diagnostics`: read-only bounded journal and explicit JSON-field projections with source locations, truncation/completeness disclosure and preserved raw data.
- `pi-cache-usage-breakdown`: additive categorized usage and cache metrics that reconcile exactly to the existing all-session totals.
- `pi-final-request-diagnostics`: opt-in request-boundary fingerprints and raw cache-field presence without prompt/credential logging or behavioral mutation.
- `pi-compaction-cache-affinity`: compatible context-management request affinity and explicit prefix-transition qualification while preserving checkpoint semantics.

### Modified Capabilities

None. The current stable spec inventory has no Pi capability covering these behaviors; existing in-flight TPS and compaction changes remain separate owners.

## Impact

Planning home: `/data/CoordExp/openspec/changes/pi-cache-efficiency-and-diagnostics`, as resolved by OpenSpec from main and both nested source repositories; no standalone store is registered.

Source owners: `/data/CoordExp/codex-tools/pi-web` for diagnostic consumers, accounting/UI and Web request observation; `/data/CoordExp/codex-tools/pi-extensions/packages/pi-codex-compact` for extension-owned requests and checkpoint behavior. Explicit artifact selectors may read Research Probes evidence but do not change its scientific source, records or active work. Shared profile `/data/CoordExp/.pi` is configuration/adoption state, not a source-edit target. Official Pi core, managed node_modules, the legacy compaction repository, proxy configuration and credentials are not edited.

No additional dependencies, model/backend switch, generic read-output redaction, journal rewrite or new scientific run is planned. Deterministic SDK/local transport tests precede a separately bounded hosted qualification. No commit, publication, versioned-release adoption or active-session reload is implied by source implementation. Concurrent TPS/subagent work must be retained. Terminal ordinary-request observability is not a claimed delivery surface; the shared compact request improvements cover Web and CLI when adopted.
