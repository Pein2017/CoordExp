## Why

Starting a small probe still requires copying config-to-template conversion and image/prompt assembly from a historical experiment. Maintained profiles also repeat the same Source256 DoRA binding. These copies obscure the separation between reusable execution mechanics and scientific choices, and the current CPU preflight prepares the same images more than once.

The user accepted a complete developer workflow: select a few explicit examples, inspect generation prefixes and annotated targets without weights, reuse the prepared inputs for exact replay or a small update, and interpret saved results through existing paired reducers. The goal is lower maintenance burden and measured preparation efficiency, with existing scientific behavior preserved.

## What Changes

- Add one shared input-planning operation at the existing inference input owner. It composes existing rendered, image-plan, prompt, native-request and optional target-encoding values; generation prefixes and annotated targets remain separate.
- Reuse rendered/image plans instead of repeating their construction. Keep pixel materialization explicit through the existing native operation, with caller-owned processor/model lifetime.
- Migrate the equivalent DORA, Logit Lens and Human13 preparation paths and the inference pipeline's repeated config conversion. Remove superseded implementations after caller-level parity checks.
- Provide a small direction-local CPU inspection entry accepting an existing resolved inference profile and explicitly selected IDs or count/seed, without the historical Source256 cohort gate. Existing strict preflights retain all their gates.
- Consolidate the identical Source256 language-DoRA binding in its existing direction runtime. Keep the fixed training surface, objective, optimizer, DDP order and artifact ownership in their profiles. Exclude a consumer if immutable source bindings prevent a compatible migration.
- Document and exercise the complete prepared-input → native replay/update → existing diagnostic-consumer path. Do not create another trainer, trajectory format, plugin registry, or universal reducer.
- Measure old/new preparation time and operation counts on the same real rows. Report any speedup only for the measured path; lower code duplication does not prove model throughput improvement.

## Capabilities

### New Capabilities

- `research-probe-input-preparation`: explicit tiny-cohort inspection, separate generation/annotated-target preparation, and reusable native inputs with preserved media/token/span identity.

### Modified Capabilities

None. Existing native replay, research-profile loading, fixed experiment semantics, artifact integrity and lifecycle requirements remain unchanged. The Source256 binding consolidation is an internal frozen-contract refactor.

## Impact

Expected owners: `src/inference/{inputs,prompt,image_plan,pipeline}.py`, `src/qwen/encoding.py`, DORA input/runtime and inspection code, Logit Lens and Human13 request construction, their focused tests and operator documentation. No new dependency or production promotion is intended.

Implementation lives in `/data/CoordExp/.worktrees/research-probes-upgrade-20260912`, branch `codex/research-probes-upgrade-20260912`, forked from `ef6d44d1196a7043deeef90cefc90136bce9859f`. The active `parallel_owner_research` package, its records, `tests/test_training.py`, and dirty compass/index are excluded. Merge back is authorized only after the active task releases affected executable dependencies and the latest integration candidate is verified; the canonical worktree lock remains intact.

The user authorized at most one simultaneous GPU and 60 cumulative GPU minutes for mechanical real-model acceptance. No quality experiment, dtype change, ReFT adaptation, distributed topology change, reward/credit change, historical artifact rewrite or active task interruption is included. Internal APIs may change with migrated callers; existing CLI and persisted scientific meanings remain compatible.

Evidence: the upstream-comparison report at `/data/CoordExp/.worktrees/coordexp-infras/research/investigations/ms-swift-upstream-comparison-2026-09-12/research-probing-report.md` (SHA256 `7053ecbe8762cc15dda3fa0d47b20ae665c423ce23c6a72a7f9036fe75424d2a`), independently checked against current source. Existing public DoRA selection, magnitude restoration and diagnostic reducers make several report suggestions unnecessary. Full frozen training producers remain byte-bound and are not general cleanup targets.

Isolation and resource receipt: `/data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912/isolation-baseline.json`. Scientific acceptance is outside this engineering change.
