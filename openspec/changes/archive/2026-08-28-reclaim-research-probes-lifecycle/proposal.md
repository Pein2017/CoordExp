## Why

The research-probe lifecycle designed on 2026-08-24 (`establish-research-probes-baseline-v1`) was executed only up to tagging: no probe has ever completed the fork → return-records → retire loop (zero `probe-final/*` refs), 30+ probe worktrees/branches sit 80–500 commits behind `research-probes`, and the research main itself carries the pre-baseline era's scaffolding (315 flat `scripts/research/*.py`, 84 experiment unit directories, results landing in `memories/` instead of `research/`). The user now wants `research-probes` to serve as a clean research main from which several long-lived, direction-level worktrees fork in parallel and return their records; that requires executing the lifecycle once for real and reclaiming the accumulated entropy.

## What Changes

- **Simplify the lifecycle to what is actually load-bearing.** New research directions fork a `probe/<direction>` branch at `.worktrees/<direction>` from `research-probes` HEAD; the admission owner already binds the exact commit and clean status, so `research-base-vN` tags become milestone markers cut after reusable-mechanics merges, not fork prerequisites. Return is records-only (`research/` units, routers, provenance); code stays on the direction branch and is preserved by a `probe-final/<direction>` annotated tag. Code promotion, when a real second consumer exists, may go either `probe → research-probes` or `probe → research-probe-infras → research-probes`; the default remains experiment-local.
- **Make the integration lane unambiguous.** `.worktrees/research-probe-infras` is permanent; its branch is renamed to `research-probe-infras` after the stale, unmounted branch of that name (`62274a97d`) is tagged and deleted. This supersedes the "superseded HOLD" disposition recorded by `establish-research-probes-baseline-v1` task 1.4.
- **Retire dead lanes with evidence preserved.** 15 fully-returned lanes (7 wave-1 `human13-*`, 8 `rp-crossover-*`) and the closed Human-13 N/K factorial lineage receive `probe-final/<name>` tags; 12 pre-baseline lanes with un-returned content receive `archive/<name>` tags; then worktrees are removed and local branches deleted. `image2299-mechanism-microscope` stays live. Remote branches, `permanent-owner-bridge-cache-validation` (HOLD), root `main`, `coordexp-infras`, and the tooling worktrees are untouched.
- **Return the one un-returned closed result.** The Human-13 N=13 K4/K8 factorial result exists only as a `memories/notes/` entry plus a `contract.md` on its branch; this change creates its `research/` unit before its lane is retired, exercising the return checklist once.
- **Reclaim scaffold entropy by deletion, not relocation.** `scripts/research/` scripts with zero consumers or test-only consumers are deleted end to end (script, dedicated tests, dedicated configs); replay of any historical result binds to the `research-base-v2` tag, which contains every deleted file. Scripts cited by research records as replay entries and shared mechanics imported by other production scripts are kept. `src/` receives an audit-first pass in the same evidence format; cuts there require separate user approval.
- **Records hygiene.** Four zero-inbound root-level reports and one orphan `research/*.md` move to `docs/history/`; four investigations idle since July move to `research/archive/`; `docs/catalog.yaml` and routers follow; a retroactive governance receipt records `research-base-v2`; the five completed OpenSpec changes are archived; regenerable ignored residue (`.pi-worker/` 1.7G, `.serena/cache`, `$evidence_dir/`, `temp/`) is deleted.
- **Cut `research-base-v3`** at the reviewed post-cleanup commit and fast-forward the integration lane to it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. This is lifecycle governance, ref/worktree retirement, record hygiene, and evidence-gated deletion of experiment-local scaffolding; no supported runtime, config schema, artifact, or admission behavior changes. `skip_specs: true`.

## Impact

- Routers and configuration: `docs/BRANCH_AND_WORKTREE_POLICY.md`, `docs/AGENT_INDEX.md`, `docs/PROJECT_CONTEXT.md`, `openspec/config.yaml` context, `docs/catalog.yaml`.
- Git refs: ~28 new annotated tags (`probe-final/*`, `archive/*`), ~28 local branch deletions, ~26 worktree removals, one branch rename, one new baseline tag.
- Tree: deletions under `scripts/research/`, `tests/research/`, research `configs/`; moves under `research/`, `docs/history/`; new research unit for the N/K factorial result; receipts under this change.
- Disk: ~2.6 GB of ignored residue removed from `.worktrees/research-probes`; retired worktree directories freed.
- Live consumers that must keep working: `image2299-mechanism-microscope` (imports `src.inference.hf_backend`, `src.vis`, `src.config.inference`, `src.data.geometry`, `src.qwen.runtime_loading` read-only), the admission owner and its two consumer adapters (44 CPU-only tests), and every `research/` replay citation.
