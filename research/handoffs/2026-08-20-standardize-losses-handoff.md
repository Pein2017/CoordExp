# Handoff: standardize-coordexp-swift-supervised-losses (start of build)

Written 2026-08-20 by the Claude Fable lead session that completed and
archived the decompose change. Start a fresh task from this file plus the
change's own OpenSpec artifacts; do not carry the decompose session context.

## Program position

reconcile (ARCHIVED 2026-08-19, `eb2dc97ab`) → decompose (ARCHIVED
2026-08-20, `68191f7ea`, 59/59, zero-P0/P1 completion audit) → **standardize
losses (this change, 0/31)** → observability (0/40).

## Authority and inputs

- Owning change: `openspec/changes/standardize-coordexp-swift-supervised-losses/`
  (proposal/design/tasks are the sole scope authority).
- SDD plan: `docs/superpowers/plans/2026-08-12-standardize-coordexp-swift-supervised-losses.md`
  (execution notes only; the OpenSpec change owns scope).
- Current owners after decompose: see `docs/IMPLEMENTATION_MAP.md` — losses
  live in `src/losses/` (runner owns configured loss assembly and
  normalization), supervision records in `src/supervision/tokens.py`,
  facade `src/training/pipeline.py`, model-bearing lifetime
  `src/training/session.py`, trainer `src/training/supervised_trainer.py`.
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`, branch
  `coordexp-swift`, HEAD at handoff `68191f7ea`, tree clean,
  `openspec validate --all` 21/21.

## Authorization scope

The 2026-08-19 blanket pre-authorization ("我先提前授权所有的内容...") was
scoped to the decompose program and is SPENT. CPU-only spec/test/impl work on
this change proceeds under the standing program direction; any launch-bearing
action (GPU run, production cache mutation, paid externals) needs a fresh
packet with its own authorization basis — ask the user.

## Live production facts a loss change must not silently break

- Pack cache v3 root `.cache/coordexp_swift/packing` holds exactly two
  immutable targets (train `8f11237f…`, eval `3b30c157…`) for the smoke
  config; loss-semantics changes that alter cache determinants (owner set in
  `src/training/pack_cache.py`, 31 entries incl. `supervision_tokens`,
  `micro_step_schema`) change fingerprints → a rebuild is a NEW absent-target
  publication, never a mutation. Check the change's tasks for whether a
  turnover is in scope before touching any determinant owner.
- Frozen characterization fixtures `tests/fixtures/training_orchestration/`
  are byte-frozen evidence; loss work must not regenerate them.
- Logging rows carry per-loss keys (`loss/base_ce`, `loss/token_type_gate`,
  `finite/*`, segment counts, token_weighted_diag) — completed-step row
  schema is a protected compatibility surface (see
  `archive/2026-08-20-decompose…/receipts/wave-8-compatibility-comparison.json`).
- Losses runner: `src/losses/runner.py`; normalizer currently
  `segment_balanced` in smoke/prod configs.

## Working pattern that held (keep)

Lead + ephemeral Agent-tool workers; independent Opus auditor with
author≠signer separation at wave gates; freeze→verify→execute in one
no-commit window for anything launch-bearing; acceptance only on replayable
receipts (counts from JUnit, sha256 inventories, git plumbing), never worker
self-report; one bundled correction round per task.

## Host/tooling traps (all hit this week; receipts in decompose archive)

- rtk hook silently no-ops `env`-prefixed and `env -u`-prefixed commands
  (exit 0, zero output, no side effects). Use `bash -c 'export …; unset …; …'`
  wrappers and confirm real output, never exit codes alone.
- heredoc piped into `conda run` swallows stdout — write a script file, then
  `conda run -n ms python file.py`.
- `run.json` `measurement.phases` dict is key-sorted; execution order is
  `measurement.phase_order`.
- Single-process full-matrix pytest on this GPU host cannot be CUDA-pure:
  CUDA-initializing suites poison CPU-semantics probes
  (`Wave6ProbeError: CUDA was initialized before CPU comparison`). Judge
  matrix failures by failure-set diff + fresh-process replay, never counts.
- Poisoned same-length/same-second `.pyc`: always set
  `PYTHONDONTWRITEBYTECODE=1` for gates and mutation probes.
- `conda run -n ms` for all Python; GPUs shared — check `nvidia-smi` and pick
  free devices; user can pause GPU 6/7 jobs on request.

## Open housekeeping (user-visible, outside this worktree's authority)

- `/data/CoordExp/AGENTS.md` and
  `/data/CoordExp/.worktrees/research-probes/AGENTS.md` carry the 2026-08-19
  superpowers/TDD calibration edits but are uncommitted in their own trees.
