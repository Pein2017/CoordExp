# Wave-0 Baseline (tasks 0.1-0.5)

Recorded 2026-08-20 by the Claude Fable lead (session c9895ff9).

## 0.1 Predecessor pins

| Fact | Value |
| --- | --- |
| reconcile | ARCHIVED `2026-08-19-reconcile-coordexp-swift-training-contracts` (`eb2dc97ab`) |
| decompose | ARCHIVED `2026-08-20-decompose-coordexp-swift-training-orchestration` (`ebfa78ff1`/`68191f7ea`, skip_specs, completion audit 0 P0/P1) |
| standardize-losses | ARCHIVED `2026-08-20-standardize-coordexp-swift-supervised-losses` (close `ed9d01b53`, delta-preservation `452664a2d`/`f8a8c1a1b`, archive+spec-sync `3d390b108`; completion audit 0 P0/P1) |
| this change's base commit | `3d390b108` (tracked tree clean at recording) |
| `openspec validate --all` | 20 passed / 0 failed post-losses-archive |

No disagreement found between stable specs, code, docs, or archive
dispositions (the losses completion audit verified the docs at the same
source bytes this change starts from).

## 0.2 Delta rebase (executed by a dedicated builder, lead-verified)

The three spec deltas (authored 2026-08-12 against pre-losses stable text)
were rebased onto the freshly synced stable specs. Preserved verbatim: the
5 lead-enumerated missing scenarios plus 3 builder-found paragraph-level
phrase losses ("sum of weighted objective terms", "term-count and
finite-status", "forward-eval rows", and the losses-authored protected-
diagnostic sentence in Non-Finite Gates). Task-0.2 proof obligations now
stable text inside the delta: canonical `loss/<term>/raw` + configured
weight + `loss/<term>/weighted`; the bare alias MUST NOT be restored;
omitted zero-weight optional terms have no field family. Deliberate
pre-losses-base modifications retained and flagged (timing-barrier
relaxation, "unacknowledged corrupted", "or synchronized skips") — all
authored 2026-08-12 intent, not stale-base reverts. `Rank-Zero
Presentation Sinks` verified under ADDED. Lead re-verification: scenario
diff zero-missing, strict validation valid, zero quote-terminated bare
aliases. Replay scripts: jobs tmp `scenario_diff.py` / `grep_proof.sh`.

## 0.3 Owner/import graph pins

- To-be-created owners correctly ABSENT at base: `src/runtime/metrics.py`,
  `src/artifacts/observation_publisher.py`.
- Existing seams present: `src/training/reporting.py` (368 lines, canonical
  rows), `src/training/session.py` (3534, wiring), reduction currently in
  `src/runtime/train_runtime.py::gather_metrics/_reduce_metric_reports`
  (Wave-2 migration source).
- Facade check: `src/training/pipeline.py` has zero
  console/tensorboard/SummaryWriter references; no TensorBoard usage
  anywhere in `src/` at base. No competing owner.
- Cache invariant carried from the losses change: fingerprints
  `8f11237f…`/`3b30c157…`; observability surfaces are not determinant
  owners; `observability.steps` must be confirmed absent from determinant
  payloads at the Wave-1 gate (payload reads enumerated in the losses
  change's wave-0 receipt).

## 0.4 Command manifest

`receipts/command-manifest.json` frozen at `3d390b108`: per-wave gate argvs,
append-only amendment rule, cache invariant, standing-GPU-grant note
(packets + bounds still required for tasks 3.8/5.5), and the two
TO-FREEZE placeholders (wave-2 gloo probe, wave-3 fp16 GPU probes,
wave-5 smoke) to be retired by amendments when their packets freeze.

## 0.5 Entry baseline

`wave0-entry-baseline` argv (config/losses/runtime/reporting/artifacts/
exact-resume suites): **890 passed / 0 failed / 0 skipped** at the
untouched base commit. Entry audit receipt: `wave-0-entry-audit.md`.
