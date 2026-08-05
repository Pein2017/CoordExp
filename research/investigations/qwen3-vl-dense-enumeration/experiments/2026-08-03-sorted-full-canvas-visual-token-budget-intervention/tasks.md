---
title: Sorted Full-Canvas Visual-Token-Budget Intervention - Tasks
description: Implementation, smoke, capture, analysis, and independent-audit checklist for the full-canvas token-density probe.
type: investigation
role: research-tasks
authority: non_normative_research
unit_id: 2026-08-03-sorted-full-canvas-visual-token-budget-intervention
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-03
---

# Tasks

## Design and implementation

- [x] Freeze the single changed factor, causal question, strongest alternative,
  cohorts, calibration, retention gates, claim boundary, and stop rule.
- [x] Complete independent Fable scientific design and Opus preprocessing-seam
  audit before implementation.
- [x] Implement CPU treatment-pool preparation and byte-exact current-pool
  reproduction.
- [x] Implement a compact sealed prompt/identity overlay over the predecessor
  plan without rebuilding owner/candidate geometry.
- [x] Implement the score-only treatment shard wrapper with exact arm and
  overlay binding.
- [x] Implement treatment calibration, recovery, retention, uncertainty, and
  decision analysis.
- [x] Add focused unit and synthetic-contract tests.

## Representative smoke

- [x] Pass CPU preparation and invariance checks for all twelve images.
- [x] Pass one real-HF current-arm parity case on image `13348`.
- [x] Pass one real-HF doubled-budget case on image `13348`.
- [x] Measure candidate batches `1`, `8`, and `16`; freeze the largest admitted
  value before the full capture.
- [x] Review the emitted media/grid/prompt/arm receipt and one conclusion-
  bearing score row end to end.

### Evidence for the items above

Smoke run root `20260803T161241Z`.

- Lead post-fix regression: `272 passed` across the five focused preparation,
  wrapper, analyzer, predecessor-scorer, and predecessor-merge files; Ruff
  undefined-name/import checks are clean.
- CPU preparation: byte-exact current-pool reproduction from raw for **all
  twelve** images (dimensions and `executed_media_sha256`).
- Current-arm parity on `13348`: across the 28 matched candidate rows, maximum
  `complete_box_logprob_sum` difference `2.574920654296875e-05`; the 1000-bin
  x1 vectors are identical.  Both are within the frozen `1e-3` parity gate.
- Doubled-budget case on `13348`: `1014 -> 1944` merged visual tokens, grid
  `[1, 72, 108]`, all admitted, no free-decode rows.
- Candidate batches `1`, `8`, `16` all admitted; `16` frozen for the full
  capture.
- Treatment stress smoke on `14038:boundary-000|book` scored all **561**
  candidates at batch `16` in `190.66 s`: all admission receipts passed, all
  batched/scalar argmax decisions agreed, and the maximum score difference was
  `2.956390380859375e-05` against the frozen `1e-4` bound.  The executed media
  digest, grid `[1, 72, 108]`, and `2292`-token prompt matched the treatment
  stamp; observed memory remained well below the available 80 GiB.
- The same stress receipt correctly declares
  `intervention_completeness.status = "subset_smoke"`, with `1/215` frozen
  groups executed, and therefore cannot enter conclusion-bearing analysis.
  Its canonical `free-decode-sidecars.jsonl` is zero bytes and the receipt
  reports zero free-decode rows.
- Post-fix conclusion-bearing row trace: candidate
  `cand:0112f69f85d3b3ee79feb152` under `14038:boundary-000|book` retains strict
  physical-owner assignment `gt:14038:31`, the four sealed coordinate tokens,
  treatment media/prompt/overlay identity, and a finite complete-box score.
  The row and whole-shard receipt were reviewed together.

- End-to-end review (lead): `treatment-b16/13348/shard-receipt.json` inspected
  whole — status `captured`, arm/media/grid/prompt/overlay/base-plan identities
  aligned, `admission.all_admitted=true`, score-only receipt authoritative — and
  one conclusion-bearing score row `cand:05b0c2326ef1a62a1b689052` traced end to
  end (query group, normalized candidate geometry and tokens, strict owner
  assignment `gt:13348:0`, admission receipt, plan/capture hashes, treatment
  stamp all consistent).

The earlier review surfaced a provenance seam that has since been fixed:
because the successor executes the overlay's
frozen selection, which is intentionally a subset of the predecessor base plan,
`census.run_shard` stamps the base-plan-relative
`capture_completeness = "subset_smoke"` even for a complete intervention
capture. The successor now publishes a separate, intervention-relative
`intervention_completeness` block (`complete_frozen_overlay_selection` versus
`subset_smoke`, with frozen expected count, executed count and missing/extra ID
lists), the base fields are preserved untouched as provenance, and the analyzer
requires the new block before any exact set check. The reviewed `b16` artifact
predates that field and remains mechanical evidence only; the `14038` stress
smoke is the post-fix contract witness that closes the item.

## Capture and analysis

- [x] Allocate one fresh immutable run identifier.
- [x] Launch selected doubled-budget shards largest-first across available
  GPUs, with no silent OOM fallback.
- [x] Verify every selected shard is complete and carries one arm, overlay,
  predecessor-plan, model, and scoring identity.
- [x] Recalibrate treatment support on discovery native-TP controls.
- [x] Compute persistent recovery, person-only recovery, image spread, Wilson
  intervals, leave-one-image-out sensitivity, native-TP retention, resolved
  retention, and the restricted `4134` report.
- [x] Apply the frozen minimum-effect and retention gates without threshold
  motion.

### Capture and analysis evidence

- One immutable attempt root, `20260803T161241Z`, was allocated before the
  representative smoke and retained through the full capture; no partial run
  was reinterpreted under a new identity.
- All `12/12` treatment shards completed with one uniform arm, overlay,
  predecessor plan, model, tokenizer, runtime, and scorer identity.
- The complete capture contains `1,211` query groups and `167,579`
  localization score rows; all `24` receipt/score files are sealed under
  aggregate digest
  `ca822d8f659af7573dbc13c45982e9a1863a18ac1de37a3e26f655dedda1cbbc`.
- Treatment calibration used exactly `70` discovery native-true-positive
  controls and produced thresholds `2.5555611686696498` for `peak_lift` and
  `1.708293092250824` for `local_concentration`.
- The analyzer reports nominal recovery `14/63`, person-only `11/51`, seven
  images, confirmation-TP retention `64/71`, and resolved retention `86/114`.
  The final retention conjunction and minimum decisive outcome are false.
- CPU-only full reanalysis reproduced the report. The focused analyzer suite
  passed `67` tests; the combined analyzer/visualizer regression passed `160`
  tests with Ruff and `py_compile` clean.

## Review and closure

- [x] Run independent Fable audit over the fixed implementation, smoke, raw
  scores, analyzer, and proposed claim.
- [x] Resolve every decision-bearing audit finding or mark the unit blocked.
- [x] Write `results.md` only after execution and audit close.
- [x] Update the experiment index and research compass only if the route
  changes.

The contract and scientific audits found no unresolved decision-bearing
artifact discrepancy. They independently confirmed that the arm is invalid,
not positive density evidence. The experiment index and compass now route to
the CPU-only supported-false-negative reachability-prevalence unit.
