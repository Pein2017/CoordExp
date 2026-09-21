---
title: Cross-Harness Normalized Replication v1 Results
description: The frozen normalized replication stopped at its real smoke gate because the hashed inference config did not contain the frozen cases; no causal result was produced.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: blocked_at_smoke
unit_id: 2026-07-23-cross-harness-normalized-replication-v1
topic: qwen3-vl-dense-enumeration
status: blocked_runtime_failure
evidence_status: protocol_and_failure_verified_no_case_outputs
conclusion_status: no_scientific_verdict_stop
authority_scope: current_pi_worktree_only
updated: 2026-07-23
---

# Cross-Harness Normalized Replication v1 Results

## Direct Verdict

This unit produced **no coordinate or set-transition causal result**. The real
smoke failed before case-level generation because the frozen inference
configuration contains an unrelated six-image input file and therefore cannot
select image `5001`. The frozen protocol separately records the correct full
validation JSONL, but the hashed runner loads `config.data.input_jsonl` and does
not apply that protocol field.

Inference had begun and both the protocol and implementation hash were frozen.
Changing either surface and rerunning would violate the explicit instruction to
stop on a protocol defect. The four-case panel was not launched. The prior
experiment remains historical evidence and is not reinterpreted as if it used
this normalized protocol.

**Decision:** stop. This unit supports neither training nor another causal
claim. A future replication would require a new unit identifier, new immutable
run directory, and a pre-smoke assertion that the effective runtime input JSONL
contains all four physical image identifiers. No training or architecture
promotion is authorized.

## 1. Protocol and Evidence Identity

- Worktree: `/data/CoordExp/.worktrees/pi-research-probes`
- Branch: `pi/research-probes-fork`
- Git commit: `41e530c016fa70f545fa230aa5209940ad153a04`
- Unit: `2026-07-23-cross-harness-normalized-replication-v1`
- Run ID: `run-20260723T072933Z-pi-41e530c0`
- Frozen protocol SHA-256:
  `b6fd2b99bd6776d6b8ae5e8f7c4d7c8038e98de1622c698bfcea735cbc8790f8`
- Immutable run root:
  `/data/CoordExp/outputs/research-probe-forks/pi-research-probes/qwen3-vl-dense-enumeration/2026-07-23-cross-harness-normalized-replication-v1/run-20260723T072933Z-pi-41e530c0/`

Before the smoke, the preparation step verified and froze:

- all 19 base-model files, totaling `8,532,755,509` bytes, directory digest
  `204d1b82f1e3cc2616328d28c96c214752bcc577acf5b5cb85287f388e44c508`;
- the Source step-4,887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter,
  directory digest
  `8c68d10a0c3fa4f5d578559f0fddfb7e89535dd5859d4f813363480dfafe8944`;
- the special-token embedding delta, directory digest
  `729c91a9c1ee4862922f30f47c3165fc612a346941c53a2bc5905a96fdee4418`;
- inference config, source JSONL, tokenizer-bearing base-model inventory,
  greedy and sampled rollouts, reviewed support ledger, trusted prefixes,
  images, prompts, candidate rows, and per-case candidate-set hashes;
- exact `x1` lag-one coordinate interventions for all four cases;
- equal-length target and matched wrong-target span interventions in every
  case;
- the rule that current actions are outside the four-complete-successor-row
  budget.

The protocol also records one pre-inference historical discrepancy. The prior
manifest specifies wine-glass control seed `21001`, but the prior emitted panel
contains the target-seed row from `21008` at the wrong-target candidate slot.
This replication froze the manifest-specified control. No normalized result was
then generated, so that correction has no observed outcome in this unit.

## 2. Smoke Outcome

Command interval: `2026-07-23T07:31:57Z` through
`2026-07-23T07:32:22Z`; exit code `1`.

Observed before failure:

- the Qwen3 Vision-Language model (`Qwen3-VL`) base checkpoint loaded;
- the configured Source adapter and embedding delta passed backend opening;
- the execution passed the full-model `torch.float32` and tied input/output
  embedding checks before entering case execution.

Failure:

```text
ValueError: image_id '5001' resolved to 0 raw examples
```

Root cause:

- frozen `config.data.input_jsonl` points to
  `six-crop-reviewed-images.coord.jsonl`, whose six physical image identifiers
  do not include `5001`, `2685`, `7511`, or `13348`;
- frozen `protocol.source_identity.source_jsonl` correctly identifies the full
  validation JSONL containing the intended cases;
- the frozen runner loads the former and never overrides it with the latter.

No coordinate baseline, coordinate edit, native action, target action,
successor row, or raw model output was generated. No `smoke.json` was written.

## 3. Coordinate-Panel Observations

None. Panel A was not executed.

- Eligible arms executed: `0`.
- Ineligible arms established: `0`; geometry eligibility was not evaluated.
- Planned but unexecuted arms: 12 per case—two baselines, eight local `x1`
  offsets, one row-order control, and one far embedding-matched control.
- Baseline determinism, one-token parity, full-vocabulary total variation,
  candidate vectors, far-control selection, decoded rows, and owner matches
  have no runtime values in this unit.

## 4. Force-Panel Observations

None. Panel B was not executed.

All six planned arms for every case remain unexecuted:

1. `native`;
2. `target_owner_distinguishing_span`;
3. `complete_target_row`;
4. `matched_wrong_target_owner_distinguishing_span`;
5. `matched_complete_wrong_target_row`;
6. `complete_already_covered_control`.

No current action or successor row was decoded. The implementation-level tests
prove the intended accounting semantics, but tests are not model evidence.

## 5. Entity-Level Verdict

No entity-level verdict is available. There are no normalized-bin predictions,
IoU-0.30 assignments, gained owners, retained owners, lost owners, or successor
set comparisons from this unit. Trusted owner `7511:-169` was frozen with its
human-reviewed entity evidence, but it was never evaluated.

## 6. Geometry-Only Sensitivity

No geometry-only result is available. There are no IoU-0.50 assignments or
coordinate-arm outcomes. The secondary geometry view cannot be inferred from
the previous run because that run used different coordinate slots and different
force-budget semantics.

## 7. Limitations, Failed Arms, and Deviations

### Scientific limitation

The smoke stopped before the first frozen example was materialized. Therefore
`normalized-results.json` contains explicit per-case/per-arm
`not_executed` records rather than fabricated raw observations.

### Failed and ineligible arms

- Failed: `person-5001-row3` smoke input selection.
- Ineligible: none established.
- Not evaluated: all coordinate-arm geometry and far-control eligibility.
- Not executed: 48 planned coordinate records and 24 planned force records
  across the full four-case protocol.

### Execution-contract deviations

1. During diagnosis, a broad process-list command unintentionally displayed a
   different harness command line. No sibling file, artifact, report, log,
   session, or Git state was opened, and the exposed metadata was not used.
   Consequently, a strict claim of zero sibling metadata exposure cannot be
   made.
2. A prepare-only stdout log was briefly written under `/tmp` before inference
   and then deleted. It contained no model output. Every retained receipt and
   artifact is under the frozen run root or the required worktree research
   directory.

There was **no scientific protocol adaptation** after inference began. The
frozen protocol and runner remain unchanged.

## 8. Training, Replication, or Stop

Stop this unit. It provides no evidence for training or architecture promotion.
Do not repair it in place or reuse its run directory.

A future replication is scientifically reasonable only as a new immutable
unit that adds a pre-load gate comparing the effective runtime
`config.data.input_jsonl` against the protocol's source JSONL and proving all
four image identifiers resolve exactly once. That recommendation is a failure
prevention measure, not a scientific continuation from observed outcomes.

## Verification and Receipts

Completed checks:

- focused tests: `12 passed`;
- Python compilation: passed;
- Git whitespace check: passed;
- Serena error-level diagnostics for the frozen runner and test: clean before
  protocol freeze;
- protocol JSON reload and SHA-256: passed;
- full source inventory and candidate-set hash verification: passed before the
  smoke;
- run-root containment checks: passed;
- smoke command and failure log: retained;
- panel launch: correctly skipped after smoke failure.

Required deliverables:

- `protocol.json`
- `normalized-results.json`
- `run-receipt.json`
- `results.md`

Runtime receipt files are under the immutable run root. The normalized result
SHA-256 is
`2a7d41a4e2aa49591b8ef9816c3cae681b6c9d38f06dc29c1dcbabb87159f0ce`.

## Harness Accounting

Measured from the current Pi session's active branch at the pre-final snapshot
`2026-07-23T08:04:24.133466Z`:

- harness: Pi coding agent `0.81.1`;
- provider: `openai-codex`;
- model: `gpt-5.6-sol`;
- thinking level: `high`;
- assistant API turns: `142`;
- tool calls: `206`;
- input tokens: `1,482,895`;
- cached input tokens (`cacheRead`): `15,148,544`;
- cache-write tokens: `0`;
- output tokens: `110,556`;
- separately reported reasoning tokens: `30,512`;
- total tokens: `16,741,995`;
- monetary cost: `$20.092964`;
- peak reported single-call context input plus cache: `287,650` tokens;
- peak reported single-call total: `288,906` tokens;
- compaction events: `1`, with `288,906` tokens before compaction;
- wall clock since the session header: `14,884.155` seconds.

These are measured session-file counters for the active branch and include the
conversation before this replication. They include closeout through the final telemetry tool call and exclude only
the not-yet-generated final response. The protocol preparation plus failed smoke spanned 100 seconds; the
smoke command itself spanned 25 seconds.
