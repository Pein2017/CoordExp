---
title: PVCI Own-prefix Behavioral Evaluator Contract
description: Frozen research-local evaluator semantics for the PVCI own-prefix causal and safety panel.
type: idea
role: research-contract
authority: non_normative_research
status: complete
unit_id: 2026-07-11-pvci-causal-proposal-bridge
updated: 2026-07-12
---

# Own-prefix behavioral evaluator contract

This is a research-local, non-benchmark evaluator for the frozen PVCI causal
proposal-bridge panel.  It is intentionally separate from official detection
evaluation: its purpose is to compare matched decoder behavior under the
predeclared proposal conditions, not to establish a COCO headline.

## Inputs

The command consumes a panel JSON containing the full sixteen-cell condition
panel:

```text
A, B, C_off, C_on, C_another_image, C_token_permutation, C_position_only,
C_norm_matched_random × RP1.10, RP1.00
```

Each object supplies `condition`, `rp`, `run_dir`, and `receipt`.  `RUN_DIR` must contain
canonical `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `run_manifest.json`,
and `summary.json`; the scored artifact must be materialized.  The receipt
must be the final `pvci-own-prefix-condition-v2` receipt emitted by the
eight-shard producer—not a panel summary, an inference manifest, or a hand
written alias.  It is self-hashed and binds the exact condition, repetition
penalty, ordered 320 request IDs with the producer's **newline-delimited**
digest, full cohort source/output/descriptor receipts, authored and runtime
config artifacts, adapter and embedding payload-tree hashes, checkpoint
handoff and proposal-bridge state identity, generation policy, every merged
artifact (including score provenance, token trace, diagnostics, and image
plan), controller diagnostics, and the producer entrypoint, semantic module,
controller module, proposal-bridge module, plus Python/Torch runtime fingerprint.
The evaluator rehashes all those bound
files and directories; it does not search alternative keys or infer missing
identity from an adjacent panel.  All cells must agree on cohort and ordered
image IDs.  C-on and every C control must agree on checkpoint state and
runtime-config fingerprint within each RP.

The panel is a JSON object, for example:

```json
{
  "conditions": [
    {
      "condition": "A",
      "rp": "1.10",
      "run_dir": "/abs/A_rp110",
      "receipt": "/abs/A_rp110/own_prefix_condition_receipt.json"
    }
  ]
}
```

The abbreviated example is not launchable: the real payload lists all sixteen
condition/RP cells exactly once.

The evaluator consumes `src.vis.normalization.load_visual_rows` and the
frozen `src.vis.matching.match_row` implementation.  It does not reparse text
or use a second assignment rule.

## Outputs

- `own_prefix_image_metrics.jsonl`: raw per-image metrics and matcher receipt.
- `own_prefix_aggregate.json`: contextual means and count strata.
- `own_prefix_gate_decision.json`: 10,000-resample paired image bootstrap,
  seed `20260711`, and the mechanical gate decision. The research unit applies
  its ordered terminal label separately during closeout; that prose label is
  not emitted or receipt-bound by this evaluator.
- `own_prefix_eval_receipt.json`: PASS or fail-closed contract receipt.

A PASS receipt hashes the three emitted metric/gate artifacts and includes a
canonical self-hash.  This prevents a valid input receipt from being paired
with stale or modified evaluator outputs.

Reported per-image values include matched-row precision, labeled recall,
annotated under-enumeration, premature annotated stop, natural closure,
invalid/malformed/truncated flags, raw duplicate image/pair/component counts,
prediction/match counts, and unmatched predictions retained as `unknown`.
No missing denominator is silently converted to zero.

Invalid predicted geometries are preserved in the raw prediction count and in
the precision/unknown denominators.  They are excluded only from canonical
IoU matching and duplicate-geometry calculations, where no valid box exists.
Parser drops, evaluator-invalid geometries, controller-invalid rows, and the
official COCO converter's dropped predictions are distinct counters and must
not be collapsed into one generic failure count.  This preserves the cost of
malformed continuation without inventing a box for matching.

`premature_annotated_stop` means that a rollout terminated naturally while at
least one *annotated* GT object remained unmatched by the frozen matcher.  On
incompletely annotated COCO images it is not a claim that every visible object
was covered, and unmatched predictions remain `unknown` rather than automatic
hallucinations.  Official COCO bbox evaluation, when run, is an artifact-scoped
secondary diagnostic with `benchmark_eligible=false` and
`benchmark_metric=false`; it cannot replace the behavior gate or support a
headline benchmark claim.

`causal_bridge_supported` requires the four explicit negative-control artifact
cells to show no material benefit over C-off, rather than trusting a prose
assertion.
`decode_specific_bridge` means RP1.10 passes the primary screen while RP1.00
has a predeclared contradictory lower confidence bound.

Safety is evaluated independently against both A and C-off at RP1.10.  It
includes overall duplicate delta/CI, duplicate delta for the `9+`-GT stratum,
invalid and malformed deltas, the `0.95` natural-closure floor,
under-enumeration delta/CI, and premature-stop delta/CI.  Every check is
reported under `safety.by_baseline`; missing required evidence cannot pass.

## CLI

```bash
python scripts/probes/coordexp_swift/evaluate_pvci_own_prefix.py \
  --panel-json /abs/own_prefix_panel.json \
  --output-dir /abs/own_prefix_behavior_gate
```

The evaluator exits `2` and writes a `contract_fail` receipt on malformed,
tampered, missing, unpaired, or cohort-incompatible input.
