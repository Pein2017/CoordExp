# Execution handoff

This handoff continues the completed bounded
[Human-13 K-Union-to-Greedy Overfit Screen](unit.md). The scientific outcome is
owned by [results.md](results.md); the OpenSpec change remains the implementation
scope record. This file is continuity, not independent evidence or authority for
a successor experiment.

## Current disposition

- `COMPLETE_BOUNDED_SAME_PANEL_NARROW`.
- Source/K discovery, canonical manifest, one real update slice, five applicable
  sixteen-update arms, all `1/2/4/8/16` clean-greedy evaluations, and the pooled
  analysis are complete.
- A4 was omitted by its frozen 12,000-token atomic-bundle bound; A6 failed
  closed before model load because donor provenance did not match; A8-prime was
  unavailable because the no-update census never produced a valid artifact.
  These are mechanical unknowns, not scientific nulls.
- No further optimizer update, checkpoint promotion, stable-spec sync, archive,
  push, publication, or production change is authorized by this handoff.

The user asked not to over-audit or over-design. Do not repair absent arms or
extend the dose merely to make the matrix rectangular.

## Authoritative evidence

Artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-12-human13-k-union-to-greedy-overfit-screen
```

- manifest: `manifest/human13-k-union-manifest.json`, SHA-256
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`
- frozen ledger: `392` owners, `G=173`, `H=73`, `M=146`
- combined outputs: `analysis/matrix-v1-outputs.jsonl`, `338` rows, SHA-256
  `054b548f082d8d2a16fb4588e05b976763796b9e6266e82f6f0d446829aff481`
- analysis: `analysis/matrix-v1-analysis.json`, SHA-256
  `cff3f9f26ba61417a74e06bf8fce178c20f067e51528e95e6329a0b0a4cfaa8e`

Read order:

1. [results.md](results.md) for the complete table, interpretation, limits, and
   next decision;
2. [review.md](review.md) for planning and execution-review reconciliation;
3. [unit.md](unit.md) for frozen experimental semantics;
4. `openspec/changes/add-human13-k-union-greedy-overfit-probe/` for implemented
   surfaces and honest incomplete mechanics.

## Decision-bearing result

At one exposure, native language-tower DoRA improves same-panel greedy owner
coverage without an external bridge:

- A3: the largest pooled H gain among executed one-exposure H arms,
  `H+6`, `G-3`, `M+1`, unique owners `173 -> 177`, but partly same-image
  owner exchange;
- A7: `H+5`, `G-2`, unique owners `176`, but larger duplicate burden;
- A1: `H+5`, `G-4`, unique owners `174`, with the smallest targeted output
  burden and more safe images;
- full-GT capacity: `H+12`, `M+8`, `G-5`, unique owners `188`, while rows,
  duplicates, and unmatched predictions increase substantially.

All executed treatment arms deteriorate after the one-exposure region. By
exposure sixteen A3 and A7 have thousands of rows, hundreds of duplicates,
large malformed burdens, cap stops, and more Source-owner loss than H gain.
The static-target `optimize-until-satisfied` route is therefore rejected.

## Successor decision, not authorization

If the user chooses another experiment, restart from the byte-identical Source
and compare only a very-low-dose frontier:

1. owner-balanced H1 for one exposure;
2. full residual body CE for one exposure with H/M contributions kept
   separate; and
3. the same target direction with a stronger Source-preservation constraint.

Select on the Pareto tuple of H gained, G lost, M gained, duplicate/malformed/
cap burden, and generated tokens. Do not resume these checkpoints, use
post-update same-batch decoding to choose another update, or infer fresh-image
transfer from this panel.

## Claim boundary

The result establishes same-panel optimization capacity only. It does not show
validation gain, generalization, full-set mastery, duplicate-free decoding,
safe gradient direction, support expansion, architecture necessity, or
production readiness.
