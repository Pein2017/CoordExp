## Why

The completed Human-13 overfit screen shows a useful but narrow one- or
two-exposure K-hit-to-greedy region, together with two concrete failure modes.
The existing duplicate objective rejects only the final coordinate token and
does not redirect probability toward an uncovered valid owner row; longer-dose
failures are dominated by invalid rectangles; and low-dose gains can replace
previously greedy-visible `G` owners through coordinate drift.  A bounded
successor is needed to test the smallest native fixes before considering a new
architecture or a broader optimization loop.

## What Changes

- Add an experiment-local, content-addressed successor ledger derived from the
  sealed Human-13 manifest and already-authoritative low-dose outputs.  It
  materializes complete duplicate rows, their exact decision prefixes,
  covered-owner state, owner-normalized uncovered native candidates, and
  frozen `G` coordinate watch sites.  It does not relabel K-miss owners.
- Replace final-coordinate-only duplicate rejection with a hierarchical
  row-level contrast.  When an uncovered same-description owner exists, compare
  the duplicate bbox score against owner-normalized valid bbox alternatives;
  otherwise compare the complete owner-distinguishing row against uncovered
  valid rows.  With no trusted uncovered alternative, use bounded pure bbox
  rejection and never manufacture a `STOP` target.
- Add a greedy-aligned rectangle-validity gate at `x2` and `y2`: the highest
  coordinate token that yields positive width or height must outrank the
  highest invalid token.  This constrains geometry rather than forcing a
  token-identical GT box.
- Add two fresh-Source, fresh-AdamW treatment arms only.  `R1` combines the
  existing A4 any-valid target direction and Source replay with row contrast
  and the rectangle gate.  `R2` is identical but projects an adverse target
  gradient away from the frozen `G`-coordinate watch gradient before the
  optimizer step.  Reuse authoritative A4 exposure-two as the comparison;
  do not rerun it.
- Run only cumulative exposures `1` and `2`.  First require one real
  production-shaped image-14038 vertical slice through materialization,
  no-padding packing, forward/backward, AdamW, checkpoint write/read, HF
  fp32/SDPA batch-one clean greedy, and the existing analyzer.
- Use at most six GPUs on the critical path: two independent one-rank training
  jobs and four independent checkpoint evaluations.  Keep two available GPUs
  as operational reserve rather than adding non-critical work.
- Keep the work deliberately narrow: no online target refresh, K-miss
  supervision, positive terminal target, GT-IoU coordinate oracle, external
  bridge, long-dose continuation, production API change, or generalized loss
  framework.  Stop after conclusion-changing checks; do not over-audit or
  over-design this research probe.

## Capabilities

### New Capabilities

- `coordexp-swift-human13-row-contrast-successor`: Experiment-local contracts
  for hierarchical duplicate-row contrast, rectangle-valid greedy gates,
  frozen `G`-coordinate gradient preservation, bounded R1/R2 execution, and
  exact same-panel outcome projection.

### Modified Capabilities

None. Existing production packing, training, checkpoint, inference, parser,
and evaluator contracts remain unchanged and are reused by experiment-local
adapters.

## Impact

- New or extended Human-13 research scripts, pure loss helpers, typed sidecar
  records, two leaf configs, focused tests, and immutable experiment artifacts.
- Reuse of the sealed Human-13 manifest, A4 exposure-two result, current
  no-padding FlashAttention-2 path, language-tower DoRA surface, one-rank
  AdamW runtime, HF fp32/SDPA evaluator, and one-to-one IoU matcher.
- The owning research unit remains authoritative for arm meaning, evidence,
  stop rules, and claims.  OpenSpec owns only the bounded implementation and
  execution contract.
- No new dependency, architecture, production behavior, validation claim, or
  checkpoint promotion.
