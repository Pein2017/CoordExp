## Why

The approved Human-13 research unit needs a bounded way to test whether native
K-sampled, already IoU-valid owner rows can be consolidated into one clean
greedy completion.  The current generic rollout-calibration path does not own
this hash-bound overfit cohort, coherent full-residual bottleneck objective, or
same-panel arm orchestration, and broadening it would weaken existing blind-data
and production contracts.

## What Changes

- Add one experiment-local probe that binds the owning research unit, exact
  panel and checkpoint identities, explicit K-sampling requests, chronological
  duplicate-before-matching owner ledger, raw and duplicate-cleaned prefixes,
  arm plans, and outcome projection.
- Collect K=16 sampled trajectories as four batches of four independent
  `n=1` requests per image, with explicit seeds and sampling repetition penalty
  `1.10`; preserve request-to-seed identity instead of relying on child-seed
  expansion.
- Add the minimum pure objective math required by the approved matrix:
  owner-mean masked row cross-entropy, once-per-image any-valid row mass,
  coherent full-residual token bottleneck margin, and one coordinate-closing
  numerically stable unlikelihood site for every frozen duplicate event.
- Reuse the accepted no-padding varlen FlashAttention-2 path.  Within one arm,
  deterministically length-pack isolated segments under the existing 12,000
  token bound and accumulate the complete panel objective before one AdamW
  update.  Across the available eight GPUs, run independent arms as independent
  one-rank Accelerate processes rather than introducing a new multi-rank
  choreography.
- Add a no-update bottleneck census, dry-run materialization, one-image
  production-shaped vertical slice, compact performance counters, and the
  original-prompt batch-size-one HF clean-greedy analyzer needed to support a
  later launch decision.  The real vertical slice is preceded by a separately
  authorized full-panel Source/K discovery and manifest-freeze gate.
- Freeze the initial sixteen-update AdamW/DoRA surface, scheduler, clipping, and
  objective-family coefficients so arm identities cannot drift during
  implementation; any later long-dose run restarts Source and optimizer.
- Keep the implementation intentionally narrow: use the shortest evidence path,
  add no speculative general framework or duplicate evidence journal, and stop
  after conclusion-changing checks.  This operationalizes the user requirement
  **not to over-audit or over-design**.
- Do not implement or launch A2, A5, native candidate-tree training, online
  frontier refresh, GT-IoU coordinate search, K-miss supervision, an external
  owner bridge, checkpoint promotion, or any GPU/model execution in this
  planning change.  Implementation and all accelerator execution remain behind
  a later explicit user authorization.

## Capabilities

### New Capabilities

- `coordexp-swift-human13-k-union-greedy-probe`: Experiment-local contracts for
  explicit batched K sampling, frozen owner/prefix ledgers, minimal approved
  loss arms, no-padding panel-step packing, compact receipts, and clean-greedy
  outcome projection while preserving the default blind-cohort and production
  behavior.

### Modified Capabilities

None.  Existing packing, training-artifact, inference, evaluator, and generic
rollout-calibration requirements remain unchanged and are reused rather than
weakened or duplicated.

## Impact

- New experiment-local research scripts, typed records, configs, pure loss
  helpers, and focused tests under the current CoordExp-Swift source tree.
- Reuse of `src/packing`, `src/qwen`, `src/training`, `src/inference`, and the
  existing one-to-one detection matching semantics without changing their
  public production contracts.
- A new OpenSpec delta for the probe only.  The owning research unit remains the
  sole authority for the cohort, `G/H/M` interpretation, arm meanings, primary
  outcome, stop rules, and permitted scientific claims.
- No new dependency, no production blind-policy exception, no execution-model
  architecture change, and no launch or material-cost authorization.
