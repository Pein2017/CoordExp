## Why

Qwen3 Vision-Language (Qwen3-VL) dense enumeration can expose verified objects through stochastic
same-prefix sampling that ordinary greedy rollout misses, while other states
enter a duplicate, premature terminal, or geometrically incorrect branch. The
approved research unit now needs a bounded training path that acts directly on
those rollout-derived decision sites without adding an inference-time
architecture or mixing canonical supervised-fine-tuning sequences into the
pilot.

The scientific design is owned by the
[Own-Prefix Entity-Transition and Coordinate-Boundary Calibration Training Screen](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/unit.md).
This change supplies only the minimum reusable implementation needed to run
that screen.

The completed 256-image coordinate-boundary screen established that the
objective improves the requested margin at frozen Source-checkpoint prefixes
but does not improve clean self-rollout. The authorized successor therefore
adds one bounded offline trajectory-refresh comparison, owned by the
[Mixed Old-Prefix and Refreshed-Prefix Coordinate Correction](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/unit.md)
research unit. It tests whether the failed transfer is caused by stale prefix
states rather than by an ineffective local objective.

The exact-greedy-terminal successor was then closed before multi-image
training because its 256-image census produced only three unique accepted
events and its one-event smoke induced a broad repetition burst. The completed
successor is the
[Best Sampled Trajectory Positive Row Imitation Screen](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/unit.md).
It reuses the same exact-prefix StateBank and trainer while adding one
positive-only complete-row profile over a deterministically selected sampled
path. Its 512-event one-epoch screen made greedy output shorter and sometimes
cleaner, but reduced unique annotated-owner coverage at every evaluated
milestone. A targeted owner-identity follow-up shows a route-level shift toward
the physical-owner set represented by the selected routes, but ordinary owners
are lost at almost the same rate and non-admitted images regress. Only 118 of
the 238 route-added owners are direct positive-row event targets, and most of
the shift occurs on other owners in the same route family. The positive-path
profile is retained as an implemented research surface, while its identical
1,024-image coverage replication is not promoted. A separate preservation-
aware, matched-arm 256-image treatment is the next candidate.

## What Changes

- Add a frozen, exact-token rollout state-bank input path with physical-entity,
  review, provenance, split-group, candidate, and geometry eligibility data.
- Add same-prefix first-divergence branch preference for verified uncovered
  positives versus the actual harmful greedy branch, followed by
  mean-normalized coherent positive-row continuation supervision.
- Add first-wrong-coordinate supervision against a reviewed discrete set of
  acceptable coordinate tokens.
- Apply a small token-type gate at every rollout-derived decision site with a
  defined intended token type.
- Add transition-only, coordinate-boundary-only, and joint research profiles,
  with independent eligibility masks, normalization, metrics, and compact run
  receipts.
- Reuse the current full image-and-prefix Qwen forward, Weight-Decomposed
  Low-Rank Adaptation setup, optimizer/runtime, checkpoint writer, and
  inference pipeline.
- Add the one-event and 8-to-16-state smoke paths required before the formal
  screen.
- Add an explicit, default-off off-policy replay mode in which a frozen
  StateBank remains truthfully bound to the checkpoint that generated its
  trajectories while training warm-starts from a later compatible checkpoint.
- Keep refreshed rollout collection and StateBank construction outside the
  trainer, then compare an equal-budget repeat on old prefixes with correction
  on newly visited prefixes from one shared intermediate checkpoint.
- Extend transition events with fixed-budget counterfactual admission evidence
  so a locally rescued row is not trained when it merely reorders owners or
  creates larger downstream set harm.
- Add a target-scoped non-coverage prefix mode for premature-terminal events
  only, so one verified missed owner can be trained without falsely claiming
  that every earlier row or the complete covered set is resolved.
- Add one positive-path-imitation-only profile with no harmful candidate. It
  keeps unresolved history as zero-gradient context, supervises verified
  complete rows through the last added owner, equalizes total weight per image,
  and separately mean-normalizes schema-and-description sites and trusted
  coordinate sites.
- Do not add canonical supervised-fine-tuning replay, Kullback-Leibler
  divergence anchoring, Gaussian coordinate smoothing, online collection
  inside the trainer, new model heads, or inference-time control modules.
- **BREAKING for the new research profile only:** permit a declared
  rollout-calibration run to omit protected full-row base cross-entropy while
  retaining a positive-weight token-type gate over its selected research
  sites. Existing supervised-training profiles and defaults remain unchanged.

## Capabilities

### New Capabilities

- `coordexp-swift-own-prefix-calibration-training`: Defines frozen exact-prefix
  state-bank replay, entity-transition and first-wrong-coordinate objectives,
  rollout-site token-type gating, research-arm configuration, diagnostics, and
  smoke evidence.

### Modified Capabilities

- `coordexp-swift-supervision-losses`: Allows one explicit rollout-calibration
  research profile to omit full-row base cross-entropy while requiring a
  positive-weight token-type gate on every selected site with an intended
  token type; ordinary supervised-training behavior is unchanged.

## Impact

Expected owner surfaces are `src/config/`, `src/data/` or a narrow research
state-bank loader, `src/supervision/`, `src/losses/`,
`src/training/pipeline.py`, and existing training artifact/logging paths.
Inference code is reused for state collection and post-training evaluation but
its public model-forward and decoding behavior do not change. No new external
model or detector dependency is introduced.
