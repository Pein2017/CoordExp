# Conditional numerical-repeat probability

Unit `2026-09-19-recurrence-conditional-mass` asks whether sustained native
greedy recurrence coexists with low conditional probability of the frozen
numerical repeat event, or whether the conditional distribution places
substantial mass on that event.

## Authority and boundary

Root `01a0a3d5-dc45-7693-8467-4801aa7190df` owns scientific acceptance and
interpretation. Lane C owns this unit's local runner, raw draws, reduction and
artifact binding. The shared frozen panel and source manifest are owned by Lane
A and are consumed byte-for-byte after their final publication. This unit does
not select or replace panel states, and it is independent of Lane B's spatial
admission decision.

The population is at most 48 ORIGINAL native panel states: the frozen failure
and non-recurrent proxy states supplied by the shared panel. Model, checkpoint,
image, policy, source trajectory, prompt and prefix identities remain per-state
strata. Existing production-val200 records, normalized readout records, forced
routes and any other policy are excluded from the primary denominator.

## Frozen conditioning and event

For every state, the panel supplies the native prefix ending at the next-row
boundary and a current description. The primary conditional distribution is
formed after appending the exact serialized opener and current-description
prefix through `<|box_start|>`, using the tokenizer's actual single-token
serialization. The opener/description boundary is measured separately from
the native row-boundary logits and is never folded into the primary event.

The runner derives the bounded horizon from that same serializer before model
calls. For the current closed-row format this is normally four coordinate
tokens followed by `<|box_end|>`, but the receipt records the actual token IDs
and count rather than assuming a constant. The primary event is one complete
legal bbox plus its required terminator whose four coordinate bins belong to
the panel's existing same-description historical-repeat union at the frozen
`<=8`-bin predicate; overlapping union members count once. This is a
conditional numerical-repeat event, not total physical duplicate probability.
Invalid geometry within that union is reported as an
`invalid_geometry_near_repeat` (`<=8` bins), with an exact-bin invalid subset
reported separately when available.

Each draw uses the ORIGINAL full-vocabulary softmax at temperature 1, with no
top-k/top-p truncation and repetition penalty 1. A token outside the expected
grammar interval is retained as a grammar/format escape and is a non-event;
the draw is never retried or forced into a coordinate. Invalid geometry,
length/cap and invalid-geometry near-repeat outcomes are retained separately.
`1-q` is not new-owner probability.

## Fixed sampling and denominators

There are exactly 256 independent draws per admitted state, at most 12,288
draws total. The frozen native sampling batch is eight requests. Each batch
chunk gets one deterministic 63-bit stream seed before output generation,
derived by domain-separated SHA-256 from the unit ID, state ID, model,
split-aware `source_example_id` and chunk index. Every draw records that
stream seed, chunk index and within-chunk offset, making the actual generator
stream replayable without claiming per-draw reseeding. No resampling,
temperature sweep, top-k/top-p sweep, prefix-tree expansion or long
continuation is allowed.

The state denominator is the number of panel states with a completed valid
native qualification and all 256 attempted draws. Wilson 95% intervals are
reported per state over that state's draws. Pooled and model/source stratum
fractions are descriptive captured-draw summaries across heterogeneous states,
not image-level uncertainty estimates. Image-level summaries give each image
one unit and retain model/policy strata; no cross-image interval or population
rate is claimed. Failed qualification, missing source binding and runtime
errors remain explicit exclusions with reasons.

## Qualification and controls

One actual native model-entry qualification runs before the scientific batch:
it verifies source/model/policy identity, native image preparation, exact
prefix token preservation, actual serializer horizon, finite full-vocabulary
softmax, deterministic seed replay, and terminal-row parsing. A CPU known-event
self-consistency fixture compares the native full-vocabulary sampler against a
constrained-coordinate sampler. It must demonstrate that native sampling can
escape the coordinate family and that constrained sampling is a separate
control; it is not scientific evidence.

The qualification also captures row-boundary opener/EOS and description-entry
logits as diagnostics. These scores do not condition or rescue the draw event.
No stale KV cache is reused: every state reconstructs its native prefix and
image inputs through the current loader.

## Stop rule and deliverables

Stop after the qualification fails, the fixed state denominator is exhausted,
the 48-state/256-draw ceiling is reached, or the allocated four GPU-hour
provisional budget is exhausted. A mechanical runtime repair may preserve the
same contrast and must be recorded; no completed scientific draw is rerun.

The lane returns raw sampled token IDs and stop/escape reasons, compact token
scores where captured, source/prefix/image/model bindings, conditioning and
horizon receipts, per-state event counts with Wilson intervals, descriptive
image-level summaries, the CPU reducer and standalone acceptance commands.
It reports model calls, GPU-seconds, bytes and all remaining processes. The
result is a candidate for root review; Lane C does not self-accept or launch a
successor.
