## Context

The desired online-learning behavior is not full-sequence SFT. Channel-B should
roll in on the model's own compact-full prefix, identify the first reliable
oracle correction point, and train only the legal next action set at that
state. The target may be a multiple-positive set while the residual state is
ambiguous, but once a teacher-forced token fixes the branch, the rest of that
object follows a strict singleton teacher path until the next ambiguity point.

## Decisions

### Live Tokens Are Observations, Not Positives

`input_ids[target_position]` may be the model's live self-prefix token. When it
is not one of the oracle `ValidAction` tokens, it must not be inserted into
`valid_token_ids`. The atom records the live token as provenance and marks the
supervised target as a correction. Loss is computed against the oracle valid
set.

### Ambiguity-Scoped Multiple Positives

Multiple positive tokens are allowed only when the residual-state machine
enumerates multiple valid next actions for the same slot under the current
prefix. This covers object-description starts, shared description prefixes, stop
when the residual set is empty, and coordinate tokens only when several objects
remain genuinely indistinguishable until that coordinate slot.

If the selected teacher token narrows the residual state to one object, all
later object-internal atoms use singleton valid sets. This is the intended
"collapse" from ambiguous trie marginal to strict CE.

### First-Error OPD Surface

The first implementation uses the existing residual-set event surface:

- dirty or malformed spans remain masked or dropped when they cannot be
  localized safely;
- reliable spatial wrong-description and continuation events produce
  correction atoms;
- the adapter may emit a target mismatch atom when the live token differs from
  the oracle selected token;
- the residual-set loss treats singleton valid sets as hard CE and larger valid
  sets as marginal likelihood.

Future grammar-specific malformed-row repair can add more `CorrectionEvent`
types, but it must obey the same rule: oracle actions define positives; live
tokens do not.

## Diagnostics

The residual-set module reports:

- total atoms and atom weights;
- ambiguous-token targets where `len(valid_token_ids) > 1`;
- strict singleton targets where `len(valid_token_ids) == 1`;
- target-token mismatches between the live self-prefix token and the oracle
  selected token;
- coordinate ambiguous targets, so we can verify whether coord ambiguity is
  rare or a real crowded-object surface.

These metrics are intended to explain whether malformed growth comes from
format drift, wrong-type mass, over-broad schema CE, or invalid live-token
reinforcement.

## Non-Goals

- No full autoregressive marginal over hidden alternative prefixes.
- No new offline rollout-input mode.
- No broad Stage-1 SFT mixing.
- No desc semantic synonym oracle beyond current residual-state object matching.
