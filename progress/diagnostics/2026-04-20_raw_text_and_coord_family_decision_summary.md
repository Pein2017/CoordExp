---
title: Raw-Text and Coord-Family Decision Summary
date: 2026-04-20
status: decision-summary
owner: codex
depends_on:
  - progress/diagnostics/2026-04-20_raw_text_coord_continuity_probe.md
  - progress/diagnostics/2026-04-20_coord_family_basin_and_recall_comparison.md
---

# Raw-Text and Coord-Family Decision Summary

## Why This Note Exists

The two super-power studies answered related but different questions:

1. does raw-text pure-CE already have continuity, and what does that imply for
   `coord_token`?
2. across available 2B coordinate families, which setups are actually strong,
   stable, and worth pursuing?

This note merges those answers into one decision-oriented supporting summary.

## Cross-Study Conclusions

### 1. `coord_token` is not needed to create continuity

The raw-text continuity study established that:

- raw-text numeric continuity already exists in the base model
- pure-CE fine-tuning can strengthen it
- the effect is visually modulated
- and the effect can be both useful and harmful

Therefore:

- `coord_token` should no longer be justified primarily as a continuity-creation
  mechanism

If `coord_token` remains justified, that justification must come from:

- typing discipline
- decoding stability
- cleaner parameterization
- or more favorable instance-separation behavior

### 2. `center_parameterization` is the current strongest 2B family

Among the families already strong enough to matter operationally:

- `center_parameterization` is the best current performer
- it has the best matched-val200 metrics
- and it looks more baseline-efficient than `raw_text`

In short:

- if the goal is strongest current checkpoint behavior, `center` is the
  practical winner

### 3. `raw_text_xyxy_pure_ce` is the most promising research family

`raw_text` is not the strongest deployed family, but it is the most revealing
research family because:

- it already has real continuity without `coord_token`
- it reaches a strong second-place performance band
- and it still has a large recoverable recall gap

That combination means:

- there is still headroom to improve `raw_text` through decoding stability,
  ranking, or proposal realization
- without giving up the core finding that continuity can emerge natively

### 4. `base_xyxy_merged` and `hard_soft_ce_2b` are not good frontier bets

These families look weak for a deeper reason than mere recall conservatism:

- they have very poor AP
- their Oracle-K uplift is negligible
- their misses are overwhelmingly systematic
- and they show structured invalid-geometry failures

So the most honest read is:

- these are not the best families for the next research dollar

### 5. The real remaining research problem is not "make continuity"

The new problem statement should be closer to:

- how do we keep raw-text or center-like models from collapsing under dense
  repeated-object competition?
- how do we turn partially-supported detections into stable baseline outputs?
- and which parameterization gives the best balance between expressiveness and
  output stability?

## Recommended Research Priorities

### Priority 1: continue `center_parameterization` as the strong baseline

Reasons:

- best current quality
- strongest practical reference point
- best comparison anchor for any future family

### Priority 2: continue `raw_text_xyxy_pure_ce` as the high-upside research lane

Reasons:

- directly relevant to the `coord_token` necessity question
- meaningful recoverable recall gap remains
- continuity is already present without special coordinate tokens

### Priority 3: treat `base` and `hard_soft` as historical diagnostic references

Reasons:

- useful for understanding failure modes
- not currently compelling as forward product or research families

### Not prioritized in this archive: `cxcywh` and `cxcy_logw_logh`

These families were explored, but they are intentionally not part of the final
recommendation set here because:

- they did not beat the `center/raw_text` families
- and their archive-quality recall lane was not yet complete at the time of
  writing

## Final Decision-Level Summary

If the question is:

- "Do I still need `coord_token` just to get continuity?"

the answer is:

- **no, not based on current evidence**

If the question is:

- "What family should I trust most right now for strongest 2B performance?"

the answer is:

- **`center_parameterization`**

If the question is:

- "What family should I keep pushing if I want to learn whether raw text can
  replace coord-token assumptions while still improving quality?"

the answer is:

- **`raw_text_xyxy_pure_ce`**

This is the current best merged reading across both super-power studies.
