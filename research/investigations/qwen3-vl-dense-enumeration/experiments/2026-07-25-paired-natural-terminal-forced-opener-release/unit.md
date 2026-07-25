---
title: Paired Natural-Terminal Native versus Forced-Opener One-Row Release
description: Current-runtime causal comparison of native and canonical-opener-forced one-row releases at 200 Source natural stops with verified remaining owners.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-paired-natural-terminal-forced-opener-release
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_paired_causal_release
updated: 2026-07-25
---

# Paired Natural-Terminal Native versus Forced-Opener One-Row Release

Executed evidence and the bounded judgment are owned by
[the completed result](results.md).

## Decision and Primary Question

At each of the fixed 200 Source natural-terminal boundaries, does forcing only
the canonical new-row opener cause Source to produce a verified uncovered
physical owner that Source does not produce in a paired native one-row release
under the same current runtime?

This is a causal diagnostic of the immediate consequence of choosing
`continue`. It does not change the user prompt or the final `list all objects`
output contract, and it does not establish a production forced-continuation
policy or free-rollout final-set gain.

## Frozen Panel

The panel is exactly the 200
`untouched_terminal_with_remaining_owner` boundaries in:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality/panel-v1/manifest.json`

Every boundary is a distinct image, ends after a complete Source-generated
row, historically emitted the terminal token next, and has at least one
verified trusted physical owner still uncovered. The deterministic panel is
not a random sample of every Source stop, so all prevalence statements remain
panel-bounded.

## Exact Contrast

Both arms rebuild the same image and original `list all objects` prompt, append
the same literal Source-generated prefix, and use the same loaded Source model
session, parser, entity ledger, and owner matcher.

1. **Native arm:** greedily generate at most one row from the exact boundary.
2. **Forced-opener arm:** append only token `151646`
   (`<|object_ref_start|>`), then greedily generate the remainder of at most one
   row.

The forced token is not moved into the user prompt. It remains an assistant
continuation intervention after the identical historical assistant prefix.

The frozen execution settings are:

- Source checkpoint and authored HF inference config already used by the
  preceding owner-compositionality study;
- full-model 32-bit floating point runtime, scaled dot-product attention,
  physical batch size one;
- temperature `0`, top-p `1`, repetition penalty `1`;
- `max_new_tokens=64` and `malformed_limit=2` per arm; and
- eight stable boundary-ID shards, one model session per GPU.

## Primary and Secondary Outcomes

For boundary `i`, let `U_i` be the frozen set of verified owners uncovered at
the stop, `N_i` the native arm's strict matched owners, and `F_i` the
forced-opener arm's strict matched owners.

The primary boundary-level causal success is:

`((F_i intersect U_i) minus (N_i intersect U_i)) is nonempty`.

The primary report includes the success count and fraction, the number and
identity of causally gained verified owners, and a descriptive Wilson interval
within the frozen panel. It does not select one intended owner when several
owners remain.

Secondary outcomes include:

- native and forced recovery of any verified uncovered owner;
- paired gained, retained, and lost uncovered owners;
- strict covered-owner repeats;
- valid unmatched or ambiguous rows;
- invalid or incomplete rows and immediate terminal behavior;
- exact native-versus-forced row and owner-set agreement;
- results by Source margin, prefix depth, object-density, and remaining-owner
  count; and
- owner-category composition of causal gains and losses.

## Meaning-Bearing Invariants and Contract Gate

- The manifest hash, candidate-pool hash, image hash, rebuilt base-prompt token
  IDs, literal prefix token IDs, Source checkpoint identity, and authored
  inference-config hash must match their frozen receipts.
- Both arms must execute through the current HF model, canonical one-row
  generator, compact closed-row parser, and the same physical-owner matcher.
- `native` is re-executed under the same runtime; the historical terminal
  action is provenance and is not substituted as the control observation.
- Only the opener differs. No owner description, coordinate, covered-set text,
  alternate prompt, sampling, or checkpoint treatment is introduced.
- A forced valid row is not a true positive unless strict matching identifies
  a member of the frozen uncovered-owner set.
- A causal gain is not a free-rollout final-set gain; later rows, preservation,
  termination, and total trajectory utility are outside this unit.

## Smallest Real Smoke and Launch Gate

Before the full panel, execute at least two boundaries in one immutable smoke:

- one near the diagnostic continue/stop boundary; and
- one strong historical stop.

The smoke passes only if both arms share the exact image, prompt, prefix,
checkpoint, runtime, parser, and owner ledger; the native arm is not replaced
by the historical action; the forced arm records exactly one forced opener;
and both raw output and classified owner outcomes are attributable. Repeat one
case to verify deterministic output before full launch.

If the smoke exposes a semantic mismatch, hold the full run and repair only
the demonstrated seam. If it passes, run all eight shards. Shard coverage must
be exactly 200 unique boundary IDs with no overlap or omission.

The executed smoke passed on one near-threshold and one strong-stop boundary.
The near-threshold native and forced arms emitted the same covered-owner repeat,
while the strong-stop native arm terminated and its forced arm emitted a
covered-owner repeat. Repeating the near-threshold case reproduced both arms'
raw token IDs and strict owner matches exactly. Preflight also established that
the explicit frozen candidate-pool JSONL, rather than the authored config's
non-executed `data.input_jsonl` field, owns the runner's raw-example load; the
runner verifies that explicit path and hash against the manifest and records
both paths in each receipt.

## Artifact Root, Cost, and Stop

The immutable artifact root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-paired-natural-terminal-forced-opener-release/`

Expected products are smoke receipts, eight production shard receipts, a
200-case paired ledger, and one aggregate summary. The run loads eight copies
of the 2-billion-parameter Source model in 32-bit floating point and performs
two one-row releases per boundary; no optimizer state or training is involved.

Stop after the complete reduction, targeted audit of every causal gain/loss or
ambiguous outcome, durable bounded result, and user discussion. Do not launch
training, architecture work, a longer forced trajectory, or a checkpoint
comparison automatically.
