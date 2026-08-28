# Iterative Forced-Continue Extreme-Capacity Probe

## Question

When greedy dense enumeration reaches a natural `<|im_end|>` while verified
ground-truth owners remain uncovered, how much additional unique-owner coverage
can be recovered by repeatedly replacing that terminal decision with the
canonical `<object_ref_start>` token?

## Contrast

Run the same intervention over the human-refined 12-image cohort for the
sorted, random, and permutation-bundle checkpoints, each with greedy decoding
at repetition penalties 1.0 and 1.10.

## Intervention

- Generate the initial native completion with one uninterrupted standard greedy
  `generate()` call. Its token IDs must replay the existing native artifact
  exactly before the intervention is admitted.
- At a clean natural terminal with uncovered GT owners, record the terminal as
  the model's counterfactual top-1 action and execute one canonical row opener
  in its place.
- Greedily complete exactly one forced row, then release the model through one
  uninterrupted natural greedy segment until its next `<|im_end|>` or length
  boundary. Do not restart generation at each natural row.
- Preserve valid repeat and unmatched rows.
- Stop at full GT coverage, 3,084 executed completion tokens, or a malformed or
  incomplete terminal.
- A replaced terminal does not consume a completion position; its executed
  opener does.

GT coverage is recomputed after each uninterrupted natural segment and after
each forced complete row using raw cumulative predictions and
category-consistent, IoU >= 0.5, global one-to-one maximum-cardinality
assignment. A natural segment is never interrupted mid-generation based on GT;
GT only decides whether the next observed terminal is replaced. No
remaining-owner identity, category, description, or geometry is exposed to the
model.

The earlier rowwise-restart implementation and its six completed artifacts are
retained only as a mechanism control. They are not evidence for this probe:
restarting `generate()` after every row changed the native trajectory and failed
token-level replay against the established greedy artifacts.

## Primary observation

For every arm, report final and native unique-owner coverage, both immediate
forced-row gain and interval gain through the next natural boundary, both
zero-yield rates, token and force-count coverage curves, row error classes, and
the fraction of all-GT states whose unexecuted next-token probe is a natural
terminal.

## Claim boundary

This is a GT-aware extreme-capacity diagnostic. It is not native rollout
performance, GT-guided owner selection, or evidence that unrestricted
continuation is safe.

## Stop rule

The GPU probe is not admitted until a representative RP=1.0 replay matches the
existing native generated-token SHA exactly. After admission, the probe ends
after all six 12-image arms have immutable receipts, or earlier if a smoke
disproves token replacement, matching, or budget semantics. Any change to
matching, intervention token, row repair, or completion budget reopens the
research decision.
