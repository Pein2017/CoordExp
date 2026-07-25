# Transition Mechanism Phase Zero Results

The existing-checkpoint Phase Zero closed all four authorized lanes without
new training. The formal result is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-25-existing-checkpoint-transition-mechanism-decomposition/results.md`

The original decision target did not change: one `list all objects` prompt,
one free autoregressive completion, and final unique trusted physical-owner
coverage with Source retention, valid rows, useful geometry, low duplication,
and natural stopping.

## What changed in the interpretation

Transition step 36 is not explained by a pure stop-gate shift. At seven exact
Source-produced prefixes, its continue-minus-stop margin rises in all seven
cases. After the canonical row opener is fixed, the best verified uncovered-
owner row improves relative to the best plausible covered-owner row in five of
six comparable cases. This is bounded post-continue conditional row-ranking
evidence.

The stronger interpretation is not supported. Neither of the two actual
Source terminal states flips to continue. Released generation does not improve:
with a complete one-token description, Source realizes any uncovered owner in
four of six cases and the intended owner in three, while transition step 36
does so in three and two. Independently scored candidate rows are not a
normalized candidate distribution and do not prove an explicit covered-set
state.

Forcing only the row opener can recover a real missed object rather than a
repeat. Source recovered a previously uncovered bottle on held-out image 3442;
the other current terminal case and both transition releases did not recover a
strict owner. This establishes existence, not a recovery probability.

## Robust final-owner evidence

The held-out 128-image comparison remains `46 gained / 39 lost / +7 net` with
118 fewer predictions and 13 fewer strict duplicate candidates. Net direction
stays positive across token cutoffs, natural-stop restriction, matching
thresholds, and leave-one-image-out analysis. Per-image median owner net remains
zero, so this is a minority-image directional aggregate rather than a typical-
image improvement.

Arm-blinded review of all 85 changed owner references preserves the official
geometry ledger and yields a strict human-refined count of 22 genuine gains,
16 genuine losses, and `+6`. Review judgment does not replace ground truth, and
unmatched predictions are not automatically hallucinations.

## Complete-row objective diagnosis

The common exact-event projection contains 1,440 rows. Sequence sum, target-
token mean, equal description/schema-versus-coordinate group weighting, and
every 9/10/11-token stratum preserve the same likelihood ordering. The poor
full-row behavior is therefore not primarily a sum-versus-mean normalization
artifact.

Pairwise and owner-conditioned arms already gain owners by 256 generated
tokens. After that they add many predictions without further net owner gain.
At 3,084 tokens pairwise has 1,396 predictions and 0.290 owner yield; owner-
conditioned has 3,235 predictions and 0.126 yield, versus Source at 786 and
0.495. Length stops concentrate the worst tails, but paired natural-stop images
also expand. Their dominant failure is excessive continuation and output
expansion. Owner signal is present but insufficiently selective.

## Decision and next discussion

The unit disposition is `post-continue-improvement`, qualified by a concurrent
continuation shift and no row-realization advantage. No architecture, state
carrier, prompt change, one-row-at-a-time policy, objective recipe, or new
training is promoted.

A future 256-image training cohort may be mixed. Verified prefix events can
carry uncovered-versus-covered conditional owner signal; other images can
carry Source preservation, ordinary row realization, stopping controls, and
matched exposure. The role of every event must remain explicit, and non-signal
images cannot be claimed as equivalent set-expansion supervision.

If the user authorizes a new unit, the highest-value matched screen separates:

1. modest continue-versus-stop calibration;
2. conditional uncovered-versus-covered row preference after the opener is
   fixed;
3. token-level row realization; and
4. Source-policy preservation.

Use fixed 256- and 512-token free rollouts as an early gate. Report gained,
retained, and lost owners, owner gain per prediction, duplicate and invalid
rows, geometry, and length stops. Do not turn owner exchange into automatic
negative supervision. Stop now for user discussion before any optimizer
update.
