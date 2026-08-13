---
title: Human-13 On-Policy First-Bottleneck Successor
description: A bounded closed-loop same-panel comparison of full-row versus first-greedy-blocker compilation with direct owner-preserving rollback.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: planned
unit_id: 2026-08-13-human13-on-policy-first-bottleneck-successor
topic: qwen3-vl-dense-enumeration
status: authorized_planned
evidence_status: predecessor_complete_vertical_pending
updated: 2026-08-13
---

# Human-13 On-Policy First-Bottleneck Successor

## Decision

Run a bounded closed-loop successor on the exact Human-13 overfit panel.  The
controller observes the latest accepted natural greedy trajectory, selects a
K-native missing owner that composes best with the rest of that trajectory,
repairs either its complete row or only its first non-greedy token, and keeps
the update only if a fresh unconstrained clean-greedy decode preserves the
protected owner set.

The two treatments are:

- **O-Full-Safe:** current-state selected native full-row supervision plus the
  shared rectangle and dynamic duplicate terms, followed by behavior gate.
- **O-First-Safe:** the same selected row and shared safety terms, but the
  positive target acts only at the first HF token that is not strictly greedy-
  feasible, followed by the identical behavior gate.

The completed static A4/A6 and R1/R2 arms are historical references and are not
rerun.  There is no ungated treatment: the prior experiments already show that
net recall can hide owner exchange and that static duplicate loss can create
new downstream attractors.

## Question and strongest alternative

**Question:** Can a current-state, model-preferred native row be compiled into
greedy behavior with one small update while preserving all protected owners,
and is repairing only the first greedy blocker as effective as full-row CE?

**Strongest alternative explanation:** The selected row is not the true cause
of low recall.  The model may require distributed suffix changes, or any gain
may necessarily perturb prior owner trajectories.  In that case the direct
gate will reject most proposals and first-bottleneck and full-row arms will not
produce safe net conversions.

## Prior evidence

The predecessor panel is exactly 13 images and 392 trusted owners.  Frozen
Source has 173 greedy owners; the sealed K bank partitions the remainder into
73 K-hit owners and 146 K-miss owners.  The authoritative manifest SHA-256 is
`a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.

The completed static screen established:

- A4@2 reaches 181 unique owners (`H+8/G-5/M+5`) but 32 duplicates and 14
  malformed rows;
- row-contrast R1/R2@2 reach 183 (`H+9/G-6/M+7`) and reduce malformed rows to
  three, but produce 39--40 duplicates;
- the duplicate attractor is concentrated on image 14038 and occurs at newly
  reached prefixes absent from the static training ledger; and
- an aggregate Source-G gradient watch was compatible at both steps while six
  G owners were still lost, so it is not an owner-retention certificate.

Therefore more epochs on the same frozen states are not the next experiment.

## Frozen substrate

The successor reuses:

- the exact Source checkpoint, base model, language-only DoRA surface, frozen
  vision tower/aligner/embeddings, and fresh AdamW values from the predecessor;
- the sealed Source and K16 discovery artifacts and selected metric-valid
  native aliases;
- class-agnostic chronological prediction-to-prediction duplicate definition
  `IoU>0.95` before owner matching;
- cardinality-first, maximum-total-IoU one-to-one owner matching; and
- original-prompt HF fp32/SDPA batch-one clean greedy with `rp=1.0` as every
  decision-owning acceptance and final evaluation surface.

`M` is excluded from target construction, loss weighting, early stopping, and
checkpoint choice.  Incidental M gains remain reported.  Finite K-union does
not license positive EOS/STOP.

## Iteration state

Iteration zero reproduces Source exactly.  Each later iteration begins from the
last accepted checkpoint and exactly one clean-greedy panel output.  Its ledger
contains raw token IDs, parsed rows, owner matching, duplicate events, current
covered owners, protected owners, uncovered H owners, native aliases, score
evidence, and input hashes.

Protected owners initially equal Source `G`.  A newly gained H owner joins the
protected archive only after it survives one further accepted iteration.  This
prevents one transient appearance from permanently freezing a brittle branch.

One post-update decode is dual-purpose: it accepts/rejects update `k` and, only
if accepted, becomes the state for `k+1`.  The controller never performs two
updates from one stale natural ledger.

## Candidate construction

For every currently uncovered H owner, materialize each complete metric-valid
native alias after the current natural boundary.  Packed BF16/FA2 performs only
a cheap prefilter.  HF fp32/SDPA recomputes the survivor paths and defines

```text
D(r) = sum_q max(0, max_{v != r_q} z_q(v) - z_q(r_q)).
```

Keep the minimum-barrier alias per owner and at most four owners.  The old A8
maximum packed/HF drift (`1.4143`) requires HF decision ownership, but it is not
reused as a universal new-state margin.  Every trained site records aligned
packed/HF logits, actual argmax, ties, and candidate-specific drift.

For each shortlisted owner, force its complete row after the exact current
prefix, then release ordinary greedy continuation.  The cap is the larger of
twice the Source row count and Source generated-token count plus 512.  A cap
hit is harm.

Rank candidates lexicographically:

1. every protected owner is jointly coverable by an admissible one-to-one
   matching;
2. unique owner coverage increases;
3. continuation terminates naturally;
4. duplicate and malformed burdens do not increase, then row/token burden;
5. smaller HF barrier.

When canonical matching reports a protected loss, run a constrained matching
audit before rejecting the candidate.  This prevents equal-cardinality matcher
churn from masquerading as behavioral loss.

## Treatment objectives

Both arms use the same selected row, one optimizer update, owner/image
normalization, and current-state duplicate/rectangle evidence.

### O-Full-Safe

Apply mean CE to the selected native row body under its current exact prefix.
Mask terminal supervision.  The objective may choose a native coordinate alias;
it does not force canonical GT coordinates.

### O-First-Safe

Walk the selected HF path and find the earliest site at which the selected
token is not the actual strict global argmax.  Apply only

```text
relu(m_site + max_{v != target} z(v) - z(target))
```

at that site.  The selector is detached.  Do not supervise later suffix tokens.
If no blocker exists, the candidate needs no positive repair and the iteration
selects another owner or stops.

### Shared safety terms

- At selected-row `x2/y2`, accept any coordinate token producing positive width
  or height and reject the strongest invalid competitor.
- If the current natural path enters a duplicate row and a positive frontier
  exists, contrast the duplicate and selected row at their earliest owner-
  distinguishing divergence.  Do not penalize shared opener/description
  tokens.  Do not redirect toward STOP.
- Source/current-token floors are diagnostics or light regularizers only; they
  never replace the direct clean-greedy gate.

## Transactional behavior gate

Before the update, snapshot trainable DoRA parameters, the complete AdamW state,
scheduler/update counters, and torch/CUDA RNG state.  After the update, run one
fresh decision-surface panel decode.

Accept only if:

- every protected owner is jointly matchable;
- panel unique-owner count is nondecreasing;
- cap stops remain zero;
- malformed rows do not increase;
- rows are at most `1.2x` the prior accepted count; and
- duplicates increase by at most two.

On rejection, restore the complete snapshot and run a clean decode that must
reproduce the previous accepted owner set.  Candidate parameters and AdamW
moments from the rejected update must not influence later iterations.

## Contrast and budget

| Arm | Natural state | Positive dose | Behavior gate | Attempts |
| --- | --- | --- | --- | ---: |
| O-Full-Safe | refreshed accepted decode | complete selected row | transactional | <=8 |
| O-First-Safe | refreshed accepted decode | first HF blocker only | transactional | <=8 |

Optimizer update counts and selected-owner exposure are matched.  Decode and
candidate-search overhead are measured separately because forcing artificial
compute equality would waste the advantage being tested.

Initial K aliases are reused.  A K16 refresh (`batch_size=4`, `rp=1.10`) is
allowed only after two accepted iterations or support-bank exhaustion, and its
artifact must be versioned to the accepted checkpoint before use.

## Vertical gate

Before the pilot, one real O-First-Safe full-panel iteration must prove:

1. Source clean decode reproduces the frozen 173-owner identity set;
2. packed prefilter and HF shortlist receipts exist for two candidates;
3. both forced-row continuations terminate and are analyzed;
4. one packed forward/backward and AdamW proposal completes;
5. the post-update clean decode and accept verdict are bound; and
6. a deliberate rejection restores model plus optimizer state and reproduces
   the pre-update owner set.

Stop before pilot on wrong trainable surface, non-finite state, site/pack
mismatch, rollback mismatch, checkpoint/readback mismatch, cap harm, or an
unresolved real-entry error.  Do not patch around a scientific failure.

### 2026-08-13 real vertical evidence

The third fresh-root `O-First-Safe` force-reject run completed the entire
full-panel chain on one A100.  The first two fresh roots remain immutable
real-entry failures: projection had still validated only the legacy 73
`selected_rows` instead of all 309 metric-valid native aliases, then the live
payload receipt referenced the obsolete `logical_segments` field after the
optimizer update.  Both were reproduced by focused tests and repaired at their
source; neither is model-quality evidence.

The successful vertical restored 309 frontier aliases, packed/HF-scored 129
per-owner-capped candidates, released four natural forced continuations, and
selected `image 6040 / gt:6040:14 / seed 21016 row 009`.  Its forced branch
added that owner (`unique_owner_delta=+1`, HF barrier `1.0347023`) without a cap
hit.  One packed `O-First-Safe` AdamW update then changed clean greedy from
173 to 174 unique owners, but this was owner exchange: five owners gained and
four protected owners lost.  Duplicates rose from 9 to 17 and rows from 244 to
263; malformed rows fell from 11 to 1.  The direct behavior gate therefore
rejected the proposal for `protected_owner_loss` and `duplicate_burden` before
the predeclared forced-rejection reason was added.

Rollback reproduced all 173 Source owners, 9 duplicates, 11 malformed rows,
and 244 rows.  The complete model/AdamW/scheduler/counter/RNG state digest was
byte-identical before and after rollback:
`ad5a176df0417652dfe6ac068f4c79340924321242dfcf72158bc93b1f1071cb`.
No accepted checkpoint survived.  The immutable vertical summary is
`vertical/o-first-safe-force-reject-v3/vertical-summary.json`, SHA256
`64d688a07a5f3979339903d789777b85bef69c2cd3cd2dba03b91c41f8601a61`.

This passes the mechanics/rollback gate but is negative scientific evidence
for an unqualified one-site update: the selected owner was compiled, yet the
same update displaced four existing owners and created eight duplicates.  The
bounded two-arm pilot remains useful because `O-Full-Safe` can test whether
distributed row supervision changes that tradeoff, while both arms retain the
same direct gate.

## Decision-owning evidence and stop rules

The primary tuple is:

```text
(accepted attempts, rejected attempts,
 protected gained/lost, H gained, G lost, incidental M gained,
 unique owners, duplicates, malformed, cap stops, rows, tokens,
 candidate/forward/decode GPU seconds)
```

The route is promising if an arm safely compiles at least three net new owners,
loses zero original G owners, and accepts at least half its attempted updates.
These are interpretation aids, not a production promotion threshold.

Stop an arm early on rollback non-reproduction, any accepted cap hit, wrong
surface, non-finite state, no trusted frontier after its declared refresh, or
three consecutive rejected proposals with no new eligible candidate.  Stop
the experiment after eight attempts per arm regardless of outcome.

If at least 30 positive forced-continuation candidates across at least eight
images yield no safe accepted unique-owner gain, treat native local compilation
as falsified for this panel.  If O-Full-Safe and O-First-Safe are equivalent,
prefer the cheaper first-bottleneck route.  If only full-row succeeds, the
remaining suffix carries necessary distributed supervision.  If both gain but
are mostly rejected, gain-retention conflict remains the dominant problem.

## Claim boundary

This is adaptive, closed-loop, same-panel overfit control evidence.  It can show
whether the native model can safely compile already observed owner support into
these exact greedy trajectories.  It cannot establish K-miss support expansion,
generalization, validation gain, population prevalence, production safety,
full-set mastery, duplicate elimination, or architecture sufficiency.

Favor the shortest conclusion-changing implementation.  Do not introduce a
new generic trainer, evidence framework, controller service, or exhaustive
review layer when an experiment-local adapter over the existing Human-13 path
is sufficient.
