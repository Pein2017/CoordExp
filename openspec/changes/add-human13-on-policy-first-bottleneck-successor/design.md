## Context

See [proposal.md](proposal.md) and the owning
[research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-on-policy-first-bottleneck-successor/unit.md).
The predecessor already owns sealed Source/K identities, native H rows,
no-padding multimodal packing, language-only DoRA assembly, world-size-one
AdamW, checkpoint publication, HF batch-one inference, and owner analysis.
Its result isolates the missing condition: static negative and target prefixes
do not cover states reached after an update, while aggregate gradient watches
do not certify discrete owner retention.

The old A8 census also measured a maximum packed-BF16 versus HF-fp32 margin
drift of `1.4143` on its frozen sites.  That is strong evidence against packed-
only scientific selection, but not a universal margin constant for new states.

## Goals / Non-Goals

**Goals:**

- Repair a current natural state rather than replaying only Source-era states.
- Distinguish full-row imitation from the minimal first greedy blocker while
  holding candidate, optimizer dose, and behavioral safety constant.
- Make owner retention an executable accept/rollback property.
- Reuse existing Human-13 and production-shaped model/runtime seams.

**Non-Goals:**

- K-miss support expansion, positive exhaustiveness supervision, RL return
  estimation, external routing/owner architecture, validation, or deployment.
- A general transactional trainer, generic search framework, online vLLM
  service, or arbitrary multi-update inner loop.
- Token-identical coordinates or constrained final evaluation.

## Decisions

### 1. The natural panel decode is the iteration state

One immutable `FrontierIteration` sidecar binds the accepted checkpoint,
original-prompt HF output, parser/matcher result, protected set, current
covered set, duplicate branches, candidate aliases, and all scores.  The next
sidecar is written only after a proposed update passes the behavior gate.

The initial native alias bank remains the sealed Source K16 bank.  A K refresh
is demand-triggered after bank exhaustion or two accepted updates and is a
separate batch-decode artifact (`batch_size=4`, `rp=1.10`); it never changes an
already-materialized iteration.  This preserves the inexpensive initial path
while allowing support to roll forward if the pilot actually accepts updates.

**Alternative rejected:** rebuild the entire K16 bank after every attempted
update.  It spends most compute on proposals before evidence that the
controller can accept even one update.

### 2. Packed scores prefilter; HF scores decide

The packed BF16/FA2 forward ranks cheap candidate rows and limits the HF work
to at most two aliases per owner.  HF fp32/SDPA then recomputes aligned causal
scores, global competitors, ties, and first bottlenecks.  Only HF results can
enter the shortlist or loss-site binding.  The receipt stores aligned vectors,
rank disagreements, and candidate-specific drift.

No global `1.5` margin is hard-coded.  The vertical slice measures current-site
drift; the training margin is a small predeclared task margin plus the observed
nonnegative aligned reserve for that candidate.  The final HF gate, not the
surrogate margin, owns success.

**Alternative rejected:** exact HF scoring for every K alias without packed
prefilter.  It is correct but unnecessary for a 2--4 owner shortlist.

### 3. Barrier then forced continuation selects a composable owner

For alias row tokens `y_1...y_L`, HF computes

```text
D(row) = sum_q relu(max_{v != y_q} z_q[v] - z_q[y_q])
```

and keeps the minimum-barrier alias per owner.  The two to four best owners are
forced after the exact natural pre-stop (or admitted boundary) prefix, then the
model resumes ordinary greedy decode.  A constrained one-to-one matcher tests
whether all protected owners can still be covered even if the canonical
maximum-IoU matching chooses a different equal-cardinality assignment.

Candidate quality is lexicographic: protected-coverable, positive unique-owner
delta, natural termination, duplicate/malformed/row burden, then barrier.
This prevents a locally easy row from winning when its downstream state loses
an existing owner.

**Alternative rejected:** choose the lowest row likelihood or largest gradient.
Neither predicts whether the row can compose with the rest of greedy decode.

### 4. Two safe arms isolate how much of the chosen row to train

Both arms share the selected alias and update dose.

- `O-Full-Safe` applies owner/body-normalized CE to the selected complete
  native row, excluding terminal supervision.
- `O-First-Safe` finds the first HF site where the target is not the actual
  strict argmax and applies one fp32 target-versus-global-competitor hinge.

Both retain the predecessor's rectangle-valid `x2/y2` gate on the positive
path.  If the current natural decode contains a duplicate row, an optional
pairwise term compares the duplicate and selected path at their earliest
owner-distinguishing divergence.  It never applies generic unlikelihood to a
shared opener or description and never invents `STOP` as the positive.

**Alternative rejected:** add remaining-set mass, full suffix CE, owner-level
gradient projection, and KL simultaneously.  They obscure whether the first
greedy blocker is sufficient and duplicate existing behavioral protection.

### 5. The update is an in-memory transaction

`IterationTransaction` captures trainable tensors, complete AdamW state,
scheduler/update counters, and torch/CUDA RNG state before the one-update
runner is called.  The existing checkpoint format is not used as an exact
optimizer-resume mechanism.  On acceptance, the ordinary experiment
checkpoint is published.  On rejection, the snapshot is restored and a clean
decode proves the prior owner set is reproduced.

The post-update decode has two roles only: decide the update and, if accepted,
seed the next iteration.  It cannot trigger a second update from the old
ledger.  Cheap Source/current-token floors are allowed as diagnostics but do
not replace the direct gate.

**Alternative rejected:** preserve parameters but leave AdamW moments changed.
That makes a rejected update influence later accepted iterations.

### 6. Execution is vertical-first and bounded

The real vertical uses one full-panel `O-First-Safe` iteration with two
shortlisted candidates, one proposal update, one post-update decode, and an
explicit forced rejection followed by rollback reproduction.  It binds wall
time, peak memory, packed tokens, HF score calls, decode calls, state hashes,
and artifact hashes.

After that gate, two independent world-size-one arms run at most eight
attempted iterations from Source.  A pilot stops an arm early on rollback
non-reproduction, non-finite state, wrong trainable surface, cap harm, no
trusted frontier after declared refresh, or repeated scientific rejection.
Optimizer update count and candidate token exposure are matched; decode
overhead is reported rather than artificially matched.

## Risks / Trade-offs

- **[Forced-row continuation is an intervention, not the deployed policy]** ->
  Use it only for candidate selection; accept and report only unconstrained
  clean-greedy output.
- **[Matching churn can masquerade as G loss]** -> Use a constrained
  protected-coverability audit before rejecting, while retaining canonical
  matching for reported metrics.
- **[First-bottleneck repair can expose a later blocker]** -> One update per
  ledger, then refresh from the next accepted natural decode.
- **[HF scoring and repeated decode are expensive]** -> Packed prefilter,
  shortlist at most four, batch K refresh only on demand, and report decode
  cost separately.
- **[In-memory rollback is large]** -> Snapshot only the language DoRA
  trainable surface plus optimizer/scheduler/RNG state; frozen model weights
  remain shared.
- **[Same-panel gate can overfit the control rule]** -> Label evidence as
  closed-loop same-panel control only and make no validation claim.

## Migration Plan

1. Add pure ledger, score, matching-audit, transaction, and loss helpers under
   CPU tests.
2. Extend the Human-13 live path behind two new arm IDs and dry-run receipts.
3. Execute the real vertical and rollback drill under a fresh immutable root.
4. If admitted, execute the two-arm, eight-attempt pilot under fresh roots.
5. Publish bounded results and stop; do not promote a checkpoint or alter
   production defaults.

Rollback of the implementation is deletion of successor-only scripts,
configs, tests, and loss helpers.  Historical Source/K, predecessor artifacts,
and production APIs remain unchanged.
