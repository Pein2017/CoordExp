## Context

Stage-2 Channel-B currently prepares rollout-aligned targets by accepting a
rollout prefix, editing the anchor target, appending false-negative GT objects,
and training one merged teacher-forced sequence. That path is reproducible, but
it does not directly model the autoregressive point where the rollout first
goes wrong. It also keeps pressure on sorted/tail object-order conventions that
are not part of the decoding grammar.

The new design treats each valid rollout as a self-prefix correction sample. At
the first actionable error, the builder computes the remaining object set,
constructs valid next-token actions, and compiles one local correction span into
the shared teacher-forcing IR. This keeps loss modules generic and moves
Stage-2-specific reasoning into a residual-state builder.

Constraints:

- YAML/config-first; no new stable CLI flags.
- Qwen3-VL template compatibility stays intact; `<|im_end|>` is STOP/EOS and
  `<|endoftext|>` remains padding.
- `do_resize=false` and geometry/coord order invariants remain unchanged.
- Upstream HF model files remain off-limits.
- Duplicate-specific training loss remains removed; duplicate-like behavior is
  handled through residual-set correction provenance and diagnostics only.

## Goals / Non-Goals

**Goals:**

- Define Stage-2 correction as residual-set valid-action likelihood at
  grammar-valid self-prefixes.
- Use one residual-state machine to produce token-level `ValidAction` records
  and their transition-validated `next_state`.
- Compile correction events into `SupervisionAtom` / `TeacherForcingTargetIR`
  with strict causal next-token alignment.
- Support repeated same-description objects whose first branch point may be a
  coordinate slot.
- Mine strict K-valid consensus UL clusters and use them as rollout-local
  TP-like positives with separate provenance, metrics, artifacts, and default
  loss weight `0.5`.
- Preserve hard-SFT and existing Stage-2 objectives as explicit ablation
  baselines rather than mutating their behavior silently.

**Non-Goals:**

- No full recursive autoregressive subtree marginal over alternative hidden
  states.
- No duplicate-burst unlikelihood, coord regularizer, bbox geometry auxiliary,
  or soft regression objective resurrection.
- No automatic mutation of dataset GT from online UL mining.
- No always-on visualization rendering from the training loop.
- No claim that smoke/val200 performance improvement is a spec validity gate.

## Decisions

### Residual State Machine

Use a `ResidualStateMachine` that owns both valid next-token enumeration and
state transition. Its core API returns token-level actions:

```text
valid_actions(state) -> list[ValidAction]

ValidAction:
    token_id
    token_role
    next_state
    candidate_subset_after
    selected_object_id optional
    object_provenance optional
    coord_slot optional
    action_tags optional
```

Rationale: this prevents separate valid-set and transition implementations from
drifting. Corrected roll-in samples an action, not a bare token id, so the next
state is transition-validated by construction.

Alternative rejected: construct `valid_token_ids` first and run an independent
transition filter later. That creates silent object-coordinate mixing risks.

Action coalescing rule: actions are unique by `token_id`. If multiple active
candidates share a next token, that one action carries the unioned
`candidate_subset_after` and does not set `selected_object_id` until the
transition actually becomes singleton.

### Correction Events And IR Layering

Stage-2 produces `CorrectionEvent` records that own rollout provenance,
residual state, UL/repeated/FP/FN labels, anchor choice, and artifact evidence.
Those events compile into `SupervisionAtom` records. Loss modules consume only
`TeacherForcingTargetIR`.

Rationale: loss code should not know how FN, FP, duplicate-like, or UL mining
was detected. It should only see token roles, valid token ids, selected token id,
positions, loss tags, and loss weight.

### Earliest Actionable Anchor

Each rollout contributes the earliest actionable correction event by sequence
position. Transition failures anchor immediately before the invalid token.
Object-level FP or repeated-object branches anchor before object start.

Rationale: this keeps Stage-2 correction precise and avoids training on a
post-error prefix that the decoder can no longer repair.

### Corrected Roll-In And Logits Alignment

Raw bad tokens are provenance only. The teacher-forced input sequence uses a
corrected selected action. For each atom:

```text
logit_position = target_position - 1
```

The selected token must match the corrected sequence at `target_position`.

Rationale: causal LM logits before the target token supervise that target. Using
the raw bad token or a hidden state after the target would make metrics
uninterpretable.

### Coordinate Local Span

Boundary, schema, and description corrections compile to one primary atom.
Coordinate corrections compile to `bbox_tail_from_anchor`:

```text
x1 -> x1,y1,x2,y2
y1 -> y1,x2,y2
x2 -> x2,y2
y2 -> y2
```

Within that span, each slot still follows the active-candidate rule:
valid-set marginal while ambiguous, hard CE after singleton commitment.

Rationale: bbox coordinates are a local structured unit. Repairing only the
first wrong coordinate would over-focus Stage-2 correction on `x1` and undercut
object coherence.

### UL Mining

UL mining runs before correction-event selection:

1. generate K rollouts;
2. keep K-valid grammar-valid eligible rollouts;
3. desc-gated match completed objects to labeled GT;
4. pre-deduplicate unmatched same-description objects within each rollout;
5. run strict complete-link cross-rollout clustering;
6. promote clusters only when `support_rollouts == K_valid` and
   `K_valid >= min_ul_valid_rollouts` (default 3).

Promoted UL members extend only the rollout-local universe:

```text
G*_k = labeled GT union ul_promoted_local(k)
```

Rationale: K-rollout consensus can prevent real unlabeled objects from being
trained as FP, but online mining must remain provenance-separated from dataset
GT and reviewable.

The residual objective owns its rollout count under
`residual_set_correction.config.num_rollouts`; it does not require legacy
`stage2_ab.channel_b.pseudo_positive.enabled=true`. Legacy duplicate-control
and insertion-order knobs may remain inherited as provenance or diagnostics, but
they do not prune residual correction events or order corrected targets.

Per-rollout same-description pre-dedup keeps the earliest object in rollout
order as the representative. Suppressed same-rollout duplicates do not vote for
UL and cannot become rollout-local UL positives.

Mixed labeled/UL ambiguous valid-action marginals use the union of valid tokens
as support. The scalar atom weight is `1.0` if any labeled candidate is
reachable, and `lambda_ul_promoted` only when the support is UL-only. Metrics
still report labeled-valid and UL-valid mass separately.

### STOP Handling

STOP is a token-level `ValidAction` using `TokenRole.STOP` and `<|im_end|>`.
STOP and continuation actions are mutually exclusive:

```text
R_t empty -> STOP
R_t non-empty -> continuation
```

Continuation margin defaults to `0.0` and is optional.

Rationale: the core target should not weaken remaining-object continuation by
placing STOP in the same valid set.

## Risks / Trade-offs

- [Risk] Corrected sequence/logit alignment is off by one.  
  Mitigation: IR validator and unit tests require
  `logit_position + 1 == target_position`, selected token equality, and prompt
  boundary checks before smoke metrics are trusted.

- [Risk] UL mining reinforces correlated model hallucinations.  
  Mitigation: ratio must be `1.0` over K-valid rollouts, complete-link all-pairs
  geometry must pass, one rollout contributes at most one vote, UL loss defaults
  to `0.5`, and every cluster is auditable through metrics/artifacts.

- [Risk] Duplicate-like bursts leak into UL positives.  
  Mitigation: per-rollout same-description pre-dedup suppresses burst members
  from UL voting; repeated-object boundaries use only positive residual-set
  correction and diagnostics.

- [Risk] New builder complexity obscures old baselines.  
  Mitigation: old hard-SFT/current Stage-2 objective paths remain explicit
  baselines; configs select the new residual-set path explicitly.

- [Risk] Artifact volume grows too large.  
  Mitigation: scalar counters are always available when enabled, but detailed
  `ul_clusters.jsonl` appears under the Stage-2 artifact root only when
  monitor/debug/smoke artifact flags are enabled.

## Migration Plan

1. Add config schema entries for the residual-set correction objective and UL
   mining under the Stage-2 YAML namespace.
2. Implement the residual-state machine and unit-test valid actions before
   connecting it to training.
3. Implement correction-event extraction, corrected roll-in, and event-to-IR
   compilation with validator coverage.
4. Add UL metrics/artifacts and STOP/continuation diagnostics.
5. Add smoke configs that compare hard SFT/current Stage-2 baselines against the
   new residual-set path without deleting baseline configs.
6. Update Stage-2 docs/runbook after implementation verifies the new config
   surface and artifact paths.

Rollback: keep baseline configs and objective modules selectable. Disable the
new residual-set objective by selecting hard-SFT or existing Stage-2 objective
pipeline entries.

## Open Questions

- Exact numeric default geometry thresholds for UL complete-link promotion
  should be finalized during implementation after checking current config
  conventions. The config names are fixed by the spec: IoU minimum,
  center-distance scale, area-ratio maximum, and aspect-ratio maximum.
- The exact module/file split may be refined by the super-power implementation
  roadmap and code review, but must preserve the event-to-IR layering and
  residual-state-machine contract.
