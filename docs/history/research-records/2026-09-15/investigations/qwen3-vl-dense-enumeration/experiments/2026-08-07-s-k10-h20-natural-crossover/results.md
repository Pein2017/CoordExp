---
title: S K10-H20 Natural Crossover Result
type: investigation-result
status: complete_source_specific_unqualified
updated: 2026-08-07
---

# S K10-H20 Natural Crossover Result

## Disposition

This no-training unit executed exactly once and completed. Its formal evidence
is artifact-valid: every identity, endpoint, and contrast reconstructs from the
raw shard results. Its scientific disposition is
`source_specific_crossover_status = unqualified`, and formal aggregate `tau`
and both utility horizons are **null, not zero**.

The unit is conditional case-level selected-three-event evidence. It is not
checkpoint-level crossover prevalence, not an effect size, and not evidence for
any training, architecture, wrapper, token, decoder, checkpoint, or production
change. S four-coordinate `geo_sorted_xy` step-2444 remains the sole
decision-owning substrate.

## Formal identities

Formal root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-07-s-k10-h20-natural-crossover/evidence-v4`

| Artifact | Raw SHA-256 | Semantic self SHA-256 |
| --- | --- | --- |
| `pre-gpu-receipt-v5/pre-gpu-receipt.json` | `7e4adff6e272dfaad8ad5bb5656c9fc0baf7f2992d3561e37edfb613991e57ec` | `c349258554a2e308c5a4242157eb59cbb6793cdc7476c2a3a65719b845ae4571` |
| `finalization-receipt-v1/finalization-receipt.json` | `71b305dab1b25f838ba65433902cafe91e5aaa26ed80fbab792f607b1257c15e` | `c0244fb8e35e5fa4e277972085e5dc273ac6a7cda539d96ee64ce966fe5ee3ed` |
| `evidence-v4/evidence.json` | `29407f7cddd632999e3f720e9982fc4a52a31eea5398181f2f9d07dd28947d90` | `1dfa4144d094aa06f35c1cc3fb7f29b2db10efc55367f65b96c7244695d990f3` |
| `evidence-v4/evidence.receipt.json` | `9fc2fbb155bd3158f26d9085072d74e5c48b3c6dfe7496ebb4774296f359c398` | `db5649f7a9628f19b0445511d6e138be31f8dc8c04567080a81de0f230d08a85` |

Frozen source evidence, unchanged from the unit contract:

| Artifact | Raw SHA-256 | Semantic self SHA-256 |
| --- | --- | --- |
| `s-k-n-h-evidence-native-fn-supersession-v3/evidence.json` | `77f90bc48f8126ee1b071767308db603e1b64c0d19a4ca3e5ad8906cfc837598` | `e04cb63540f9e3a1b27b938ada14f694f6b7208564adcaf2c82656915c62bb95` |
| `s-k-n-h-evidence-native-fn-supersession-v3/evidence.receipt.json` | `fa11800ec7ad832646162f77c9e5a6df71d2e9ca951144e40b2fd060df46bf8e` | `80aa70f46899474ee1f8646c02d9c061babfac9bd58534f4cfa1d75aa9df12fd` |

The preGPU-v5 receipt and `execution-v4`/`evidence-v4` are consumed and
completed. The preGPU-v1 through v4 trees and the `execution-v3` failure root
remain immutable technical history and are not reinterpreted here.

## Selection and what the K10 3/3 figure means

The planner recomputed, rather than copied, the intersection
`K10.target_strict_release ∧ H20.factor_qualified ∧ ¬H20.degenerate_grammar_disruption`
over the frozen 11-event source cohort and obtained exactly `gt:2299:29`
(index 2, image 2299), `gt:13348:14` (index 5, image 13348), and `gt:16228:15`
(index 8, image 16228), in that order.

The fresh implementation reproduced all nine source `K01`/`K10`/`H20` endpoint
vectors exactly, `9/9`, across the three events.

Therefore **K10 target release `3/3` in this unit is a deterministic
cross-implementation replication of the selection condition, not an efficacy
result and not a base rate.** The events exist because K10 released the target
in the source run. The `9/9` reproduction is a strong determinism receipt for a
fresh runner, actuator seam, and finalizer; it carries no information about how
often K10 releases a target.

## Formal three-event endpoints

Frozen cells: `C00` = K01 explicit all-allowed causal no-op with hidden K00
full-vocabulary parity; `C10` = K10 hard B-exclusive oracle image-key routing;
`C01` = H20 all-layer natural-boundary history operator; `C11` = simultaneous
`K10 ∧ H20`. Admission is `pre_opener_natural`, `opener_injected=false`,
`use_cache=false`, `max_rows=3`, `max_row_tokens=256`. Opener token is
`151646 <|object_ref_start|>`.

### `gt:2299:29` — image 2299, shard-000, physical GPU 0

Target `gt:2299:29`; covered `{gt:2299:30}`; admitted history is one row
(9 tokens); prefix 1231 tokens ending on `151649 <|box_end|>`.

| Cell | Admission | Complete rows | Strict (emission order) | Target release | Unmatched | Invalid | Duplicates | STOP | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | opener, rank 1, logp `-0.00011` | 3 | 3 — `[16, 7, 38]` | no | 0 | 0 | 0 | 0 | matched |
| C10 | opener, rank 1, logp `-0.00008` | 3 | 1 — `[29]` at row 0, IoU `0.669` | **yes** | 2 | 0 | 0 | 0 | unmatched |
| C01 | opener, rank 1, logp `-0.41736` | 3 | 3 — `[30, 29, 16]`, `30` is a covered repeat | **yes** at row 1, IoU `0.826` | 0 | 0 | 0 | 0 | matched |
| C11 | **first token `291` (`'ed'`); opener rank 4, logp `-6.34561`** | 0 | 0 | no | 0 | **1** | 0 | 0 | invalid_token_grammar |

C11 ran one forward, set `row_started=false`, retained
`terminal_reason=invalid` with `reason=first_token_not_opener`, and remains
`mechanically_valid=true`. Per the frozen contract this is a scientific
outcome, not technical invalidity.

### `gt:13348:14` — image 13348, shard-001, physical GPU 1

Target `gt:13348:14`; covered `{gt:13348:0}`; admitted history is one row
(10 tokens); prefix 1372 tokens.

| Cell | Admission | Complete rows | Strict | Target release | Unmatched | Invalid | Duplicates | STOP | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | opener, rank 1, logp `-0.00403` | 3 | 1 — `[13]` | no | 2 | 0 | 0 | 0 | unmatched |
| C10 | opener, rank 1, logp `-0.00894` | 3 | 1 — `[14]` at row 0, IoU `0.727` | **yes** | 2 | 0 | 0 | 0 | unmatched |
| C01 | opener, rank 1, logp `-0.01653` | 3 | 2 — `[1, 13]` | no | 1 | 0 | 0 | 0 | unmatched |
| C11 | opener, rank 1, logp `-0.00568` | 3 | 1 — `[14]` at row 0, IoU `0.563` | **yes** | 2 | 0 | 0 | 0 | unmatched |

C11's ten-component endpoint vector is identical to C10's. The raw trajectory
differs (different coordinate bins and IoU), so this is endpoint identity, not
trajectory identity.

### `gt:16228:15` — image 16228, shard-002, physical GPU 7

Target `gt:16228:15`; covered `{gt:16228:42}`; admitted history is **two rows**
(18 tokens); prefix 1354 tokens.

| Cell | Admission | Complete rows | Strict | Target release | Unmatched | Invalid | Duplicates | STOP | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | opener, rank 1, logp `-0.00006` | 3 | 2 — `[35, 36]` | no | 1 | 0 | 0 | 0 | unmatched |
| C10 | opener, rank 1, logp `-0.00007` | 3 | 1 — `[15]` at row 1, IoU `0.782` | **yes** | 2 | 0 | 0 | 0 | unmatched |
| C01 | opener, rank 1, logp `-0.00607` | 3 | 1 — `[36]` | no | 2 | 0 | 0 | 0 | unmatched |
| C11 | opener, rank 1, logp `-0.01691` | 3 | 1 — `[15]` at row 0, IoU `0.772` | **yes** | 2 | 0 | **0 reported** | 0 | unmatched |

C11's endpoint vector is again identical to C10's.

**Byte-identical repeated rows with `duplicates = 0`.** C11 rows 1 and 2 have
identical token identifiers
`(151646, 8987, 151647, 151648, 151683, 152112, 151703, 152138, 151649)`,
giving the same box `[16, 368, 40, 389]` and the same coordinate bins
`(13, 442, 33, 468)`. The machine reports `duplicate_count = 0`. This is
correct under the field's definition and is not an artifact defect; see field
semantics below. The artifacts are not rewritten.

### Componentwise contrasts

| Contrast | `gt:2299:29` | `gt:13348:14` | `gt:16228:15` |
| --- | --- | --- | --- |
| C10−C00 static | strict `-2`, release `+1`, unmatched `+2` | strict `0`, release `+1`, unmatched `0` | strict `-1`, release `+1`, unmatched `+1` |
| C01−C00 history | strict `0`, release `+1`, unmatched `0` | strict `+1`, release `0`, unmatched `-1` | strict `-1`, release `0`, unmatched `+1` |
| C11−C10 history given K | admission `-3`, complete `-3`, strict `-1`, release `-1`, unmatched `-2`, invalid `+1` | **all zero** | **all zero** |
| C11−C01 static given H | admission `-3`, complete `-3`, strict `-3`, release `-1`, invalid `+1` | strict `-1`, release `+1`, unmatched `+1` | release `+1` |

### Hidden K00 technical parity

At every event the K00 full-vocabulary arm runs with no attention actuator and
C00 runs the explicit all-allowed K01 mask. Measured parity is `27/27`
candidate steps with `per_forward_max_abs_delta = 0.0` against a `1e-4`
tolerance, in all three shards. The explicit no-op mask is numerically inert.

## Endpoint field semantics

These definitions are load-bearing for every number above and must be quoted
with the numbers.

- `duplicates` counts **strict physical-owner identity repeats only**:
  `duplicate = strict_physical_owner_match ∧ owner ∈ already_emitted_owners`.
  A row that matches no physical owner can never be a duplicate under this
  field, even when its emitted tokens are byte-identical to a previous row.
  `duplicates = 0` therefore means "no owner was strictly re-emitted"; it does
  **not** mean "no row content repeated". The `gt:16228:15` C11 pair above is
  the concrete case.
- `target_release` is `target_owner_id ∈ strict_owner_sequence`. It is a
  release indicator. It is not a retention indicator, and "C11 retains the
  target" is the wrong reading of `target_release = true`.
- `unmatched` counts parser-valid rows whose box matches no physical owner
  under source-specific matching. It is a scientific outcome, never silently
  converted to technical invalidity.
- `complete_rows` is `parse.valid_rows`, not the emitted row count. The
  `gt:2299:29` C11 cell emitted one invalid row and reports
  `complete_rows = 0`, `invalid = 1`.
- `row_admission` counts rows whose opener was generated by the model.
  `opener_generated_by_model` is true if and only if the first generated token
  is the opener.
- `STOP` is native terminal emission. It is `0` in every cell of this unit.
- `mechanically_valid` remains `true` for the `gt:2299:29` C11 cell; grammar
  collapse at the first token is an outcome, not an execution defect.

## Why formal tau and utilities are null, not zero

A cell is source-specific qualified only when it is mechanically valid, has
`strict_count > 0`, has `strict_count == complete_rows`, and has zero
`unmatched`, `duplicates`, `ambiguous`, `malformed`, `invalid`, and `STOP`.
Unqualified cells are `{C10, C11}` at `gt:2299:29` and **all four cells** at
`gt:13348:14` and `gt:16228:15`. Aggregate tau requires all three events
qualified, so `qualified_event_count = 0` on every component and every mean is
null.

Null is the correct value for three independent reasons:

1. **Contract.** The qualification predicate is unmet, so no source-specific
   denominator exists.
2. **Semantics.** An unmatched row has no physical owner. A delta in
   `strict_count` across cells therefore mixes "routed to a different owner"
   with "emitted a box matching nothing". Averaging that into a tau would
   silently promote a mixture of routing change and grounding failure into one
   causal estimate.
3. **Arithmetic.** Zero would also be factually wrong. `C10−C00 target_release`
   is `+1` at all three events, and `C11−C10 complete_rows` is `[-3, 0, 0]`.
   The componentwise deltas are published under `component_contrasts` and are
   descriptive only, under
   `tau_scope = "source-qualified endpoint arithmetic only; no hidden-state interaction claim"`.

## Static carrier claim

Hard B-exclusive K10 routing is **oracle routing and target transcription
sufficiency only**. Under the exact natural pre-opener boundary it is
sufficient to make S emit one row whose box strictly matches the physical
target owner, on the selected events. It does not carry general owner
grounding:

- every K10 cell produces exactly `2` unmatched rows, at all three events;
- matched strict count falls relative to its own baseline at `2/3` events
  (`3 → 1` and `2 → 1`);
- horizon-3 owner utility at `gt:2299:29` is net `-2`, charged `-4`;
- the unmatched K10-family rows are near-identical boxes clustered on the
  target region, not dispersed errors. At `gt:13348:14` and `gt:16228:15` they
  are also small (roughly `25×25` and `21×21` pixels); at `gt:2299:29` they are
  not small (`[168,393,246,682]` and `[170,491,253,735]`), so the "small box"
  reading is scoped to the two later events only.

This must not be described as a natural owner slot, an owner field, a covered-
set pointer, or a trainable soft routing mechanism. It is a hard attention
restriction supplied from ground truth.

## Natural history and admission claim

H20 ablates the latest completed row's key positions at every layer.

- It degrades opener confidence at all three events and flips the opener at
  none: opener log-probability moves from `-0.00011`, `-0.00403`, `-0.00006`
  under C00 to `-0.41736`, `-0.01653`, `-0.00607` under C01.
- Its owner effect varies in sign: target release `+1` at `gt:2299:29`,
  strict `+1` at `gt:13348:14`, strict `-1` at `gt:16228:15`. It never
  releases the target at the latter two events.
- **Recency attribution is confounded at 2 of 3 events.** H20 blocked `9`,
  `10`, and `9` key positions against admitted histories of `9`, `10`, and
  `18` tokens. At `gt:2299:29` and `gt:13348:14` the latest row *is* the entire
  admitted history, so H20 is indistinguishable from full-history ablation
  there. Recency-only attribution rests solely on `gt:16228:15`, which is also
  the event where H20 alone lost an owner.

## The single-event admission interaction

`C11 − C10` is identically zero on all ten components at `gt:13348:14` and
`gt:16228:15`. The only place the crossover does anything is `gt:2299:29`, and
there it acts on admission rather than routing: the opener falls to rank 4 with
log-probability `-6.34561` and the argmax becomes `291` (`'ed'`), so no row
starts.

An audit-derived diagnostic, descriptive only and not part of the formal
endpoint contract, quantifies this. Interaction on the opener in log-probability
space, `C11 − [C00 + (C10−C00) + (C01−C00)]`:

| Event | C00 | C10 | C01 | C11 | additive prediction | interaction |
| --- | --- | --- | --- | --- | --- | --- |
| `gt:2299:29` | `-0.00011` | `-0.00008` | `-0.41736` | `-6.34561` | `-0.41734` | `-5.93` nats |
| `gt:13348:14` | `-0.00403` | `-0.00894` | `-0.01653` | `-0.00568` | `-0.02145` | `+0.016` |
| `gt:16228:15` | `-0.00006` | `-0.00007` | `-0.00607` | `-0.01691` | `-0.00608` | `-0.011` |

This is a **single-event admission interaction at `n=1`**. It is not a
checkpoint-level history route, not a general grammar failure mode, and not
evidence that `K10 ∧ H20` composes destructively in general.

## Post-opener claim boundary

Every cell here is measured from a natural pre-opener boundary with no seeded
opener, so the first generated token is itself an outcome. The `gt:2299:29` C11
result occurs *at* the opener step. A seeded post-opener protocol, including the
2026-08-05 A3 `gt:2299:2` result recorded as post-opener conditional
oracle-routing sufficiency, structurally cannot produce or exclude it, because
the opener is supplied rather than sampled.

No claim transfers in either direction. This unit neither corroborates nor
contradicts the A3 post-opener result, and the A3 result cannot explain away
the `gt:2299:29` collapse.

## Case-level versus checkpoint-level boundary

Denominators are `3` events and `3` images. The frozen source cohort separates
case-level from checkpoint-level status with a checkpoint floor of at least `3`
events and `2` images.

This unit does **not** reach checkpoint level for the crossover, and event count
is not the reason. The three events were selected conditional on the source
arms' own outcomes, so the design is conditional case-level by construction. No
prevalence, base rate, or effect size for `K10 ∧ H20` follows from it, whatever
its cell counts. The evidence records this as
`claim_scope = "case-level selected-three-event evidence only"`.

## Retrospective design and power failure

A non-null aggregate tau was **structurally impossible before the launch**, and
this was computable from the sealed plan alone.

Aggregate tau requires all three events qualified, and the qualification
predicate requires zero unmatched rows. Plan-v1's own frozen
`source_endpoint_vectors` already recorded `K01` with `unmatched_rows = 2` at
`gt:13348:14` and `= 1` at `gt:16228:15`, and `K10` with `unmatched_rows = 2` at
all three events. Because C00 replicates K01 deterministically, and it did,
`gt:13348:14` and `gt:16228:15` could never qualify regardless of what the
operators produced.

This is a retrospective design and power failure of the selection rule, which
conditioned on source-arm outcomes without requiring baseline qualifiability.
It is recorded so the same shape is not repeated. **No retry, re-selection,
repair launch, or new crossover is authorized or proposed.** The executed
evidence stands as-is.

## Provenance seam: runtime identity is not executed-GPU provenance

Each `execution-v4/shard-NNN/runtime_identity.json` carries
`status: "completed"` alongside a `runtime` block with `gpu_used: false` and
`model_loaded: false`. That block is the **sealed CPU-only preGPU-v5 runtime
identity, passed through verbatim**. It is a pre-launch contract, not an
attestation about the executed run.

Executed-GPU provenance for this unit comes from `logs-v6` (which records
checkpoint-shard loading and the emitted result hash per shard), the per-shard
`result.json`, `terminal_summary.json`, and `aggregate.receipt.json`. Any
downstream consumer must not read the `runtime_identity.json` `runtime` block as
evidence that no GPU or model was used. This seam is documented, not repaired;
no artifact is rewritten.

## Strongest alternative

**Hard-oracle target selection with regional grounding collapse, plus
event-specific grammar and admission disruption — not separable history
routing.** Every C10 cell carries exactly two unmatched near-duplicate boxes
clustered on the target region. At the two events where C11 admits rows, its
endpoint vector is identical to C10 and `C11 − C10` is zero on all ten
components; at the third event, the divergence is an admission failure rather
than a routing difference.

A second alternative stays open: **the owner-matching instrument, not the
operators, may be the limiting factor.** The untreated C00 baseline is
`unmatched` at 2 of 3 events. Until the unmatched boxes are shown to be
genuinely unsupported rather than matcher failures, no crossover contrast on
this event family can be qualified.

## Decision table: realized outcome to next discriminator

| Realized outcome | Cells | Routed discriminator |
| --- | --- | --- |
| `unmatched` | **9 of 12** — C10 ×3, C11 at `gt:13348:14` and `gt:16228:15`, C01 at both, **C00 at both** | Owner-matching and grounding validity on existing artifacts: are the unmatched boxes genuinely unsupported, or matcher failures? |
| `invalid_token_grammar` | 1 — C11 at `gt:2299:29` | Admission and token diagnosis at the natural boundary before any endpoint comparison. |
| `matched` | 2 — C00 and C01 at `gt:2299:29` | Bounded component contrasts only, within the case-level scope. |
| `duplicate` | 0 reported; 1 true row-content repeat exists | Not fired, by the field definition above. |
| `native_STOP` | 0 | Not fired. |

The dominant branch is `unmatched`, and it fires on the untreated control. The
routed discriminator is therefore an **owner-matching and grounding validity
check on existing artifacts**. It is CPU-only in principle and requires no new
model execution, but it is **not authorized and not executed here**; it is a
later user-owned decision.

## Decisions

- **Training: HOLD.** No training route is proposed or recommended. There is no
  qualified estimand a training objective could target: the crossover produced
  no separable routing effect, aggregate tau was structurally null before
  launch, the only interaction is `n=1` in the admission channel, and the
  control arm fails owner matching at 2 of 3 events.
- **A3 secondary contrast: DO NOT RUN.** A3 is admissible only if S raises a
  concrete commit-wrapper-specific attribution question. It does not. S's
  `row_contract` has `commit_token_id: null`; this wrapper has no commit token.
  The `gt:2299:29` failure is an argmax displacement at
  `151646 <|object_ref_start|>`, a special token common to both configurations.
- **P4: DO NOT RUN.** `authorize_P4 = false` in plan, receipt, and evidence;
  `no_p4 = true` in the frozen execution policy. Nothing in the result creates
  a P4 predicate.
- **No sweep.** `no_sweep = true` is frozen in plan-v1 and preGPU-v5. More cells
  of this shape cannot qualify while the control is unmatched.
- **No promotion.** No architecture, objective, decoder, wrapper, token, or
  checkpoint promotion follows from this unit.

## Lineage

The `logs-v1` through `logs-v5` attempts and the `execution-v3` failure root
remain immutable technical history: zero-byte transport artifacts, pre-model
`h0_root` ordering failures, manifest/plan projection failures, and the
`GateTechnicalInvalid: … C11 callback requires scalar input_ids` gate failure
carry no scientific endpoint and are not reinterpreted. The independent audit
record is in [review.md](review.md).
