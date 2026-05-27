## Context

Stage-2 should correct the model on states it actually visits during rollout,
without relying on sorted order, edited-anchor targets, duplicate-specific
unlikelihood, or coordinate repair. The intended first version is deliberately
offline and replayable:

```text
checkpoint -> K rollout_attempts per sample -> prepared JSONL
prepared JSONL -> target IR -> training
```

This is DAgger-like because each round trains on self-prefix states generated
by the current or recent policy, but v1 does not generate inside the training
loop.

## Key Decisions

### Reuse Existing OpenSpec Change

The existing `add-stage2-residual-set-ul-correction` change is reused and
rewritten. Older Stage-1 trie-marginal artifacts and older residual-set drafts
are treated as superseded background, not as normative contract.

### Trie / Multiple-Positive Equivalence Correction

Stage-2 trie and residual-set correction are the same semantic object: a
residual-state dynamic valid set over the actions still legal after one
rollout attempt's own prefix. The public module names `stage2_trie_ce` and
`residual_set_correction` are compatibility-facing aliases for that object.

The earlier repaired-candidate plan that merged alternatives around one
privileged segment or rollout ordinal is a semantic drift and is superseded. A
single rollout may not define the coordinate system for multiple-positive
supervision. Each retained rollout attempt is an independent self-prefix
training sequence; multiple-positive means the current residual state exposes
multiple valid next actions, not that other rollouts are forced to share one
rollout's token prefix.

### Prepared Rollout Records

New prepared rollout JSONL records must provide:

```text
response_token_ids
raw decoded text
decode_mode
sampling_seed optional
generation_config_hash
sample/image provenance
```

Residual trie configs name this file at:

```text
stage2_ab.pipeline.objective[name=residual_set_correction|stage2_trie_ce].config.prepared_rollout_jsonl
```

`response_token_ids` are required for new data. Raw text is diagnostic and
review surface, not the canonical training prefix. Missing token ids should
drop the sample in strict mode; explicit legacy fallback may re-encode raw text
for smoke/legacy ablation only and must record `dirty_prefix_reencoded=1`.

Absolute rollout-time assistant spans are not required. Training assembly
reconstructs prompt plus assistant context in the current tokenizer/processor
environment and validates assistant span, logit positions, and target positions
there.

### Rollout Attempts

Default attempt generation:

```text
K = 4
attempts:
    1 greedy rollout_attempt
    3 sampling rollout_attempts
```

After generation all attempts are equal. `greedy` is decode metadata, not an
anchor. Exact duplicate attempts are deduplicated by `response_token_ids`, or by
exact raw text only when token ids are unavailable in an explicit legacy path.
Approximate duplicates are kept.

### Template Boundary Adapter

Target assembly must not hand-author schema strings. It must use a
Stage-1-compatible template adapter:

```text
TemplateBoundaryAdapter:
    resolve Stage-1 assistant template / serialization policy through
        src/detection/template.py::get_detection_template
    render assistant text for chosen supervision objects
        via RenderedAssistantSequence / object_entries / separator_spans
    tokenize with the active tokenizer / processor contract
        via src/detection/tokenization.py::tokenize_rendered_detection_conversation
    expose object, separator, terminal, and stop spans
    slice suffix from a requested boundary state
    validate no deterministic schema token is duplicated
```

For the checkpoint-compatible v1 path, newline/separator/EOS placement follows
the resolved Stage-1 template. If the resolved template renders newline at a
position, newline is a deterministic schema/control token there, not an optional
alternative. Future no-newline migration is a separate change.

### Shared Target IR

The runtime-facing atom uses canonical `logit_position`:

```text
SupervisionAtom:
    logit_position
    target_position optional
    token_type
    role
    valid_token_ids
    selected_token_id optional
    weight
    target_kind
    provenance
```

`target_position` exists for validation/debug. If present:

```text
logit_position + 1 == target_position
```

Loss modules consume `logit_position`, not a separate `position_index` field.
Older prose may use `position_index` as a synonym only.

In the current shared IR, residual-set atoms map onto
`src/training/teacher_forcing/ir.py::SupervisionAtom`, where
`target_position`, `selected_token_id`, `allowed_token_roles`,
`selected_token_role`, `loss_weight`, `coord_role`, and `loss_tags` are required
or first-class fields. A future IR migration may simplify the conceptual atom,
but v1 must satisfy the current validator.

### Type Loss And Inner Loss

Token-type exclusivity is standalone and enabled by default:

```text
lambda_type = 1.0
lambda_inner = 1.0
```

The type module globally supervises schema/text/coord mass. Existing registry
names may be `struct` for schema/control and `desc` for free-text description;
the residual-set implementation must define this mapping explicitly rather than
mixing names ad hoc. The inner objective uses valid-set marginal likelihood:

```text
L_inner = -log sum_{v in valid_token_ids} p(v)
```

Hard CE, deterministic schema tokens, newline when template-rendered, and EOS
are singleton valid-set cases.

`valid_token_ids` must come from `ValidAction` records returned by the
residual-state machine. When corrected roll-in selects one action, that action's
`next_state` drives later suffix and atom construction.

### Sequence Granularity And Normalization

One retained `rollout_attempt` becomes one training sequence. The sequence may
contain all eligible non-conflicting correction atoms. Atoms sharing the same
`logit_position` and target merge provenance; conflicting atoms are diagnosed
and resolved deterministically.

Sequence loss is normalized by atom weights:

```text
sequence_loss =
    sum_i atom_weight_i * atom_loss_i
    / max(eps, sum_i atom_weight_i)

batch_loss =
    mean(sequence_loss over retained rollout_attempt sequences)
```

No additional clean/dirty/UL bucket normalization is applied in v1.

### Residual Scan State

The scanner walks rollout rows in order. A committed row is a legal row matched
to a remaining GT or promoted UL object by exact normalized description and
IoU `>= 0.75`. Tie-breaks are deterministic.

Uncommitted rows include invalid geometry, structural malformed rows, duplicate
bursts, low-IoU localization misses, failed/unpromoted UL candidates, and
wrong-description unmatched rows. Uncommitted rows do not update emitted or
remaining state.

### Dirty Prefix Handling

Dirty tokens may remain as masked context when the builder can still locate a
reliable boundary and valid logit position. No loss is applied to dirty prefix
tokens unless a reliable correction atom is attached at a causal position.

Structurally malformed middle spans:

```text
resync reliable:
    keep malformed span as masked dirty context
    no atoms/type loss inside the malformed span
    continue scanning suffix

resync unreliable:
    cut back to last stable boundary or drop sample
```

Trailing incomplete object spans are removed from their `<|object_ref_start|>`
and the prefix reverts to the last stable boundary.

### No Coordinate Repair

The prior `bbox_tail_from_anchor` design is deleted for Stage-2 v1. Do not
implement raw-rollout coordinate repair, coordinate tolerance, nearest-GT
repair, invalid-bbox repair, low-IoU coordinate refinement, or regression-ish
bbox losses.

Bbox geometry is a commitment gate:

```text
legal + exact desc + IoU >= 0.75:
    commit and update remaining

otherwise:
    uncommitted dirty prefix
    remaining unchanged
```

Coordinate supervision still occurs in constructed teacher-forced suffixes for
selected remaining objects, and in the optional clean GT stream if explicitly
enabled.

### Constructed Suffix

Constructed suffixes contain all remaining supervision objects plus EOS if the
complete suffix fits. Object order is deterministic random:

```text
base seed = 17
derived from sample / rollout / suffix-start boundary
GT and promoted UL shuffled together
```

Order is fixed for the built correction sequence in v1. It is not resampled per
epoch or optimizer step. Clean-success rollouts are skipped by default.

Optional clean GT SFT stabilizer may be supported, but default mix is `0` and
it should use GT only in v1.

### UL Mining

UL mining is collect-then-classify:

1. aggregate K rollout attempts after exact dedup;
2. parse rows and collect legal unmatched non-duplicate proposals;
3. cluster same-normalized-description proposals across distinct rollouts;
4. require cluster IoU `>= 0.9`, `K_valid >= 4`, support from every valid
   retained rollout id, and support/K_valid `>= 1.0` by default;
5. reject GT conflicts and near-GT gray-zone clusters;
6. promote remaining passing clusters as sample-local UL supervision objects
   with default weight `0.5`.

Near-GT gray zone:

```text
same-desc IoU >= 0.75:
    reject as GT conflict / already explained

same-desc 0.30 <= IoU < 0.75:
    reject from training, keep review artifact

same-desc IoU < 0.30:
    eligible if all other UL gates pass
```

No hard cap is applied to the number of promoted UL objects per sample in v1.
High UL/GT ratio is diagnosis-only.

Consensus admits a UL cluster; it does not define one canonical training bbox.
For rollout attempt `k`, the promoted UL supervision object uses rollout `k`'s
own member bbox/desc. The medoid/representative bbox is review metadata only.
If exact duplicate removal or invalid attempts leaves fewer than four valid
rollout ids, default v1 behavior is no pseudo-positive promotion for that
sample.

UL review rows are written to:

```text
monitor_dumps/ul_clusters.jsonl
```

Rows use canonical norm1000 xyxy bboxes and include enough image provenance for
later visualization. PNGs are not generated by default.

### Duplicate Burst

Duplicate burst is pred-vs-pred, same normalized description, legal positive
area, same rollout attempt, and IoU `>= 0.95` with an earlier prediction. It is
checked before GT/UL matching. Duplicate rows are uncommitted, cannot vote for
UL, and do not produce duplicate-specific unlikelihood.

### Spatial Wrong-Description Conflict

A legal bbox with high desc-agnostic IoU to a remaining GT or promoted UL but a
different description is a `spatial_wrong_desc_conflict`:

```text
desc-agnostic IoU >= 0.75
desc mismatch
```

When the rendered/tokenized text span and earliest divergence position are
reliable, it MUST receive an earliest-divergence description correction atom
with reduced weight:

```text
label_conflict_weight = 0.25
effective weight = context/source weight * label_conflict_weight
```

It is never committed, never removes a remaining object, never enters UL
candidate/promotion, and does not get a dedicated artifact in v1. Compact
diagnostics and capped examples are sufficient.

If the divergence span is not reliable, the builder MUST diagnose the no-atom
reason instead of silently treating the conflict as ordinary FP or UL evidence.

## Risks / Trade-offs

- Dirty prefixes may contain tokens the model should not imitate. Mitigation:
  prefix labels remain masked; atoms only attach at validated causal positions.
- Offline prepared rollouts may become stale after training. Mitigation:
  use round-based refresh rather than online generation in v1.
- UL mining may reinforce correlated hallucinations. Mitigation: strict
  consensus, exact dedup, duplicate exclusion, near-GT gray zone, UL weight
  `0.5`, and review artifacts.
- Type loss may expose malformed parser mistakes. Mitigation: prefer drop/resync
  over weakening type supervision.
- Template compatibility is fragile. Mitigation: centralize rendering/slicing in
  `TemplateBoundaryAdapter` and validate spans.

## Migration Plan

1. Update OpenSpec/docs to make this rewritten contract canonical.
2. Replace or refactor current Stage-2 target builders into layered modules:
   row segmentation, row classification, semantic scan, atom extraction, target
   sequence assembly.
3. Add the template boundary adapter and shared target IR validator before
   connecting the trainer.
4. Implement prepared-rollout JSONL ingestion with strict token-id handling.
5. Implement residual-set correction, UL mining, label-conflict handling, and
   diagnostics.
6. Preserve hard-SFT/current Stage-2 baseline configs unchanged.
7. Run targeted unit tests, then small offline rollout smoke/overfit runs.

Rollback: select baseline objective modules and ignore the residual-set
objective path. Prepared rollout artifacts are offline inputs and do not mutate
dataset GT.

## Open Questions

- Exact file/module split is deferred to the super-power implementation plan.
- The optional clean GT stabilizer config name and sampling ratio surface should
  be finalized during implementation, with default disabled.
