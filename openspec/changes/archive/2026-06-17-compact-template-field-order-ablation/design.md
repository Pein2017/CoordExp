## Context

The current compact template implementation has two partially overlapping
histories:

- the older object-field-order contract controls JSON object key order with
  `custom.object_field_order: desc_first | geometry_first`;
- the newer semantic compact template contract controls closure tokens,
  separators, prompt examples, token-row requirements, and artifact template
  provenance through a semantic template id.

The bbox-first final ablation should reuse both histories without creating a
third compact-format surface.  The target comparison is:

- desc-first rich compact rows with sorted object ordering;
- bbox-first rich compact rows with sorted object ordering;
- both under standard Stage-1 SFT, global static packing length `12000`, LLM-only
  trainability, the new chat template, and checkpoint
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.

Current implementation signals that must change after approval:

- `src/detection/template_contracts.py` renders compact rows desc-first only.
- `src/detection/template.py` strict compact parsing assumes object-ref segment
  before box segment.
- `src/config/schema.py` rejects `detection_template.object_field_order` for
  compact templates and keeps `custom.object_field_order` as the active
  data/prompt order surface.
- prompt, dataset, SFT packing, and cache code already have partial
  `detection_template_id` and `object_field_order` plumbing, but compact row
  bytes do not yet compose the two axes.

This change is docs/spec/plan only until user approval.

## Goals / Non-Goals

**Goals:**

- Define a two-knob compact contract:
  - template id selects closure/separator/wrapper family;
  - `custom.object_field_order` selects desc-first versus geometry-first row
    order.
- Add `compact_object_closed` so the compact closure family covers no closure,
  object closure, box closure, and object+box closure.
- Preserve sorted object instance ordering as an independent ablation axis.
- Make prompt examples, strict parsing, render spans, packed/encoded cache
  fingerprints, inference provenance, and evaluator preflight consume the same
  resolved pair.
- Plan standard Stage-1 SFT configs for desc-first and bbox-first rich compact
  ablations before any production run.

**Non-Goals:**

- No implementation before explicit user approval.
- No new CLI flags.
- No free-form row separator, closure-token booleans, parser-mode, or raw string
  template authoring surface.
- No migration of standard Stage-1 SFT into `TeacherForcingRollin`; standard SFT
  remains an active Stage-1 surface.
- No benchmark or mAP claim from this planning change.
- No eval-time heuristic that guesses template or field order from raw text for
  metric-bearing artifacts.

## Decisions

### Decision: Two authored axes, not one overloaded id

`detection_template.id` owns the compact wrapper family:

```text
compact
compact_object_closed
compact_box_closed
compact_object_box_closed
compact_object_box_closed_lines
```

`custom.object_field_order` owns the order of the two semantic segments in each
row:

```text
desc_first:
<object-ref segment><box segment>

geometry_first:
<box segment><object-ref segment>
```

For the requested rich bbox-first ablation, the canonical row is:

```text
<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|><|object_ref_start|>{desc}<|object_ref_end|>
```

Alternative considered: add ids such as `compact_object_box_closed_bbox_first`.
That would avoid a second resolver input, but it would duplicate an existing
research knob and make the JSON and compact contracts diverge.

### Decision: Keep newline behavior in the template id registry

The line variant remains a semantic id.  This preserves the small registry and
avoids adding `row_separator` as a free authored knob.

Alternative considered: add a third `row_separator` key.  That would make more
variants easy but would reintroduce the independent serialization controls that
the semantic-template work deliberately removed.

### Decision: Strict parsers must consume both axes

The parser selected for training validation, inference materialization, and
post-hoc preflight must know both template id and object field order.  It must
reject rows that use the opposite order instead of auto-detecting them.

Alternative considered: parse either order for compatibility.  That would make
inspection easier but would weaken the ablation because a mismatched run could
silently score as if it used the intended order.

### Decision: Object ordering remains independent

`custom.object_ordering` keeps owning object instance sequence.  The final
ablation pair should pin `custom.object_ordering: sorted` so any mAP or
duplication-severity difference is attributable to row field order plus the
expected stochastic training variance, not instance-order policy.

### Decision: Cache and provenance use the resolved pair

Encoded-sample caches, static packing fingerprints, prompt hashes, resolved
training config, inference summaries, and `gt_vs_pred.jsonl` rows must record
or derive from both:

- `detection_template_id`
- `object_field_order`

`global_max_length` remains part of static packing identity and must be `12000`
for the requested production ablation family.

## Risks / Trade-offs

- Parser/render mismatch can produce empty predictions or false invalid rows ->
  Mitigation: roundtrip tests for each supported template id x field order pair.
- Cache reuse can contaminate desc-first versus bbox-first training -> Mitigation:
  fingerprint tests that change only `object_field_order`, only template id, and
  only `global_max_length`.
- Adding `compact_object_closed` increases template surface area -> Mitigation:
  keep it registry-only, no separate authored closure booleans.
- `compact` geometry-first without closure tokens is more fragile for human
  inspection -> Mitigation: strict marker parsing and the production ablation
  uses `compact_object_box_closed`.
- mAP can be misinterpreted as a wrapper metric -> Mitigation: evaluator spec
  states mAP scores normalized predictions, while duplication severity should be
  reported as a separate behavior diagnostic.
- The previous `detection-template-variants` wording can confuse future readers
  -> Mitigation: add supersession notes to that change and keep this follow-on
  change as the normative amendment for the two-knob ablation.

## Migration Plan

1. Review and approve this docs/spec/roadmap packet.
2. Implement template-contract support for `compact_object_closed` and
   field-order-aware compact segment rendering.
3. Update strict parsing, spans, target rendering, prompts, dataset/SFT paths,
   packing/cache fingerprints, inference provenance, and evaluator preflight.
4. Add paired desc-first and geometry-first Stage-1 SFT configs with sorted
   ordering, packing length `12000`, LLM-only training, and the natural-adjacent
   checkpoint.
5. Run targeted unit tests and two tiny smoke runs before any production train.
6. Launch production only after config resolution, cache identity, smoke output,
   and artifact provenance are verified.

Rollback before training is a normal git revert.  After training a bbox-first
checkpoint, rollback requires preserving the run artifacts and metadata because
the generated rows are no longer desc-first-compatible.

## Open Questions

No blocking implementation questions are left in this proposal.  The remaining
approval decision is whether to proceed with the implementation exactly as
specified here.
