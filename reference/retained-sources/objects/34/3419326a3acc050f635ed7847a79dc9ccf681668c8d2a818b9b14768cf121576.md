## Why

The current model can fit canonical dense-scene transcripts, but this does not
show how to learn a skipped annotated object without damaging later enumeration.
The existing sampled-positive calibration contract excludes GT paths that were
never sampled; a separate GT-supervised research entry is needed rather than
silently weakening that contract.

## What Changes

- Add an explicit, opt-in research replay path for a model-produced prefix
  followed by an annotation-authored correction. Preserve exact prefix tokens
  and distinguish the generated context from the teacher continuation.
- Materialize paired interception and actual-history backfill examples from
  one Source trajectory bank. Annotated targets need not have sampled support
  or an already-successful Source suffix.
- Support full-remainder and selected-owner row supervision with explicit
  per-token masks and fixed paired normalization. Keep the termination anchor
  identical; do not infer scene exhaustion from a partial COCO annotation list.
- Compare internal language DoRA with a separately saved output-row residual
  using the same full-vocabulary CE, not a new QP objective. Cold inference must
  prove which payload was actually applied.
- Run one shared reference and three alternatives, then stop after fixed-dose
  pilot training and native-greedy evaluation. The scientific population,
  doses, contrasts and outcome rules belong to the linked research unit.

The 2026-09-07 user ruling selects existing COCO only: no LVIS, annotation
enrichment, completeness experiment, new label ontology or data expansion.
Eight GPUs are available for the later apply phase. This proposal itself
authorizes no training or code changes.

## Capabilities

### New Capabilities

- `research-gt-correction-replay`: explicit GT-correction provenance, isolated
  supervision masks, unchanged partial-label termination anchor, surface-bound
  update/save/cold-readback, and bounded independent-arm execution.

### Modified Capabilities

None. Existing sampled-positive own-prefix calibration, canonical CE, QP,
inference defaults and evaluator contracts remain unchanged. The new entry
must not masquerade as the old calibration profile.

## Impact

Reuse current research complete-action replay and optimizer-state mechanics,
native HF frontend/encoding/decoding, the selected-output-row inference hook,
and existing annotated-owner comparisons. Add only the experiment-specific
bank builder, trainer/cold-readback entry and their caller-facing checks; do not
build a generic training/scheduling framework or change production defaults.

Scientific owner: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/unit.md`.
Implementation owner: this change. Team operation remains governed by
`/data/CoordExp/.codex/skills/native-subagents-guide/SKILL.md`; any later skill
revision requires an observed coordination problem, a scoped edit and a check,
not a speculative rewrite during planning.

Non-goals: new RL sampling courses, inference rewind, an external second pass,
new tokens/ledger architecture, global sorting enforcement, new optimizer,
rank search, model promotion, winner combinations, or automatic extra doses.
