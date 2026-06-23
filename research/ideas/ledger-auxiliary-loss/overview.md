---
type: idea
title: Ledger Auxiliary Loss
description: Explores whether a training-only object-coverage ledger loss can improve Stage-1 autoregressive detection coverage memory without changing inference.
tags: [stage1, compact-detection, teacher-forcing, auxiliary-loss, coverage-ledger, duplication, recall]
state: active
updated: 2026-06-23
---

# Ledger Auxiliary Loss

## Current State

Ledger Auxiliary Loss is an active Stage-1 detection teacher-forcing research
pilot. Its design and implementation plan are preserved as research context,
while the current checked implementation state is launch-prep / not-yet-run:
the smoke route, preflight command, metrics, and artifact contracts are
documented, but the tiny smoke training run still requires explicit runtime
approval and complete local data/model assets. It has not been promoted into
stable training guidance.

The v0 scope is intentionally narrow: add a training-only auxiliary objective
to test whether autoregressive hidden states can learn an internal object
coverage ledger during normal teacher-forced forward passes. The initial scope
does not modify inference, decoding, tokenizer allocation, or model
architecture at inference time. V0 deliberately selects the existing
`compact_object_box_closed` detection template because the ledger state needs
both `<|object_ref_end|>` and `<|box_end|>` tokens; `compact_full` is treated
as an older chat-template/schema label rather than the current Stage-1 semantic
template id.

## Central Question

Can a teacher-forced autoregressive detection model learn a prefix-conditioned
coverage state that separates already-emitted annotated objects from
not-yet-emitted annotated objects, and does that internal separation correlate
with improved recall or reduced duplication in later evaluation?

## Philosophy

Ledger loss imposes an internal object-coverage inductive bias. It encourages
the autoregressive hidden state after each completed object span to encode
which annotated objects have already been emitted. This may help if duplication
and low recall arise from weak prefix-side coverage memory. However, it does
not by itself guarantee that the model will use this coverage state to select a
new valid object, nor does it solve visual perception, small-object
localization, missing annotations, or router/commitment failures.

## Worktree And Branch Handles

- worktree: `/data/CoordExp/.worktrees/ledger-auxiliary-loss`
- branch: `codex/ledger-auxiliary-loss`
- base: local `main` at `87c03b1a1292addd0ad3ecc8f43c61075ce6019a`

## Main Reading Path

- [Draft](draft.md) - original v0 idea and requested implementation shape
- [Discussion](discussion.md) - resolved scope, philosophy, and open design gates

This idea is active and does not yet have a final `experiments/` packet or
`conclusion.md`.

## Current Decision Boundary

- Use Stage-1 detection teacher forcing as the first implementation and
  evaluation surface.
- Use structured rendered/tokenized object-entry metadata as the source of
  truth for ledger state positions and emitted-object identity.
- Treat token-stream parsing as a debug/assert fallback, not as the canonical
  training path.
- Use `compact_object_box_closed` for v0. Both `<|object_ref_end|>` and
  `<|box_end|>` are mandatory; plain `compact`, `compact_box_closed`, and
  historical `compact_full` are not acceptable ledger-pilot templates.
- Use the upstream dependency docs before choosing the visual-token access
  point; do not edit installed Hugging Face Qwen model files.
- Capture the Qwen3-VL-convention projected `image_embeds` through the normal
  forward/helper/hook path. Do not recompute the vision tower for the production
  ledger loss. These embeddings are post vision blocks and post
  PatchMerger/aligner, not raw pre-merger `vision.blocks[-1]` states.
- Scope v0 detection inputs to exactly one image.
- Keep the ledger path strict once enabled: unexpected missing metadata,
  malformed object rows, or zero-object training examples should fail fast.
- Detach visual object embeddings by default.
- Start with full state-object scoring and implement it as a ragged flattened
  pair computation over one unpadded physical sequence per forward pass. Reject
  static packing and padding-free packing in v0 until offset rewriting is
  explicitly tested.
- Pool object visual embeddings from the minimal enclosing set of visual patch
  embeddings whose grid cells contain the bbox pixels; do not treat a bbox as
  having "no visual embedding" under normal aligned-image assumptions.
- Map bboxes to the Qwen3-VL LLM-facing post-merge visual-token grid, not to
  raw pre-merger vision-block patch states. Keep bbox endpoints as floats until
  post-merge cell selection, use half-open geometry, and fail fast on any
  image/grid/placeholder mismatch.
- Use the object row's `<box_start>` position as the object-region alignment
  anchor when applying a direct force between the coordinate-start state and
  the selected visual region. The required causal check is that the next-token
  label after `<box_start>` is the first coordinate token for the same object.
- Keep the region-anchor force inside the same ledger objective, but expose it
  as a named subterm for metrics, e.g. `region_anchor_loss`. For v0, this
  subterm binds only the current row object as positive and masks all
  non-current objects. Use a separate region-anchor state projection while
  sharing the visual object projection with the main coverage-ledger scoring
  path.
- Use final-layer hidden states from the same forward pass for prompt-end and
  row-completion ledger states.
- Keep the active config surface simple: expose only meaningful choices and do
  not add knobs for hardcoded v0 behavior such as strictness.
- Use the active Stage-1 config path `objective.terms.coverage_ledger`; do not
  revive retired `objective.modules` authoring.
- Make the ledger projection dimension tunable as `ledger_projection_dim`,
  defaulting to `256` unless design review finds a better value.
- Score state-object pairs with normalized fp32 projections and
  temperature-scaled logits. Default temperature should be conservative
  enough for v0 stability, with a hard validation floor rather than silent
  clamping.
- Use separate loss weights for the main coverage-ledger BCE and the
  region-anchor binding subterm. Default both `coverage_weight` and
  `region_anchor_weight` to `0.1` for an observable first pilot.
- Include AUC and thresholded accuracy as monitoring metrics if they can be
  computed from the same forward-pass ledger logits and labels without a new
  dependency. Use metric keys under `teacher_forcing/ledger/*`.
- Launch first with a 128-sample smoke/overfit run to debug runtime configs,
  efficiency, and expected ledger separation. Select the 128 examples randomly.
  The smoke/overfit phase must emit both numeric JSONL alignment dumps and a
  16-sample overlay visualization gallery for bbox-to-visual-token mapping.
  The random selection must use seed `20260623` and write a selected-sample
  manifest under the run artifact root.
- Follow the repo's applicable golden rule for the pilot: `per_device=1`, each
  forward pass consumes one long unpadded physical sequence.
- Keep the objective experimental until a later promotion decision.
- Do not create an OpenSpec change unless the loss/config semantics become a
  stable compatibility-sensitive contract.
- Keep executable implementation details in a later super-power design/spec
  and implementation plan.

## Source Map

- `research/ideas/ledger-auxiliary-loss/draft.md` - initial user-provided v0
  idea and constraints.
- `research/ideas/ledger-auxiliary-loss/discussion.md` - discussion decisions
  and next grill gates.
- Current design spec:
  `docs/superpowers/specs/2026-06-23-coverage-ledger-auxiliary-loss-design.md`
- Future implementation plan:
  `docs/superpowers/plans/2026-06-23-coverage-ledger-auxiliary-loss.md`

## Next Action

Wait for explicit user approval before implementation. If approved, start from
the reviewed super-power design spec and implement the feature with targeted
tests, then run only the documented 128-sample smoke/overfit preflight and
launch path before considering production-scale training.
