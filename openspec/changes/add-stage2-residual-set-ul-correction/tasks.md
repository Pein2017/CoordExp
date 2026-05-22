Implementation tasks for the rewritten Stage-2 residual-set self-prefix
correction contract. These tasks intentionally reset the older implementation
checklist; prior `bbox_tail_from_anchor` and earliest-only assumptions are
superseded.

## 1. OpenSpec And Documentation

- [x] 1.1 Validate and refine this OpenSpec change after subagent review.
- [x] 1.2 Update stable Stage-2 docs/runbook only after the spec is approved.
- [x] 1.3 Record the final implementation roadmap in a super-power plan before
  code changes.
- [x] 1.4 Keep progress evidence in `progress/`; do not turn smoke metrics into
  OpenSpec success gates.

## 2. Prepared Rollout Input

- [x] 2.1 Define prepared rollout JSONL schema with required
  `response_token_ids`, `raw_text`, decode metadata, and sample/image
  provenance.
- [x] 2.1a Add strict config support for
  `stage2_ab.pipeline.objective[name=residual_set_correction].config.prepared_rollout_jsonl`
  and reject residual-set configs that omit it. Update the active schema
  allowlist/validator (`STAGE2_RESIDUAL_SET_CONFIG_KEYS` and related required
  key checks), not only target-builder defaults.
- [x] 2.1b Replace the old residual-set config surface (`num_rollouts`,
  `coord_span_policy`, `ul_geometry`, `artifact_policy`, and other removed
  keys) with the refined strict key set. Add/update
  `tests/test_stage2_ab_config_contract.py` coverage so a spec-compliant
  `prepared_rollout_jsonl` config parses, omitting it fails at the dotted path,
  and old coordinate-repair configs are rejected.
- [x] 2.2 Implement strict missing-token-id behavior: drop+diagnose for new
  data; explicit legacy fallback only for smoke/ablation.
- [x] 2.3 Implement exact duplicate rollout-attempt dedup by token ids, with raw
  text fallback only in legacy mode.
- [x] 2.4 Emit decode-mode sliced diagnostics for clean success, invalid rate,
  dirty corrections, and UL support.

## 3. Template And Target IR

- [x] 3.1 Implement `TemplateBoundaryAdapter` for Stage-1-compatible assistant
  rendering, tokenization, span exposure, and suffix slicing using
  `src/detection/template.py::get_detection_template`,
  `RenderedAssistantSequence`, and
  `src/detection/tokenization.py::tokenize_rendered_detection_conversation`.
- [x] 3.1a Replace residual-set suffix construction that hand-authors compact
  schema strings, including `_render_compact_objects`,
  `_build_compact_prefix_text_data`, `_compact_object_and_desc_spans`, and
  ad-hoc suffix-token slicing, with adapter-produced rendered/tokenized spans.
- [x] 3.2 Add validator coverage for assistant span location, prefix/suffix
  boundary safety, schema duplication, and template-rendered stop/separator
  behavior.
- [x] 3.3 Introduce or refactor shared `SupervisionAtom` with canonical
  `logit_position` and v1-required `target_position`.
- [x] 3.3a Map the conceptual residual atom onto the current shared IR fields in
  `src/training/teacher_forcing/ir.py`, including required `target_position`,
  `selected_token_id`, `allowed_token_roles`, `selected_token_role`,
  `loss_weight`, `coord_role`, `loss_tags`, and `provenance`.
- [x] 3.4 Validate `logit_position + 1 == target_position` and selected-token
  equality against `input_ids[target_position]`.
- [x] 3.5 Keep loss modules consuming atom-local fields only.

## 4. Type And Inner Loss

- [x] 4.1 Enable standalone token-type exclusivity by default for Stage-2
  residual-set atoms.
- [x] 4.2 Implement inner valid-set marginal likelihood with singleton hard CE /
  EOS behavior as special cases.
- [x] 4.3 Implement per-sequence weighted mean over all active atoms, with no
  second clean/dirty/UL bucket normalization.
- [x] 4.3a Carry rollout-attempt sequence boundaries through the target IR/segment
  metadata so the residual-set loss can compute per-sequence weighted means
  before averaging sequences.
- [x] 4.4 Add diagnostics for type mass, wrong-type mass, atom counts, atom
  weight sums, and sequence loss.

## 5. Row Pipeline

- [x] 5.1 Implement row segmentation over raw rollout token ids/text without
  assigning supervision meaning.
- [x] 5.2 Implement row classification for committed, pending UL candidate, and
  uncommitted rows.
- [x] 5.3 Implement semantic state scan over remaining GT/promoted UL objects
  using exact desc normalization and IoU commit threshold `0.75`.
- [x] 5.4 Implement dirty-prefix recovery gate and malformed-span resync policy.
- [x] 5.5 Implement trailing incomplete object removal back to the last stable
  boundary.
- [x] 5.6 Ensure no raw-rollout coordinate repair atoms are emitted.
- [x] 5.7 Remove or fail-fast reject `coord_span_policy` / `bbox_tail_from_anchor`
  in residual-set configs and migrate existing residual smoke configs away from
  that key, including
  `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`
  or any replacement smoke preset.

## 6. Correction Atom Extraction And Suffix Assembly

- [x] 6.1 Collect all eligible non-conflicting atoms per rollout attempt.
- [x] 6.2 Merge same-logit-position identical targets and diagnose conflicts.
- [x] 6.3 Build deterministic random remaining-object suffixes with base seed
  `17`, fixed per correction sequence, using selected `ValidAction.next_state`
  to drive later suffix/atom construction.
- [x] 6.4 Skip clean-success rollouts by default.
- [x] 6.5 Support optional clean GT SFT stabilizer stream with default disabled.

## 7. UL Mining

- [x] 7.1 Implement legal unmatched non-duplicate proposal collection.
- [x] 7.2 Implement cross-rollout same-desc clustering with cluster IoU `>= 0.9`.
- [x] 7.3 Implement K-valid denominator, support from distinct rollout ids, and
  default consensus ratio `1.0`.
- [x] 7.4 Implement same-desc GT conflict and near-GT gray-zone rejection.
- [x] 7.5 Implement promoted UL medoid representative, weight `0.5`, and no
  hard per-sample UL cap.
- [x] 7.5a Ensure cross-rollout UL consensus is admission-only: each retained
  rollout attempt trains against its own promoted member bbox/desc, while medoid
  or representative bboxes remain review metadata only.
- [x] 7.6 Emit canonical `monitor_dumps/ul_clusters.jsonl` with norm1000 xyxy
  bboxes and image provenance; migrate any step-scoped writer path if needed.

## 8. Duplicate And Label Conflict

- [x] 8.1 Implement duplicate burst detection as same-desc pred-vs-pred IoU
  `>= 0.95` within one rollout attempt.
- [x] 8.2 Ensure duplicates cannot vote for UL and never emit duplicate
  unlikelihood.
- [x] 8.3 Implement `spatial_wrong_desc_conflict` with desc-agnostic IoU
  `>= 0.75`, weight multiplier `0.25`, no commit, no UL candidacy.
- [x] 8.4 Emit compact label-conflict diagnostics without adding a default
  standalone artifact file.

## 9. Configs And Baselines

- [x] 9.1 Add strict residual-set objective config path without silently mutating
  hard-SFT/current Stage-2 baselines.
- [x] 9.2 Keep default clean GT SFT mix disabled.
- [x] 9.3 Reject removed bbox/coord/geometry/duplicate live objective modules in
  the residual-set path.
- [x] 9.4 Provide smoke configs for default self-prefix correction and any
  explicitly approved stabilizer/ablation paths without config explosion.

## 10. Verification

- [x] 10.1 Unit-test prepared rollout schema, exact dedup, and legacy fallback.
- [x] 10.2 Unit-test template boundary adapter and causal logit alignment.
- [x] 10.3 Unit-test token-type loss and valid-set marginal behavior.
- [x] 10.3a Unit-test action-based valid-set construction so selected
  `ValidAction.next_state` commits subsequent object-internal tokens correctly.
- [x] 10.3b Unit-test per-sequence weighted mean loss with one-atom and many-atom
  rollout sequences and non-unit atom weights.
- [x] 10.4 Unit-test dirty-prefix resync/drop, malformed masking, truncation, and
  invalid bbox behavior.
- [x] 10.5 Unit-test UL promotion/rejection/gray-zone/no-cap diagnostics.
- [x] 10.5a Unit-test rollout-local UL member bbox supervision distinct from
  medoid/review bbox.
- [x] 10.6 Unit-test duplicate burst and spatial wrong-desc conflict behavior.
- [x] 10.6a Unit-test deterministic row commitment tie-breaks in same-desc
  crowded cases.
- [x] 10.6b Unit-test canonical UL artifact relative path.
- [x] 10.7 Run targeted tests with `conda run -n ms python -m pytest <targets>`.
- [ ] 10.8 Run small offline prepared-rollout smoke/overfit checks from the
  `et-rmp-ce-ckpt-3660+` / checkpoint-3664 base before model-quality claims.
  CPU prepared-rollout preflight is recorded in
  `progress/diagnostics/2026-05-22_residual_set_refactor_preflight.md`; the
  4-GPU smoke command remains intentionally unlaunched pending explicit
  approval for the high-cost run.
