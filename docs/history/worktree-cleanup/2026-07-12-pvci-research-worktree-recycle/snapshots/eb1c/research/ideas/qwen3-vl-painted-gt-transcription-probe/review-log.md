---
type: investigation
title: Qwen3-VL Painted-GT Transcription Probe Review Log
description: Records review-convergence findings and resolutions for the painted-GT transcription probe.
tags: [coordexp-swift, painted-gt, review-convergence]
state: active
updated: 2026-07-04
---

# Review Log

## 2026-07-04 Research Docs Review

Mode: docs/spec/plan.

Mutation scope: docs-only.

Reviewed artifacts:

- `research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md`
- `research/ideas/qwen3-vl-painted-gt-transcription-probe/experiment-plan.md`
- `research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md`
- `research/ideas/qwen3-vl-painted-gt-transcription-probe/index.md`

Review lanes:

- contract/spec and launch-gate auditor: completed; verdict held for targeted
  revision; no P0; P1 findings accepted.
- model-behavior validity auditor: completed; verdict held for targeted
  revision; no P0; P1 findings accepted.
- implementation-surface mapper: completed; verdict held for targeted
  revision; no P0; P1 findings accepted.
- research narrative synthesizer: completed; verdict held for targeted
  revision; no P0; P1 findings accepted.

Rejected findings:

- none. All material findings were accepted or folded into an accepted finding.

Accepted P1 findings and resolutions:

- Gate thresholds were too qualitative. Resolved by adding strict unpainted
  val200 baseline identity, pass/hard-fail/gray-zone thresholds, mark-control
  margins, row-validity floors, and self-prefix block conditions.
- Mark-dependence controls were not operational enough. Resolved by separating
  paint-all controls from stepwise controls and adding deterministic hard
  negative selection rules.
- Stepwise teacher-prefix could silently supervise prefix rows. Resolved by
  requiring previous GT prefix labels to be ignored and only the current row to
  be label-bearing.
- Repetition-penalty comparisons could be mixed. Resolved by making
  `temperature=0`, no sampling, `repetition_penalty=1.10` the primary gate and
  `1.05` a sensitivity panel only.
- Baseline reproduction lacked an exact metric handle. Resolved by pinning
  `eval_coco_fixed_gt_scale/metrics.json`, key `mAP`, accepted value
  `0.4111788135144427`, and launch threshold `mAP >= 0.40`.
- Adapter capacity changes needed load/trainable receipts. Resolved by
  requiring adapter-load and trainable-surface receipts before GPU launch.
- Stepwise controller and painted-data owners were unclear. Resolved by
  requiring branch-local OpenSpec owner surfaces for materialization,
  controllers, parser/eval artifacts, DoRA config, and validators.

Accepted P2 findings and resolutions:

- The source-truth plan needed the overview's decision posture. Resolved by
  adding a `Decision Posture` section to `experiment-plan.md`.
- Debug F1 was referenced before being defined. Resolved by requiring OpenSpec
  to define denominators, matching rules, invalid-row handling, duplicate
  handling, and metric scope before use.
- Random schedule interpretation was underspecified. Resolved by scoping V1 to
  one frozen random schedule per image as a single-seed diagnostic only.
- Reviewer timeout/disconnection semantics were implicit. Resolved by marking
  timeout, disconnection, missing evidence, and vague output as unresolved.

Second-round review:

- contract/spec auditor: completed; no remaining P0/P1; one P2 governance-log
  detail accepted and fixed here; verdict converged for user manual review.
- model-behavior validity auditor: completed; no remaining P0/P1/P2; verdict
  converged for user manual review.

Second-round rejected findings:

- none.

## 2026-07-04 Claude Review Triage

Mode: docs/spec/plan.

Mutation scope: docs-only.

Reviewed artifact:

- `review/claude.md`

Accepted findings and resolutions:

- Claude's P0 on LLM-only baseline DoRA versus all-tower DoRA was accepted, but
  the initial `merge_then_fresh_all_tower_dora` resolution was superseded after
  user review. The research docs now adopt `warm_start_expand_dora`: the
  original pretrained base model remains the base, matching language-side
  adapter tensors are warm-started from the step-917 LLM-only adapter, missing
  vision/aligner adapter tensors are initialized, and coordinate/wrapper
  special-token embeddings are loaded through the approved trainable embedding
  mechanism.
- Claude's invalid/unparseable denominator concern was accepted as an OpenSpec
  carry-in. The plan now requires malformed, invalid, and unparseable outputs to
  remain in diagnostic denominators as failures or misses.
- Claude's painter-to-coordinate geometry concern was accepted. The plan now
  elevates geometry provenance into an acceptance gate routed through
  `src/data/geometry.py::coord_bins_to_pixel_xyxy` and `src/qwen/images.py`
  no-resize planning.
- Claude's interpretation caveats were accepted. The docs now state that V1
  marks provide geometry but not category text, and that paint-all remains
  entangled with canonical `geo_sorted` output order.
- Claude's self-prefix cost note was accepted as an OpenSpec carry-in. The plan
  now requires the controller serialization boundary, object-step cap, and
  max-new-token budget per step to be pinned before launch.

Additional user-approved clarification:

- The training infrastructure must be upgraded before the real research
  experiment to support partial adapter warm-start and expansion. This is
  acceptable branch-local infrastructure work.
- The original pretrained base checkpoint remains the base; do not save or rely
  on a merged full model as the primary path.
- Receipts must cover every trainable parameter boundary, including expanded
  adapter tensors, coordinate-token embeddings, wrapper-token embeddings,
  optimizer groups, and dense-base freeze status.

Findings already resolved by the preceding patch:

- Numeric threshold policy: already resolved through the pass, hard-fail, and
  gray-zone gate.
- Repetition-penalty ambiguity: already resolved through primary
  `repetition_penalty=1.10` and optional `1.05` sensitivity panel.
- Bounded self-prefix hard-gate semantics: already resolved through the bounded
  diagnostic and block/narrow-proceed rules.
- Stepwise prefix label masking: already resolved through the current-row-only
  label-bearing contract.

Rejected findings:

- none.

## 2026-07-04 Self-Driven Handoff Review

Mode: docs/spec/plan.

Mutation scope: docs-only.

Trigger:

- The user approved the self-driven long-term research task and asked for Git
  version management while they were away.
- The user had already approved the `warm_start_expand_dora` direction.

Review lanes:

- contract/spec auditor: completed; no P0; P1 findings accepted.
- upstream relation tracer: completed; no P0; P1 findings accepted.
- model-behavior validity auditor: completed; no P0; P1 findings accepted.

Accepted P1 findings and resolutions:

- Decode surface was underspecified. Resolved by making
  `free_raw_generation` the primary launch-gate decode surface, forbidding
  compact grammar/trie constraints in the primary gate, and requiring a
  `decode_surface` identity table in OpenSpec.
- Primary mode score was underspecified. Resolved by predeclaring
  `debug_detection_f1` for `paint_all` and `per_target_step_debug_f1` for
  stepwise modes, with invalid/unparseable outputs retained in denominators.
- `warm_start_expand_dora` lacked complete runtime identity. Resolved by
  requiring base model, tokenizer, processor, generation config, special-token
  id map, warm-start report id, source adapter, source embedding payload, and
  evaluated adapter checkpoint identities in manifests.
- Warm-start tensor copy was too shape-oriented. Resolved by requiring exact
  canonical source-to-target key maps, tensor hashes, and post-copy equality
  checks; ambiguous shape-only matching is forbidden.
- Special-token embedding payload identity was underspecified. Resolved by
  choosing the repaired step-917 payload under
  `outputs/coordexp_swift/infer/val200_support/repaired_special_token_embeddings_step917`
  as the canonical source embedding payload.
- Tiny overfit training budget allowed unbounded checkpoint selection. Resolved
  by predeclaring `2` initial epochs, extension up to `8` total epochs only
  under recorded improving behavior, all-checkpoint reporting, and an explicit
  selected-checkpoint rule.

Accepted P2 findings and resolutions:

- Frozen self-prefix was optional in a way that could blur comparative claims.
  Resolved by keeping it optional only for hard-safety diagnostics; any
  self-prefix improvement/degradation claim now requires paired frozen and
  trained self-prefix rows.
- Review log scope and stop state were stale and duplicated. Resolved by
  recording this new review round and replacing the stop state below.
- The idea index did not expose stop state. Resolved by adding a status pointer
  to `index.md`.

Current stop state:

- Superseded by the Superpowers implementation roadmap review below.

## 2026-07-04 Superpowers Implementation Roadmap Review

Mode: docs/spec/plan.

Mutation scope: docs/OpenSpec/research-plan only. No `src/`, `scripts/`,
`configs/`, `tests/`, GPU launch, or training artifact mutation occurred during
this review loop.

Reviewed artifacts:

- `docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md`
- `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-decode-eval/spec.md`
- `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-launch-gates/spec.md`
- `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-materialization/spec.md`
- `openspec/changes/add-painted-gt-transcription-probe/tasks.md`
- `research/ideas/qwen3-vl-painted-gt-transcription-probe/experiment-plan.md`

Review lanes:

- contract/spec auditor: completed; one P1 and one P2 accepted and patched.
- upstream relation tracer: completed; no P0/P1; P2 provenance hardening
  accepted and patched.
- module-boundary mapper: completed; no P0/P1; P2 anti-duplication guardrails
  accepted and patched.
- model-behavior validity auditor: completed; no P0/P1; P2 metric semantics
  accepted and patched.
- focused contract re-check: completed after patches; one remaining P1 and one
  P2 accepted and patched.

Accepted P1 findings and resolutions:

- The roadmap pinned exact V1 stop/EOS behavior, but OpenSpec only required an
  unspecified stop/EOS policy. Resolved by adding the normative
  `V1 stop/EOS and parser-strip policy` requirement to the decode/eval delta:
  Qwen `<|im_end|>` as `eos_token_id`, tokenizer `pad_token_id`, raw token
  trace retention, exactly-one terminal stripping through
  `strip_policy=terminal_im_end`, `stop_reason` values `im_end` and `length`,
  truncation counters, and comparison rejection for stop-policy mismatch.
- The research experiment plan still carried the older checkpoint-report
  wording while declaring itself the branch source-truth experiment contract.
  Resolved by updating the tiny-overfit gate to require planned epochs, actual
  epochs, eval cadence, every evaluated checkpoint, full checkpoint metrics,
  loss trajectory, selected-checkpoint rule, selected-checkpoint reason, and
  selected-checkpoint provenance.

Accepted P2 findings and resolutions:

- Final gate provenance was stronger in the roadmap than in OpenSpec. Resolved
  by extending the launch-gates delta and task `7.7` with actual epochs, loss
  trajectory, selected-checkpoint reason, and selected-checkpoint provenance.
- Pack-cache ownership could drift into painted-specific training code.
  Resolved by requiring `src/painted_gt` to emit a stable materialization
  identity consumed opaquely by generic packing-cache determinants; pack-cache
  must not parse painted manifests or import `src.painted_gt`.
- Painted decode controllers could accidentally hand-roll chat-template prompt
  text or tokenization. Resolved by requiring controller tests and artifacts to
  prove use of the existing template/inference prompt owner and prompt-token
  parity evidence.
- Paint-all full-response parse failure accounting was ambiguous. Resolved by
  defining one invalid generation-attempt false positive plus unmatched-GT
  false negatives when no concrete malformed row attempts can be segmented.
- Missing confidence metadata could be misread as a hard metric failure.
  Resolved by defining scoreless debug-F1 as eligible through emission-order
  fallback when a scoreability policy is recorded; missing scoreability policy
  metadata remains invalid.
- Repaired selected-token embedding payload metadata hashes were only partially
  pinned. Resolved by adding SHA256 values for
  `special_token_embeddings.json`, `special_token_embeddings.safetensors`, and
  `repair_receipt.json`, plus the `shared_embed_delta` tensor hash and nonzero
  count.
- The accepted val200 input JSONL path lacked a content hash. Resolved by
  pinning SHA256
  `9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4`
  and requiring the sanity-source probe to assert it.
- The Superpowers report-test checklist omitted actual epochs and
  selected-checkpoint reason. Resolved by requiring tests to reject reports
  missing any required checkpoint provenance field.

Rejected findings:

- none. All material findings were accepted and patched.

Verification after final patches:

```bash
openspec validate add-painted-gt-transcription-probe --strict
openspec instructions apply --change add-painted-gt-transcription-probe --json > temp/painted_gt_openspec_instructions.json
python - <<'PY'
from pathlib import Path
paths = [
    Path("docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md"),
    *Path("openspec/changes/add-painted-gt-transcription-probe").rglob("*.md"),
    *Path("research/ideas/qwen3-vl-painted-gt-transcription-probe").glob("*.md"),
]
for p in paths:
    text = p.read_text()
    for bad in ["TO" + "DO", "TB" + "D"]:
        assert bad not in text
    assert text.count(chr(96) * 3) % 2 == 0
print("docs hygiene: ok")
PY
git diff --check -- docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md openspec/changes/add-painted-gt-transcription-probe research/ideas/qwen3-vl-painted-gt-transcription-probe
```

Verification result:

- OpenSpec validation passed.
- OpenSpec instruction JSON materialized successfully.
- Markdown fence and residue hygiene passed across the scoped planning docs.
- `git diff --check` passed on the scoped planning docs.

Current stop state:

- Planning/spec/roadmap baseline is converged for source implementation under
  the user's self-driven research authorization.
- Source implementation remains gated by the roadmap tasks, review loops, and
  targeted tests.
- GPU tiny training and the larger two-epoch painted run remain gated by
  implementation, source probes, visual audit, baseline sanity, tiny gates, and
  final launch-gate review.
