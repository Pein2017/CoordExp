# Fourth fit: completed training-set evaluation and user stop

The exact step64→256 continuation produces a positive natural-greedy training-set signal, but the cumulative stage remains incomplete. Work stops here for user discussion. No fifth fit, data expansion, new training recipe, mechanism probe, held-out evaluation or GPU job is launched.

## Accepted outcome

| Metric | Parent64 | Final256 |
|---|---:|---:|
| Frozen training owners covered | 177/232 | 212/232 |
| Frozen training owners missing | 55 | 20 |
| Current all-known owners covered, after this review | 183/248 | 213/248 |
| Current all-known owners missing | 65 | 35 |
| Physical repeat rows | 11 | 11 |
| Confirmed false-object rows | 14 | 3 |
| Physical-identity unknown rows | 34 | 4 |
| Parser-invalid rows | 365 | 18 |
| Natural EOS images | 10/11 | 11/11 |
| Capped images | 1 | 0 |

The frozen232 paired contrast retains171 owners, gains41 and loses6: net+35. All six frozen-owner losses occur in image528944. The current248-owner supplemental contrast retains171, gains42 and loses12: net+30. This supplement includes reviewed objects discovered after the training bank was frozen; it is not a claim that all248 were training targets.

Six images have zero missing on the frozen232 population and no physical/structural debt. Only three images (99937,210457,323322) have zero missing on the current all-known248 population. Category correctness remains separate.

| Image | Frozen targets | Parent64 covered | Final256 covered | Final missing | Lost old | New fixed |
|---|---:|---:|---:|---:|---:|---:|
| 25274 | 33 | 21 | 33 | 0 | 0 | 12 |
| 59571 | 23 | 12 | 18 | 5 | 0 | 6 |
| 99937 | 8 | 8 | 8 | 0 | 0 | 0 |
| 210457 | 5 | 5 | 5 | 0 | 0 | 0 |
| 219546 | 41 | 27 | 34 | 7 | 0 | 7 |
| 323322 | 7 | 7 | 7 | 0 | 0 | 0 |
| 351017 | 25 | 17 | 25 | 0 | 0 | 8 |
| 388795 | 14 | 12 | 13 | 1 | 0 | 1 |
| 417044 | 33 | 30 | 33 | 0 | 0 | 3 |
| 477415 | 33 | 28 | 32 | 1 | 0 | 4 |
| 528944 | 10 | 10 | 4 | 6 | 6 | 0 |

## Review and FP semantics

The user clarified during review that view_image is for unmatched rows only. Accepted v2 policy inherits the existing class-agnostic, cardinality-first one-to-one IoU>=0.5 matches. Fixedv4 matches are preserved first; remaining predictions use the same matcher against v5-only accepted owners. Only remaining valid unmatched rows need original+bbox+crop review. IoU0.8 is diagnostic. A geometric match does not establish category correctness.

The paired review accounts for all858 raw rows:394 inherited matches,81 visually reviewed valid unmatched rows,383 parser-invalid rows. Earlier matched-row visual observations remain locally preserved, but do not veto inherited matches. This matched-inheritance result is not directly comparable to earlier all-row strict-extent physical summaries without applying the same rule.

All viewed interpretations are saved in per-image review-notes.jsonl and root view-notes.jsonl. Exact image/box/literal evidence is reusable; physical repetition is recomputed in current generated order. Original worker candidates remain untouched; accepted policy projections fix inherited matches, recover verified class labels from existing owner annotations, and apply six explicit root rulings to unmatched rows. An unaccepted projection with a null-owner repeat was preserved as superseded before correction.

At final256 the three confirmed false rows are composite/background proposals in59571 (two) and219546 (one). Four other rows remain physically uncertain and are not counted as FP. Four small rear-chair upholstery boxes in388795 are partial repeats with wrong class/extent, not four new nonexistent physical owners; their direct CE stays masked. The partial table label remains unknown and is not admitted as an independent object.

Final valid rows have10 wrong classes and4 unknown classes; all18 parser-invalid rows also carry class-unknown metadata. Final parser drops are13 malformed_object_span and5 geometry_invalid. Class errors and invalid output remain separate from owner recall.

## Training fit and execution

Same11 corrected synthetic teacher routes, fixed232 owners,2203 active CE tokens,928 coordinate tokens and11 EOS targets. Language DoRA only; base/vision/embedding/lm_head frozen. AdamW lr1e-5 with exact optimizer/RNG continuation; bbox expected-coordinate raw-axis hinge weight0.01. No rollout refresh or new recipe during this continuation.

Added192 optimizer updates and2112 image forwards; cumulative256 updates and2816 forwards. Training used GPU0 and completed in3474.64s (57.91min). Eight cold readback workers used nine model loads for33 saved native requests; controller training+readbacks completed in3816.58s (63.61min). All checkpoint/optimizer tensors finite, source identity, prompts, media/grid, empty assistant prefix, tokens, decoder/parser and stop bindings passed root CPU replay.

| Update | Training objective | Mean geometry hinge |
|---|---:|---:|
| 1 | 1.823892 | 0.02017474 |
| 16 | 1.307717 | 0.01375194 |
| 32 | 1.063720 | 0.00655288 |
| 64 | 0.691585 | 0.00236260 |
| 128 | 0.383503 | 0.00007012 |
| 192 | 0.241790 | 0.00000000 |
| 256 | 0.165972 | 0.00000000 |

The last32 objectives decrease at every saved update, from0.198074 to0.165972; no observed plateau is claimed. The logged teacher-forced geometry hinge reaches zero while final greedy still has geometry/parse failures. These observations establish an objective/readout gap, not its causal mechanism. No convergence or transfer claim follows from this training-only result.

| Checkpoint | Valid parsed rows | Invalid rows | EOS/cap images |
|---|---:|---:|---:|
| 128 | 277 | 248 | 10/1 |
| 192 | 262 | 13 | 11/0 |
| 256 | 231 | 18 | 11/0 |

128 and192 are saved native structural diagnostics; full paired unmatched owner review covers64 and256. No held-out validation veto was used.

## Annotation persistence

The local JSONL now preserves169 original GT objects plus79 valid unlabeled owners across11 images. All77 earlier unlabeled entries keep their non-provenance content. This evaluation adds one verified tan donut from parent417044 p52 and one visible utensil from final219546 p38 whose specific class is unknown. The79 unlabeled list includes19 unknown-class entries with null descriptions and explicit masking semantics. Every owner binds an actual local decision and original+bbox+crop evidence.

The completed training bank remains frozen232. Latest annotation catalog isv6 with248 known owners; latest data export is annotations-with-unlabeled-v4/annotations.jsonl. No new training consumes it.

## Interpretation for discussion

Observed: more optimizer updates on the same CE recipe improve greedy coverage and stopping markedly. Observed: improvement is uneven; image528944 loses six old objects and the final trace still repeats and emits invalid rows. Inference: CE remains a useful learning channel here, but lower teacher-forced loss alone does not certify complete, clean greedy trajectories. The expected-coordinate hinge is not a hard guarantee on native geometry. This round does not distinguish residual underfitting, trajectory-distribution effects, or other causal explanations. Work is stopped for discussion rather than selecting another experiment.

## Reproducible evidence

- [Accepted physical ledger](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-review-extraction-v1/ledger.json)
- [Current-known248 supplement](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-review-extraction-v1/current-known-v6-supplement.json)
- [Root acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-review-extraction-v1/root-acceptance.json)
- [All858 raw rows and decisions](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-review-extraction-v1/full-review-results.jsonl)
- [Review evidence index](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-review-extraction-v1/evidence-index.json)
- [Matching inheritance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-owner-reviews-v1/match-inheritance-v2.json)
- [Execution acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/root/fourth-fit-execution-acceptance-v1/root-acceptance.json)
- [Training curve](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/root/fourth-fit-execution-acceptance-v1/training-curve.png)
- [Enriched annotations](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/annotations-with-unlabeled-v4/annotations.jsonl)
- [Annotation acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/annotations-with-unlabeled-v4/root-acceptance.json)

Replay the existing CPU receipts:

```bash
CUDA_VISIBLE_DEVICES='' python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/root/fourth-fit-execution-acceptance-v1/verifier.py --verify-existing
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-owner-reviews-v1/normalize_accepted_policy.py
python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/fourth-fit-owner-reviews-v1/validate_and_extract.py --write
```
