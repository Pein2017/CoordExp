# Fresh128 fixed coordinate-readout norm policy

**Independently lead-accepted and closed: inference evidence is valid; physical population benefit remains unresolved.** No training or policy tuning. User stopped exhaustive visual review; only partial/selective visual evidence may be used. No global-policy or teacher promotion. The later user-requested TP/FN/unmatched accounting and human-review visualization export are a separate follow-up; they do not reopen model execution or change this unit's frozen scoring.

## Lead acceptance

The lead replayed the saved-output reducer and tensor verifier on CPU, obtaining exact equality with the sealed results. Checks cover494 unique file bindings, all128 fresh images,72 policy batches,32 old diagnostic sequences including batch companions,9724 fresh shadow steps,137 first-disagreement tensors, effective output-row norms/factors, selected-record object/path preservation, and the stopped physical-review scope. All ten original model PIDs are absent and no experiment model process remains. The worker task is separately active on the user's subsequent visualization request; task activity is not renewed inference.

[Lead acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/lead-acceptance.json) and [CPU verification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/lead-verification.json) own this settlement. Original candidate protocol/state/result bytes are preserved in a [bound snapshot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/candidate-records/manifest.json); sealed terminal and numerical artifacts are unchanged. The [lead verifier](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/lead_verify.py) resolves only those explicitly preserved candidate records when replaying their original bindings.

## Main result: gains with incumbent turnover

| Cohort | Images | O known matches | N known matches | G | L | Net |
|---|---:|---:|---:|---:|---:|---:|
| Fresh population | 128 | 578 | 592 | 30 | 16 | 14 |
| O structurally healthy | 119 | 540 | 541 | 13 | 12 | 1 |
| O structurally unhealthy | 9 | 38 | 51 | 17 | 4 | 13 |

The fresh fixed bank contains919 owners: FN341→327. The frozen primary matcher is class-agnostic, one-to-one at IoU≥0.5; these known geometric matches are not class-aware benchmark TP counts. Eleven images improve in net known matching, eight worsen and109 have unchanged counts; unchanged counts can hide swaps. Fourteen images lose at least one incumbent, including seven with gains as well. Two legal-border incumbents are lost. The paired image-bootstrap95% interval for mean net matches is[-0.015625,0.28125] (10000 resamples,seed19), including zero. This is not a token-level significance estimate.

“Structurally healthy” means O reached EOS with zero invalid/malformed/strict-repeat debt; it does not establish complete or physically correct detection. Its119 images net only+1 (G13/L12), versus+13 in the other9. Thus the pooled positive signal is concentrated in already abnormal outputs and does not establish a broad improvement.

| Full fresh output debt | O | N |
|---|---:|---:|
| Geometry-invalid complete rows |476|5|
| Malformed rows/fragments |2|0|
| Strict valid repeats |604|40|
| Literal repeats including invalid |1032|37|
| Literal invalid repeats |446|0|
| Annotation-unmatched valid predictions (UNKNOWN) |1033|423|
| Capped images |2|0|
| EOS images |126|128|

Image339094 introduces one geometry-invalid row despite a structurally healthy O baseline. No new capped, malformed or strict-repeat image is introduced. Lower UNKNOWN count is not a precision gain by itself: it can include fewer duplicates, discarded real unlabeled owners or removed false instances.

## Shadow evidence and limits

On9724 active N steps there are1294 raw-versus-scaled winner disagreements across121/128 images; every observed disagreement is coordinate→coordinate (x1:313,y1:306,x2:274,y2:401). EOS is included in all traces, and no observed EOS/syntax/category winner is directly switched at the same N state. The policy nevertheless changes coordinate-versus-noncoordinate score competition mathematically; absence of an observed cross-type winner here does not generalize to other states.

Seven images need no action change to reproduce the recorded path;19 have exactly one disagreement. Under the same deterministic execution and tie handling, one changed action followed by original greedy reproduces those19 literal N routes including EOS. None of those19 changes its known-owner set. All remaining repeated-disagreement trajectories require those changes to reproduce THAT exact route, not necessarily to retain equivalent physical coverage on another route. No withdrawal experiment was executed.

The accepted old four cases remain separate. They exactly reproduce their previous outcomes:417044 G12/L0,477415 G16/L0 and clean EOS;351017 still caps;7116 preserves4/6 with clean EOS. Their N disagreement counts are36,45,442 and3 respectively. The two rescue cases have their last disagreement at action offsets305 and239, leaving14 and22 unchanged decisions. Therefore the current evidence does not establish a single early-action explanation for those rescue routes.

## Data, runtime and reproducible evidence

The128 images are a seed19 uniform sample of121834 eligible unique images from current processed COCO train+val coord views (126 train,2 val), excluding current train256/dev128 and the four diagnostic IDs. They are fresh relative to this study, not guaranteed unseen in base pretraining or all historical research. Input views retain the same919 annotations; only image-path rebasing and x1/y1 object ordering needed by the frozen template were materialized. The original source lines are preserved. Selected rows agree across current pixel/norm/coord views and are unchanged from pre-filter backups. Older dataset-wide coord/norm manifest drift is documented, not declared resolved.

R16, its paired original embeddings, native FP32/SDPA, empty prefix,RP1,bs4 groups and cap3084 are fixed. N applies the prior exact all1000-coordinate factor vector at all steps to all batch members. Effective output rows include shared delta; bias is absent; non-coordinate/EOS logits are unchanged at the seam. Input/model parameter checks pass. Four separate original diagnostic batches admit/verify the capture path; their members do not expand the fresh scientific population.

The independent CPU verifier checked72 completed policy units,32 exact old diagnostic sequences including companions, matching O/N native input/prompt/media/grid/MRoPE/readout identities, every active shadow token and prefix hash, and137 sparse first-disagreement tensors. Full fresh reduction exactly matches the previously reduced fresh-only subset used to freeze review selection.

Cost: 33817 batch model forwards, 4409 allocated GPU-seconds (including model load), 1232 seconds from admission start through last model exit. Generation-only receipt time is4035.940 GPU-seconds. All owned model jobs exited0; none remains live. CPU-only preparation/scoring corrections are documented; no GPU retry or scientific arm was added.

- [Artifact index](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/ARTIFACTS.md): complete paired tokens/text/boxes/stops, per-owner ledgers, active shadow winners/margins/roles/offsets, sparse full logits/final head inputs, effective readout rows/factors and provenance.
- [Authoritative numeric result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/result.json), [shadow summary](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/shadow-summary.json), [saved-evidence verification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/saved-verification.json), [cost/job closure](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/cost-and-jobs.json).
- [Frozen panel](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/panel.json), [cohort acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/cohort-acceptance.json), [bounded provenance check](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/cohort/provenance-check.json), [selective-review amendment](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/physical-review/amendment.json).

Full-sequence KV, attention and hidden-state trajectories were NOT captured. Sparse tensors occur only at the first shadow disagreement; complete active winner traces are retained for future targeted replay. No unknown prediction was admitted to GT, no model weight/embedding was changed, and no optimizer was created.

## Decision

This fixed policy substantially reduces harmful output debt and gives a modest known-match gain concentrated in abnormal R16 baselines, with measurable incumbent and border damage. It is a useful conditional policy effect, not established physical-FN improvement across the population, a safe global correction, a norm-origin mechanism or a teacher ready for wholesale distillation. Hold training/global promotion. Any further experiment is a separate lead decision; none is launched.

## Selective physical review after user stop

The32-image sampling manifest was frozen, but the lead overinterpreted the budget as a mandatory all-page/all-row review. The user explicitly stopped this aggressive viewing. Both workers were interrupted; no census was completed or required after the amendment. They returned partial notes reporting25 images encountered (11 Terra,14 Luna); seven selected images remain unreviewed. “Encountered” is not complete adjudication. Root made one original-image spot check on339094 and no additional crops.

Candidate notes include unchanged visible objects, localization/extent differences, fruit/skis/plush-toy grouping ambiguities, and a possible clipped-repetition-to-visible-birds change on309264. None is promoted into new GT or a population physical-FN estimate. In particular, known IoU gains/losses can occur while both outputs still denote the same physical owner. Dense cases and unclear identity remain HOLD.

The inventory validator found two nonexistent Luna page paths (217495 A-page2/B-page2); those coverage claims and that image's correspondence conclusion are held, not silently repaired or used quantitatively. Source notes are preserved unchanged. This is a concrete review-record limitation; it does not invalidate model inference or known-reference scoring. No weighted physical-net estimate is reported from the stopped/selective subset. The root owns the overbroad review brief and has recorded the correction in the team observations.

Physical benefit across the128-image population therefore remains unresolved. There is no physical-precision or old-UNKNOWN-retention promotion.


[Partial visual evidence and limits](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/physical-review/result.json), [team observations](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/team-observations.md), [candidate terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/terminal.json).
