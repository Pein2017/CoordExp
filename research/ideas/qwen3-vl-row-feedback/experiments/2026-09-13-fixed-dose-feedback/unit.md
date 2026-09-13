---
title: Fixed-dose row-end feedback versus an ordinary contextual slot
description: A bounded paired architecture pilot and supporting content-use diagnostic.
type: idea
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-13-fixed-dose-feedback
topic: qwen3-vl-row-feedback
status: ready_for_serial_paired_fit
evidence_status: distributed_gate_failed_serial_recipe_retained
updated: 2026-09-13
---

# Current contract

The user explicitly authorized this research round after the two independent Astra xhigh assessments, delegated execution to the lead, and made the currently free eight GPUs available. The earlier grill-me launch pause is resolved by that instruction. The review remains historical advice, not an experiment result.

From the frozen N16 anchor, does an additional row-end-to-input continuous feedback route improve complete natural enumeration over an ordinary contextual slot at the same admitted supervision exposure?

This is a fixed-dose architecture pilot. It does not identify absent native memory, a semantic owner ledger, isolated persistence, or a sample-efficiency curve. The strongest alternatives are extra effective computation/readout/gradient paths, target preference overwhelming history, and geometry/ordering drift.

## Ownership and isolation

Worktree: /data/CoordExp/.worktrees/row-feedback-pilot-20260913
Branch: codex/row-feedback-pilot-20260913
Base: f9ff74e11cf7ee08c71479ed80018e3a3510173d
Runtime output: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-row-feedback-pilot/

The research-probes base and the other task's dirty files, acquisition, physical admissions, fit and confirmation roles remain independently owned. No production or shared src changes. New maintained experiment code belongs to probes/row_feedback/. The lead owns this unit, protocol freeze, scheduling, integration and final acceptance.

## Proposed implementation invariants

Two separately trained arms start from the same N16 adapter, with identical language-only DoRA trainable tensors, optimizer, supervision records, visible token targets/exposure and protection mapping.
- S (ordinary slot): one internal slot after every visible box-end token, using the existing box-end embedding.
- F (feedback slot): the same embedding plus the just-completed row's final normalized hidden state, rescaled to the RMS of that existing embedding. Scale is fixed at 1; no learned projection, new vocabulary, layer or coefficient search.
- Both keep original image evidence and append-only history. This is an extra continuous route, not overwritten recurrent memory.
- The slot is internal, consumes a physical model position, is never output or directly predicted, and its logits predict the next visible token. Both arms use identical insertion rules, including erroneous generated rows; no GT controls state writes.
- Training and supplied-history replay construct all prior slots causally, with corresponding MRoPE/cache/mask positions. Future targets cannot enter earlier feedback.
- Future loss must retain the intended gradient through the feedback source; ordinary-slot history gradients also remain live. An inadvertent detach is technical-invalid.
- Native N16 teacher protection is mapped to the same visible histories/next-visible-token targets in both arms. It is distillation across protocols, not same-input parity. No protection at a repaired position requires the known repeated action.
- Same visible output allowance: 3084 generated tokens. Internal slots/work are counted separately and cannot reduce the allowed visible rows.
- FP32/SDPA, original model/base/prompt/coordinate geometry and greedy T0/top-p1/RP1 remain fixed.

Workers must surface a concrete infeasibility before substituting one of these invariants. Ordinary implementation details belong to the execution owner.

## Supervision and independent endpoint

Prefer a read-only immutable snapshot of already lead-admitted credible successors once available from the other task. Do not mutate its bank or independently re-admit its uncertain rows.
The bounded fallback is the physically admitted c/w evidence previously used to train N16, using validated post-completion successors rather than assuming entry-only CE supervises exclusion. These histories and witnesses were produced by the older positive7-support50-81 acquisition adapter, not N16. N16 is the new fit anchor and native teacher. Eligibility must be established by the data owner and accepted by the lead before fitting.
Pilot maximum: 16 successor packages, distinct-image-first. Minimum: 8 credible successor transitions across 4 images. Do not relax physical trust or treat unknown/unmatched as negatives to reach the floor.
Freeze an independent 32-image natural endpoint, excluding all training/admission, old exposed/fresh panels and the other task's current confirmation roles. Select up to 8 dense endpoint images for physical review before viewing arm outputs.
Report annotation-relative gained/retained/lost owners, geometric thresholds, physical re-entry/alias drift, invalid/malformed rows, output burden, EOS/caps. Original strict-repeat rule remains any-class native-pixel IoU > .95, once per later valid row; lower-IoU same-owner revisits are separate.
No automatic checkpoint promotion. A positive single-seed pilot remains bounded evidence; a valid null closes this implementation at this dose.

### Accepted supervision provenance

The lead verified the immutable `data-v2/supervision-bank.json` under the output root (file SHA256 `1f0a2c2b82e3bf696a7c9721db0d5b3c5ee34eee01f8f29ef669446c4428cd1c`): 16 physically admitted c/w transitions across 11 images. The original admission decisions affirm single-owner c absent from h and physically distinct nonduplicate w. Literal h+c and w IDs agree with their source records. Unknown or unmatched objects supply no negative targets.

The two original acquisition packets bind rollout producer fingerprint `024e46a512491b15d8715218c9fe7970707e7449f8b354b40ae37122c2c7ac9b`; fit anchor/native teacher fingerprint is `a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80`. v1 conflated these roles and is preserved as superseded. This bank supports adaptation from N16 on vetted older-anchor histories; it supplies no current-N16 on-policy provenance. Extent/occlusion caveats from the physical reviews remain attached. Bank acceptance does not freeze the remaining training recipe or authorize extra scientific arms.

### Frozen endpoint and common loss components

The primary readout is the paired change F minus S in distinct category-matched annotation owners at IoU 0.50 on the frozen 32-image endpoint. IoU 0.60/0.80, gained/retained/lost identities, strict repeats, malformed/invalid rows, visible burden and EOS/caps diagnose the nature and cost of that change. No post hoc scalar tradeoff weights are introduced. A coverage gain accompanied by greater repetition/caps or unresolved physical aliases is reported as a qualified tradeoff, not unqualified improvement in complete enumeration. Zero or negative primary change supplies no support for a coverage benefit at this fixed dose. The eight-image physical review limits what can be claimed about physical owners; annotation-unmatched detections remain unresolved until reviewed.

`evaluation/selection-v2.json` (file SHA256 `22471903a08d9f075f352703cf4bdfe79ee9e1419abe86fa74756fc4255c693d`) binds 32 independent images containing 280 annotations. Eight images were selected for physical review by annotation density with deterministic tie-breaking before any outputs. The lead replayed source, image and selection verification. Panel and dense-review ID digests are respectively `71747ff0f5de871baa8c9e5fcf38e1838f17e4fdd41dafa83acb9e96b6c00597` and `a91164e03bc4f3b5844af6c904daf5cfcfd1dea50a388606d6ae0e0b42de03f5`.

Before measuring the complete-update cost, freeze these common components: two admitted packages per optimizer update; token-mean entry-c CE averaged across the two packages (weight 1), plus token-mean post-c w CE averaged across the same two (weight 1), plus token-mean protected-position native-teacher KL averaged across all 54 original normal-reference records (weight 100). Regenerate teacher probabilities with the frozen N16 adapter at matching visible histories and explicit original protection positions; the historical reference adapter and margin scalars are not reused. No conditional-w KL or margin component. The reference corpus has 5,768 visible action tokens and 5,759 protected positions.

Use AdamW with learning rate 1e-5, betas (0.9, 0.999), epsilon 1e-8, weight decay 0, foreach false, gradient clipping at 1. One cyclic two-package schedule covers all 16 packages. The dose candidates 16/32/64 optimizer updates therefore provide 2/4/8 exposures per package to each CE component. These are pilot exposures, not the older 256-update N16 training dose. The immutable final execution packet still needs the protection binding, exact schedule/seed, measured cost and selected dose before scientific fitting.

## Cost and finite stages

One paired fit only. No quality-driven site, scale, dose or checkpoint sweep.
Initial technical path allocation ceiling: 2 GPU-hours. It must include real causal replay/gradient sensitivity, an optimizer update, saved-adapter reload and a cold natural continuation.
Before fitting, use measured complete update cost and measured endpoint cost to select the largest common dose in {16,32,64} that fits a total round ceiling of 48 allocated GPU-hours, including technical, teacher and endpoint work. Dose selection is cost-only and precedes any trained endpoint. If the minimum dose cannot fit, close fit readiness honestly rather than silently enlarging budget.
The 48 GPU-hours is an execution ceiling selected by the lead for the newly authorized finite round, not a claim of expected runtime or user acceptance of the earlier 24-hour proposal.
Freeze the final bank, schedule, seeds, tolerances, teacher mapping and dose in the execution packet before the first scientific optimizer update. Technical mutated adapters are disposable and never become fit anchors.

## Supporting content-use unit

Prepare independently; it does not gate the paired fit. Once the runtime is valid, use at most 4 preselected exposed image/owner cases with same-owner aliases where reviewed evidence exists. Compare correct feedback, exact self replay and one wrong-owner/norm-matched replacement at frozen boundaries; include a delayed decision if supported.
A selective predicted change supports use of this route. Generic corruption only shows dependence. A failed or inapplicable swap cannot disprove native owner state or veto the architecture trial.
Preserve a separate receipt and conclusion from the primary natural endpoint.

## Accepted first real slice

The lead inspected `technical/runtime-f-v2/train-receipt.json` (SHA256 `0807eb491dfb293f129beb87f20b82d032c02e509c7ac6fa637f17f3d73521cc`) and `technical/cold-f-v2/cold-receipt.json` (SHA256 `50818a332d3bbdaccef9233f11e4904afb74a677d8b0c736e92dda4c5c4d1a2b`). The 588 language DoRA tensors contain 18,006,016 trainable scalars. One disposable F update moved the adapter by L2 0.0423447. Later-target gradient at the final feedback source was 0.0386209 and became zero when only that source was detached. History-only first-layer K activation gradients remained nonzero under detached F (0.119353) and S (0.205394); generic parameter gradients are recorded separately.

Each training replay executed 57 model forwards, one image forward and 28 internal slots. Peak CUDA allocation was 66,422,482,944 bytes on an 80 GB A100. Saved adapter fingerprint `4682ce4ec8980dc59b82fc17e26fa8dcbba1f3d3e00df459886689dfb39660cf` is disposable and cannot become a fit anchor. A separate cold process loaded it and generated 16 visible tokens at a 16-token technical cap. That establishes persistence and short continuation, not the 3084-token endpoint cost. All 16 bank records were materialized with exact prompt/image/grid/media checks from the original train/dev source union.

The first attempt failed before model execution because the isolated worktree lacked ignored embedding-gate evidence. Its receipt is preserved; the loader now explicitly binds the passing read-only source gate in `/data/CoordExp/.worktrees/research-probes`. Total initial GPU0 process wall time, including that failure, was 64.349 seconds (0.017875 allocated GPU-hours). Runtime code SHA256 at this checkpoint is `94e9a2f472af2a1e60f06f215f72a2a362c9b2773918f98a7f85f983f1ef57f6`. Longer full-recipe memory, complete-update cost and full-budget natural decode remain open.

### Accepted teacher and full-budget decode cost

The native N16 teacher cache is lead-accepted: `teacher/cache-v1/manifest.json`, file SHA256 `6a8676f311e77c22a957ac623d723e9b08a3d94a299e6a6749991e5fdcba7640`, internal payload digest `99699c486d1741128fb5c32ecc063451e073a532d4643121d57bf418d3c92bcb`. These are different hash roles; the initially reported internal digest was not the file digest, and the file was unchanged. All 54 tensor-file hashes and original source/position bindings were verified. The longest 325-token record and the 76-position geometry-mask record also passed fresh tensor-content and normalization checks. Full vocabulary is 152,670; 5,759 positions are protected. GPU2 cache generation used 54 model/image forwards in 79.31 seconds with 9.23 GB peak allocation.

`technical/runtime-natural-cost-v2/cost-receipt.json` (SHA256 `382f1c8f358edfb813de88981c6831b29f2e5223dbfa04b20d867a568a879e92`) binds one original N16 load and identical empty-history input for the preselected exposed spoon image. S ended at 287 visible tokens, 29 slots and 23.814 synchronized seconds; F reached the 3,084-visible-token cap with 377 slots in 254.037 seconds. These are technical cost observations, not trained-arm quality evidence. Raw outputs are sealed separately. Cumulative GPU0 allocated process wall, including both preserved preallocation failures, is 359.349 seconds (0.099819 GPU-hours).

`decode-cost-projection-v1.json` reserves 16 GPU-hours for the 64-call endpoint, nine content generations, donor replays and loads. It projects the maximum permitted visible/slot counts at 1.5 times the larger observed average cost per model forward: 12.358 GPU-hours for the endpoint and 1.746 for content generations before rounded overhead allowance. This is a conservative planning estimate, not a proved latency bound. The remaining decision is complete-update feasibility/cost; no scientific fit or dose has been selected.

### Complete-recipe measurement checkpoint

Committed implementation: `8e0bb0ce3c5835bcf8fe13bcf7f232c001f0a143`; the lead freshly passed all 31 row-feedback tests before launch. `training/cost-packet-v1.json` (SHA256 `a7a0eb023a22fe26bf3d1ee7e0b6564d51ea2f435b02506e5cb7a096c0c30a80`) authorizes one disposable update per arm, each independently restarted from N16. The two selected packages maximize literal h+c+w length across distinct images, independent of outputs. They are image 219546 owner 702889 and image 25274 owner 1331684. Physical GPU1 is assigned to paired_training (Luna xhigh), S then F serially; the combined outer wall ceiling is 6,600 seconds. No scientific fit is authorized by this packet.

S completed with exit 0: `training/cost-S-v1/training-receipt.json` and its outer receipt record 58 replays, 1,560 model forwards, 58 image forwards, 724 internal slots, and all 5,759 protected positions. Its synchronized complete update took 369.647 seconds; outer process wall was 403.806 seconds. Peak CUDA allocation was 76,037,164,544 bytes, reservation 84,232,110,080 bytes. Reported loss 1.943998 is the declared sum of record-mean components, including KL weight 100 after averaging over 54. The lead inspected these exact receipts. F was then launched by the same owner and is pending acceptance at this checkpoint.

F subsequently completed with exit 0: `training/cost-F-v1/training-receipt.json`, SHA256 `e53bfa9dcd2032cb839c7708cc52dbd93b80e8ef79e668232ad40d6e7c84b4f6`. It has the same exact coverage and counters, 366.147 seconds per complete update, 400.658 seconds outer wall, and 76,036,441,600 bytes peak allocation. S receipt SHA256 is `b5401dc6b486ef9974cebe2fc5ac7c885dddd6b338ec1441d46b04b6760f9a05`. The lead freshly checked both complete component calculations, nonzero finite updates, 588 tensors / 18,006,016 scalars and exact cross-arm packet, bank, protection, teacher, schedule, materialization, anchor and code identity equality. These remain disposable technical updates, not the scientific fit.

Root acceptance is `training/serial-cost-acceptance-v1.json` (SHA256 `aebb269dcfb684ab8d170c8136943b68af24c7e67b7d360b324b6c69b4fc76f6`). Known measured spend is 0.349954956 GPU-hours, including 0.026673458 for teacher preparation; six earlier attempts have internal elapsed receipts but no outer launcher timing. Full stage caps, rather than these incomplete outer totals, support the dose budget. One old scalar resource counter reports 16 admitted packages as materialized; the exact materialization hash map shows that the cost processes prepared their two selected packages plus 54 normal records. This is a receipt counter correction, not a change to the verified loss exposure. Original receipts remain immutable; the new trainer will report the actual count.

After both cost receipts settle, inspect full identities, denominators, finite updates, memory and outer costs before freezing dose. Choose serial paired fits or a bounded data-parallel implementation from measured time; do not infer parallel efficiency. Any distributed extension must retain global component denominators and all prior-state gradients and pass a real complete-update equivalence check against these serial disposable updates before scaling. No new scientific axis is implied. Freeze the final code, topology, schedule and largest affordable dose before scientific fitting; restart both arms from original N16.

The measured serial path would take approximately 6.6 wall hours for 64 updates per arm. To reduce elapsed time, the lead authorized a bounded explicit distributed-gradient implementation: independent records are replayed/backpropagated on separate ranks, gradients are summed once before the common global clip and AdamW step. Global component weights stay 1/2, 1/2 and 100/54. No world-size averaging, altered histories, extra updates or changed target exposure. feedback_runtime (Sol xhigh) now solely owns training.py/tests for this extension; paired_training (Luna xhigh) retains cost/launcher evidence and has relinquished those write surfaces. CPU implementation is authorized first; GPU topology tests need a fresh root-bound packet.

Before observing distributed output, freeze the real complete-update equivalence gate for each arm against its serial saved adapter: exact global record/mask/input identity and counters; each component mean and total loss within absolute tolerance 1e-6 plus relative tolerance 1e-5; preclip global gradient norm relative error at most 1e-4; full adapter tensor L2 difference divided by serial update movement at most 1e-3; maximum absolute tensor difference at most 2e-6. All trainable tensors must be finite and all frozen base parameters unchanged. Do not loosen these thresholds after failure. Run two-rank equivalence before a four-rank cost slice, then select topology from actual cost and the existing 48 GPU-hour ceiling. All topology slices count in the initial two-GPU-hour technical ceiling.

The lead freezes the scientific dose at **64 optimizer updates per arm**, with eight exposures of each admitted package to each CE component. This is the largest registered candidate affordable on the already verified serial path: 1.5 times the sum of measured S/F update times, multiplied by 64, plus measured setup/exit overhead is 19.640 GPU-hours. Even reserving the full two-GPU-hour technical ceiling, full two-GPU-hour teacher ceiling and 16-GPU-hour endpoint/content allowance gives 39.640 GPU-hours, below 48. This choice precedes any distributed result or trained endpoint. Parallel execution may be selected only if its measured projected total also fits the same ceiling; otherwise retain the verified serial route at this same dose. No additional fit or dose sweep is authorized.

### Distributed gate outcome and final execution packet

The lead passed all 37 package tests and committed the bounded distributed implementation as `2eb4d108`. Training source SHA256 is `5a1123a05c002ff4c7bd3390d3d39d958f62ecf184d18e2c40cdd8b0fdf95559`. The real two-rank packet was `training/cost-dp2-packet-v1.json` (SHA256 `1c289f9790cc5c2b3c55a85b8a4f7ba9a0ab2a156fda873a1619ae36283c043e`), with both independent original-N16 cost starts on GPUs 0/1. Both processes exited zero with exact global coverage and finite saved adapters; total outer allocation was 0.316298178 GPU-hours.

Root's fixed comparison script and canonical adapter inspection produced `training/equivalence-dp2-S-v1.json` and `training/equivalence-dp2-F-v1.json`. S passed every gate; its complete update took 195.283 seconds versus 369.647 serial seconds. F passed loss/component means, gradient norm and global adapter L2 gates, but its maximum element difference was 2.47458228841424e-6, above the predeclared 2e-6 ceiling. The threshold is unchanged. The distributed implementation is not admitted, and no four-rank run or extra numerical variant will be launched. This technical result does not answer the scientific feedback question.

The lead retains the verified serial route at the same frozen dose. An AST comparison against the accepted serial commit confirms unchanged existing loss, update, loading and materialization functions; existing `run_training` changes only the admitted-versus-materialized resource counter, and the CLI adds an optional distributed branch. The formal packet is `training/fit-packet-v1.json` (SHA256 `6591ae6fa8fc2dc1ef4c48c60ea884ce8c2885421e00018ad67a712d01fcab8a`), status `root_frozen_ready_for_fit`; root admission passed. It binds 64 schedule pairs, 16 packages with eight exposures each, original N16, current code and frozen teacher/data/endpoint artifacts. S uses GPU0 and F uses GPU1 concurrently, one serial process each, with a ten-hour wall ceiling per arm and a twenty-GPU-hour paired fit cap. Stop on first failure and preserve both states; no automatic retry or checkpoint promotion. Full rounded technical, teacher, fit and endpoint/content ceilings sum to 40 GPU-hours within the overall 48-hour ceiling.

Automatic native-worker wake registration was attempted at this checkpoint but failed before arming because the current App rejected `thread/agent/observe` as unsupported. No monitor was established; the lead remained active and received the ordinary worker completion. Do not infer a durable handoff from that failed call.

The source-blind physical-review renderer at `evaluation/render_blind_review.py` is CPU accepted. The lead inspected source, all eight demo artifact bindings, proposal accounting and a rendered crop. Demo manifest SHA256 is `cee231fe01ccc9ada5c80c2a2614d1a8f3838118c68c10aa73d8bac759c0b56c`. Three synthetic proposals remain represented by two exact-geometry display groups; no approximate merging or review cap is applied. This is rendering verification only. Trained endpoint outputs and the actual eight-image physical review remain pending.

`evaluation/physical-review-tools-acceptance-v1.json` also binds the CPU-accepted aggregation consumer, SHA256 `9bb0762219f35c6cbbacc78de368ab4fb46a131290cf39012099b12324cc2342`. The lead reproduced an initial map-binding gap that could flip S/F recurrence counts while keeping the blind review unchanged. After correction, the consumer requires the original prepare-review manifest and rejects any changed queue/source-map hash; the same counterexample and the remaining coverage/alias/uncertainty fixtures pass fresh checks. Physical review is still pending; this acceptance supplies no model-quality result.

## Initial package and GPU ownership

- Runtime/training owner: feedback_runtime (Sol xhigh), probes/row_feedback/runtime.py and runtime tests; GPU0 for the initial real slice, GPU1 only after explicit root scheduling.
- Supervision owner: supervision_bank (Terra high), probes/row_feedback/data.py and data tests; CPU initially.
- Fresh native teacher cache: supervision_bank also owns probes/row_feedback/teacher.py and teacher/ outputs; GPU2 only, at most 2 allocated GPU-hours within the round ceiling. Its immutable input is data-v2/protection-records.json (SHA256 078576341fce4095f985f1431ff5e770b629925c9a47308b075d393640e5e165). No optimizer updates. The 9 excluded positions in the 54-record protection mask belong to the original geometry-invalid row in image 360573; protected normal images have no overlap with the 11 supervision images.
- Independent endpoint/consumer owner: active_evidence (Sol high), probes/row_feedback/evaluation.py and evaluation tests; CPU initially.
- Content-use owner: mechanism_options (Sol high), probes/row_feedback/content.py and content tests; CPU initially.
- Root owns initializer, records, execution packet and integration. No worker commits shared files or launches unassigned jobs.
- Serial training owner: paired_training (Luna xhigh), probes/row_feedback/training.py and training tests. It consumes the settled recipe and immutable schedule candidates (SHA256 60801a6ab55b6f38ee25f4b6272c08bee8269c95953357b65de4fad6160954e3); it has no GPU permission until the lead binds a cost or fit packet. Runtime and teacher ownership remain separate.

User routing correction: execution workers use Luna/Terra/Sol. The two Astra
advisers were interrupted after an erroneous execution reassignment; root found
no live row-feedback job, GPU allocation or runtime/data implementation to
transfer. Their earlier independent scientific advice remains evidence only.
