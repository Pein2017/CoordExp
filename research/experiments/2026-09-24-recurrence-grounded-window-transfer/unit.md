# Independent first-revisit and progress windows

CPU-only source qualification, 2026-09-24. Lead task 01a0c831-b332-7e12-b931-6ebe2359c99f owns selection, physical judgments and acceptance. Existing 923-worker 01a0ce4b-9b55-7392-8a25-6a76f9e12c3a remains gpt-6-sol/xhigh. Autonomous continuation is user-authorized; this finite assignment permits no language-model load, CUDA, model/vision forward or generation. Elapsed time is not a stop criterion.

## Decision and evidence boundary

The [accepted older-record result](../2026-09-24-recurrence-older-record-routing/results.md) closes the book-Q branch: older content changes a free numerical output, but the paired prediction fails and Q remains physically UNKNOWN. The [first book revisit](../2026-09-24-recurrence-book-first-revisit/results.md) shows that blocking latest-record reading disrupts progress while leaving the first B revisit at B. We need independent, visually defensible native windows before deciding whether that asymmetry transfers. This screen is preparation, not another mechanism test or a population-rate estimate.

Qualify up to two first-revisit candidates with adjacent owner proxies A,B,B and two progress candidates A,B,C. Proxies only nominate visual review; annotation identity and absence of overlap do not prove physical identity or novelty. D means local progress candidate, not healthy image. Do not reopen Q, tune donors/positions, or launch a masking experiment here.

## Frozen source and deterministic screen

Use only the existing 273 unique split/image sources, mature145 event order then prospective128 runtime order. Bind these read-only records:

- [Original selection](../2026-09-23-recurrence-first-arrivals/selection.json), SHA256 `435c333520f5f31078a44d2afb49d07a31c986d07198f9c8140cd8f3ddf47395`.
- [All-source numerical registry](../2026-09-23-recurrence-native-x1-phase/supporting/cross-image-candidate-registry.json), SHA256 `7bf578d4484d4ebf344aa6933f2c8f456256f2327aa69daa5bcecc7e42e166e3`. Reuse its source/provenance crosswalk, never its numerical R/D labels as physical labels.
- [Older-record acceptance](../2026-09-24-recurrence-older-record-routing/lead-acceptance-v1.json), SHA256 `35897605db8f93a1278fce22e2cdf752c40f969f1ec2f85055b7d5bbe20b6819`.

Retain all273 identities in accounting. Exclude only the already intensively probed train313465, train151704 and train477415 from new picks, explicitly recording that reason. Carry every existing row-specific physical ruling/HOLD without treating it as an image-wide label. A binding mismatch is HOLD, not permission to replace a source.

For each remaining identity, in native row order:

1. Use maintained canonical row parsing and original processed annotation objects. Bind the processed input record and its coordinate projection. Compute unrounded `src.data.geometry.iou_xyxy` in the existing coordinate-bin space using `parse_source_bbox_tokens` for annotations; no pixel-IoU substitution. Reuse `recurrence_first_arrivals.prepare._load`/source bindings where applicable, but recompute the stricter eligibility below rather than inheriting its 0.5 proxy label.
2. Consider only three consecutive complete, valid-geometry canonical rows, contiguous in raw tokens and with identical description-token class. Each must have exactly one same-class annotation at IoU >= 0.75; all other same-class annotation IoUs must be < 0.25. Keep annotation IDs, boxes, raw descriptions and available crowd/group flags. An explicit crowd/group annotation is ineligible as a singleton proxy; absent flags remain unknown, not proof of singleton status. Null/no eligible annotation is not a bad-prediction label.
3. R requires proxies A,B,B with A != B. No parsed earlier box, regardless of predicted class, before the first of those two B rows may overlap B's annotation at IoU >= 0.1. D requires distinct A,B,C and the same earlier-prefix guard independently before B and C. Any unparsed/malformed object content in a tested prior prefix makes its chronology HOLD. Ordinary known serialization wrappers are not object evidence; bind the maintained parser's treatment. These conservative guards cannot rule out a physical visit with a substantially different extent.
4. Retain the earliest eligible R and earliest eligible D per image, plus explicit reasons if absent. Choose the first two R images in the frozen273 order, then the first two D images excluding chosen R images. No downstream score, intervention effect, current quality judgment or visual adjudication enters that ranking. No visual-based replacement or backfill; if fewer qualify, return fewer. Never relax thresholds or search additional sources in this assignment.

Record candidate counts and chronology HOLDs, not just selected successes. If an earlier possible physical visit contradicts a selected proxy chronology on visual review, keep the selected case and report HOLD; do not silently skip it.

## CPU deliverable and checks

Write `supporting/cpu-candidate-registry.json` and `supporting/cpu-feasibility.md`. The registry accounts for all273 source identities, deterministic eligibility/priority and existing physical rulings. For each selected window bind original raw/trace/receipt/image, effective untied model identity, annotation input/projection and original full-batch request identities. Reconcile every selected row token with native trace, prefix offsets, immediate previous row, stop/censor status and earlier possible owner visits.

Render at most four selected-image full/zoom overlays, showing fixed three-row boxes and annotation proxies; include any earlier ambiguity needed for chronology. Worker physical descriptions are proposals for lead review. If no candidates qualify, return that finite result without alternate search.

For selected cases only, processor/config-only preparation may check native prefix/left-padding/EOS-pad/three-axis geometry and original companions for the two later rows. Report the smallest reusable free-row native/sham/latest-record-mask route and its known first-opener information path, with finite call/artifact planning if meaningful. Do not implement that GPU producer or freeze a scientific effect criterion. Insufficient physical support is a valid CPU outcome.

Use a small maintained CPU helper only if needed for reproducibility, reusing existing loaders/parser/source checks. Its executable selfcheck must reject an earlier proxy visit, ambiguous annotation match, malformed prior object, raw gap/class mismatch and incorrect R/D priority. A separate CPU readback must reproduce selected identities and counts from bound native sources. No generic registry, framework or source recensus beyond this fixed273 is needed.

## Ownership and stop

Cwd `/data/CoordExp/.worktrees/research-probes`. Worker writes only this unit's `supporting/`, optional small CPU helper under `probes/training_set_completion/recurrence_grounded_window_transfer/`, and selected overlays under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-grounded-window-transfer/cpu-review/`. Lead owns unit/state/catalog/index and acceptance. No Git, predecessor, peer, shared-runtime or user model-setting edits.

Return one self-contained CPU candidate with paths/hashes, exact candidate/HOLD counts, selection and physical ambiguities, producer reuse points and zero model/vision/GPU accounting. Stop after that return; no GPU admission, replacements, retries of model work or successor are implied. Send directly to lead task 01a0c831-b332-7e12-b931-6ebe2359c99f through the existing `worker_turn.py --to lead` route, preserving both tasks' settings.
