# Next-stage direction and bounded preparation

Post-acceptance refinement, 2026-09-22: the 32-image package is now closed.
Keep candidate-bound v2 preparation and all selected data immutable. Prepare
`next-stage-preparation-v3.md` and new v3 JSON records for the choices below.
No model/GPU launch is released by this refinement.

- Propose one fresh nominal seed1729 trajectory with a **32-epoch ceiling**;
  retain cheap checkpoints at epochs1/2/4/16/32. Four epochs are diagnostic,
  not an underfitting rejection. The successful32-image run had about273
  equivalent panel passes, so neither4 nor32 larger-panel epochs are claimed
  dose-equivalent or guaranteed sufficient. No LR or architecture search.
- Freeze a96-image training sentinel: all retained32 plus64 additional images
  selected by metadata/hash before outcomes (24 ordinary,24 dense same-class,
  8 dense other,8 middle-density). Propose full source and epoch32 evaluation
  on all1024 training and256 validation images, plus sentinel-only native
  evaluation at epochs4 and16: **2752 analytical cells**, rather than5120.
  Epoch32 is the planned definitive endpoint; sentinel results do not select
  a lucky full-panel checkpoint. Verify whether the exact32 previous source
  cells can be reused under identical requests/source/runtime/policy. Reuse
  them if eligible and separately count reused cells versus new model work.
- Refine the operational guardrail proposal to at most5% newly bad images,
  1% newly capped images,5% newly annotation-owner-recurrent images, and1%
  newly severe-run images, separately on the1024/256 panels. Use floor(f*N):
  training51/10/51/10 and validation12/2/12/2. These are prospective operational
  tolerances for the user's "not worsen too much", not statistical guarantees.
  Keep the v2 bad-image/severe-run definitions and exclude UNKNOWN. The matching
  net-incidence limits are redundant and need not be another gate. Retain
  paired malformed-span and run-length severity for existing source failures;
  numerical incidence compliance alone is not automatic lead acceptance.
- Resolve the **actual CPU packing plan** and deterministic rank/epoch/checkpoint
  schedule, including tails, omitted/repeated images and exposure counts. Exact
  once-per-epoch presentation must be demonstrated or any unavoidable padding
  excess explicitly proposed; do not claim exact epochs from480 estimated packs.
  Reuse the maintained packing/cache entry. This authorizes a bounded1024-image
  CPU packing preparation, not a GPU feature cache or model forward. If the
  maintained entry cannot do this without a model call, return that specific
  dependency rather than silently allocating a GPU.
- Recompute the complete training/decode/tail/reserve forecast for this plan,
  proposing the unchanged **new-stage8-hour/64-GPU-hour envelope**. Use all8 GPUs
  productively through training/source-evaluation overlap and a work-conserving
  queue. Fixed cells and contrasts do not require idle/static worker buckets.
  Preserve incomplete cells as explicit HOLD if a cap binds; no altered decoder
  cap, denominator, architecture or automatic budget extension.

These refinements supersede the4-epoch/5120-cell proposal below; they do not
change the user's primary training-fit criterion. Return a concrete v3 packet
with the existing1024/256 identities, actual packing/exposure, config, queues,
guardrails, source-reuse decision and conservative budget. The lead will freeze
the executable successor contract after reviewing these remaining facts. Do not
rerun the closed pilot or edit its immutable candidate records.

User clarification, 2026-09-22: validation degradation is foreseeable; prioritize
training-set learning, while preventing excessive format/duplication regression.
This supersedes the earlier lead emphasis on validation coverage improvement as
the next stage's primary success condition. It changes the prospective proposal,
not any completed result or the running package's frozen selection.

2026-09-22. The user requests an intermediate data scale after the current
feasibility package passes, before full-data production. The lead chooses one
proposed scale: **1024 training images and 256 validation images**. Do not run a
512/1024 by 128/256 grid. This document authorizes preparation only; the current
monitor and seed-2718 repeat retain priority and their existing contract.

Persistent pair: lead 01a0c1f3-dbef-7b63-b2da-8dc7072cea8d; worker 922-worker
01a0c726-ad7c-7cc0-89b7-d76ac6fcf027, gpt-6-astra/low. Use the existing direct
return route and maintain the user-selected settings. Cwd is
`/data/CoordExp/.worktrees/research-probes`.

Prepare a concise proposal using the current canonical asset index
[research/assets.md](../../assets.md), especially `coco12k-geo-sorted-xy-v1` and
the same composed mature untied source. The former docs/RESEARCH_ASSETS.md path
has moved. Preserve the current package's admission, selection and results.

Proposed scientific question: can this fixed recipe learn natural enumeration
across a substantially larger training panel, while keeping format and
duplication degradation within explicit limits? Training-set natural coverage,
coordinate fidelity, valid output and completion are primary; lower teacher
loss alone is insufficient. Validation coverage/CE degradation is recorded
separately and does not by itself reject the recipe. Current 32-image feasibility
remains established. This is a recipe-scale fitting test, not an
address-component causal comparison or a novel-mechanism claim.

Preparation output:

- Propose one reproducible 1024/256 split from compatible existing assets, using
  metadata/annotations rather than intervention outcomes. Include ordinary and
  dense same-class examples and report density/class/length distributions.
  Prefer keeping the current corrected 32 in training as a separately reported
  regression group, with 992 additional images; preserve annotation precedence.
  Exclude current fit/monitor exposure from a fresh 256-image validation panel
  and disclose all known earlier checkpoint/SFT exposure. Validate disjointness
  with canonical image identity and available content hashes, not bare row IDs.
- Keep the architecture, supervised ordering, loss and maintained entry unchanged.
  Propose a fresh start from the same mature live source, not the 32-image
  overfit checkpoint. Use the current selected optimization recipe as the
  starting proposal; do not launch LR searches or architecture ablations.
- Express training exposure in epochs and actual examples/packs/tokens; do not
  blindly carry over 512 updates from the 32-image fit. Prepare fixed 1/2/4-epoch
  checkpoint costs for one trajectory, including source and validation decode,
  and distinguish insufficient training fit from validation regression.
  Estimate from existing measured throughput and disclose extrapolation limits.
- Propose fixed native evaluation and stop/selection rules before any next-stage
  outcome. Rank eligible checkpoints primarily by training-panel natural
  coverage/completion and coordinate quality, subject to explicit format and
  duplication guardrails on both training and validation. Report paired
  source-versus-trained bad-image incidence, recurrence/run length, malformed
  output, geometry and EOS/cap as well as aggregate counts. Propose concrete
  tolerances for lead review before launch; do not invent post-outcome cutoffs
  or a zero-regression requirement. Validation coverage/CE alone must not select
  settings or veto stronger training fit. UNKNOWN remains separately reported;
  incomplete annotations cannot make unmatched predictions verified false
  positives. No held-out improvement threshold is requested at this stage.

Worker-owned preparation surfaces: a new `next-stage-preparation-v3.md` in
this current research unit, and JSON/JSONL manifests/config proposals under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/`.
Use ordinary maintained helpers; no new training infrastructure is requested.
Do not edit lead-owned state/frontier/catalog or create another active research
unit. Source records and configs must obey the output storage policy.

CPU preparation only: no new GPU allocation, model/vision forwards, fitting,
scientific evaluation or giant cache pass. Reuse existing measurements. Return
the proposal, exact source/split bindings, compatibility checks, estimated
envelope and any material choice with the current package's final candidate.
Do not delay its existing milestone reports or closure. The lead will reconcile
the completed package and freeze the next-stage launch; no new user research
direction is needed merely to prepare this requested scale. No successor launch,
full-data production, publication or self-acceptance is authorized by this note.
