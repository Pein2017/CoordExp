# Five-route work sample v2: exact native action replay and coordinate credit

Frozen by root after the user's harder-task request. This replaces the earlier CPU recount benchmark; do not
implement that superseded task. Root dispatches the identical frozen brief to
Sol-medium/high/xhigh and Astra-low/medium. Only candidate slug and assigned
physical GPU differ. Root owns acceptance and pricing; candidates must not
inspect other candidates, root acceptance code or benchmark telemetry.

## One practical component

Implement a small native replay component that returns raw chosen-token action
log-probabilities and applies **supplied synthetic credit to coordinate tokens
only**. Prove the same component works on ragged mixed-image batches and exact
singletons. This is scoring/masking qualification, not a GT reward, new training
objective, architecture, trainer or framework. No optimizer construction/step,
parameter update, new generation, extra images, network or new dependencies.

Cwd: `/data/CoordExp/.worktrees/self-rollout-behavior`.
UNIT: `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-a-owner-visual-followup`.
Write only `UNIT/benchmark/candidates/SLUG/` and
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a-owner-followup-v1/benchmark/SLUG/`.
Use apply_patch and bare python in the configured environment. No helpers or delegates.

Bound:35 minutes engineering/verification and one real qualification invocation
using only the assigned A100, at most5 minutes from model-open through GPU
release. Report initial completion separately from any root-requested bundled
correction. No automatic GPU retry or qualification expansion. If the real seam
cannot fit without core changes, return the concrete limitation and proposed
narrower seam; do not silently substitute a mock, singleton loop for batching,
detached-only gradient check, or longer budget.

## Frozen four cells and source identity

Read the completed A directory:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a/`.

- `baselines.jsonl` SHA256
  `0117a01b7d2b4b28489d18b6d71120af5f77053221cd732eaff217d1c177c077`.
- `branches.jsonl` SHA256
  `970446748b6b3c73937b3f09a06acf605f85e7da10d56185399ff47435a4cb7c`.
- Select **only**, in this order: image`coco2017_train_000000081589` branches0,1;
  image`coco2017_train_000000106497` branches0,1. No outcome-based selection.
- Native prompt lengths1320,1320,1362,1362; exact retained prefix lengths
  19,19,29,29; action lengths10,10,9,10. Validate observed values, not fabricated
  padding examples. Use mixed-image batch pairs `(cell0,cell2)` and `(cell1,cell3)`.
- Source is original step2444 through old-worktree
  `configs/coordexp_swift/infer/source256-policy-source-v1.yaml`, native FP32/SDPA,
  original unmerged DoRA and selected embedding delta. Bind backend/payload
  receipts and source hashes, not a claimed equivalent model.
- The shared `input-v1/source16.jsonl` is an audit serialization with absolute
  media refs, **not native-loadable**. Reuse the A producer's validated canonical
  raw_train256 selection route: frozen image IDs, original-row hashes and media
  hashes. Do not edit or rewrite that panel.

Old worktree below means `/data/CoordExp/.worktrees/dora-prox-linear-n2`.
It and the completed A producer are read-only dependencies; hash every directly
reused helper/config and retain their absolute paths in the receipt.

## Behavioral contract

1. Validate full bank hashes and each selected cell's request/media/grid,
   source, GT, exact prefix and prefix+action+suffix identities. Construct real
   native multimodal inputs; no decoded/re-tokenized replacement of the retained
   prefix/action. Preserve padding masks, image ordering and Qwen MRoPE. Scoring
   inputs stop after the action; suffix tokens are identity-checked but not
   conditioning evidence for the action.
2. Return raw FP32 chosen-token log-probabilities for **every action token**.
   An action token at full-input position`t` is scored from logit position`t-1`,
   with padding offsets handled explicitly. No RP1.10 transform, temperature,
   top-p or sampling likelihood substitution. The A action sampler was raw
   T1/top-p1/top-k0/RP1; retained action scores provide an additional provenance
   reference, not a substitute for fresh native replay.
3. Derive exactly four supervised positions per action from its actual parsed
   coordinate spans and literal token identity. Prefix coordinates, description,
   schema markers, suffix, EOS and padding are excluded. Persist action-local,
   unpadded full-input and batched tensor/logit positions, token IDs and mask.
   Do not implement “all coordinate tokens anywhere” or “last four positions.”
4. Credits, in the frozen cell order, are **synthetic** `[1.0,-0.5,0.0,2.0]`.
   Test exactly

       L = -(1/4) * sum_cells credit[cell] * sum_action_coords raw_logprob[token].

   The sum within a cell is not token-mean CE. Credits are external constants;
   never read owner returns or visual judgments to assign them. This formula is
   an engineering fixture, not adoption of a scientific objective.

## One production-shaped qualification

- Score all four cells as singletons using an existing independent native
  teacher-forced path; score the two specified mixed-image B2 batches through
  the candidate component. Disclose real forward counts and actual batch shapes.
  All chosen action log-probabilities and the weighted loss must agree with the
  singleton reference at fixed `atol=2e-4, rtol=2e-4`. Report maxima; do not relax
  tolerances after observing outputs.
- On retained **real replay logits**, check direct loss gradients: all excluded
  action-logit rows and the zero-credit cell are exactly zero; at least one
  positive/nonzero-credit coordinate row is nonzero. Compare batched versus
  singleton direct-logit gradients under the same loss normalization/tolerance.
- Also traverse the real native model autograd path on one fixed singleton,
  cell0: a nonzero-credit coordinate loss must produce finite, nonzero aggregate
  gradient on the existing language DoRA trainables; its zero-credit version
  must produce exactly zero gradients. Reuse one graph if convenient. Keep
  model eval/dropout behavior fixed and take no optimizer step. Bind the named
  gradient surface and show the DoRA parameter bytes did not change.
- **Do not assert prefix/description hidden states or shared parameters have
  zero gradients.** They can influence later coordinates. Zero direct
  supervision at their token-logit rows is the intended masking invariant.
- Demonstrate equivalent falsification with three small corruptions of these
  real fixtures: shift the coordinate mask by one token; include a nonzero-credit
  description token; swap/mismatch the two images' request/grid/media identity.
  Each must be rejected by the consumer/contract check, not merely logged.
  Tests may be CPU-only once real native evidence is captured; mocks alone do
  not establish the scoring or batching path.

Persist a compact versioned artifact containing input/dependency hashes,
literal action/position identities, raw scores, masks, credits, loss, direct
gradient evidence, model-gradient norms/hashes and unchanged-parameter evidence.
Save enough actual selected-logit/gradient data to permit independent loss/mask
checking; avoid full-prompt vocabulary dumps. A **fresh CPU process** must reload
and verify the artifact, expected positions/loss/gradients and corruption
rejections without trusting only stored PASS booleans. No second model load is
required for this cold consumer. Publish its receipt, exact PID/command/exit,
elapsed/peak memory/forward counts, tests and a short `result.md`.

Deliver `candidate` with documented real-run and CPU-consumer commands, not
`lead-accepted`. Root replays the consumer and owns one bounded correction and
final comparison. Stop after this component and qualification; omit all64
owner recount, visual labeling, reward selection and speculative hardening.

## Named feasibility evidence (read-only; not a pre-implemented solution)

- `src/inference/hf_backend.py`: `_materialize_native_inputs` accepts multiple
  requests and validates expected prompt/media/grid identity;
  `teacher_forced_evidence` supplies exact-history raw singleton evidence with
  explicit Qwen positions. `teacher_forced_chosen_token_logprobs` explicitly
  rejects batch size>1, so it is a reference, **not existing batch support**.
  SHA256:`53c2b68bfc65f91fa1848319f6171af6dd78776a270eac0884dfe5c97d461cef`.
- `scripts/research/train_source256_ce_rloo_round.py`:
  `_TrajectoryScorer.forward` is an existing differentiable singleton action
  replay with causal slicing; reuse its native mechanism without running its
  optimizer/controller. SHA256:
  `7742349c34e2fb17b3b45a39fb54041eb3b78784a5d04ee35de99a28c892e6dd`.
- `scripts/research/run_fixed_encoding_object_centered_spatial_eligibility_crossover.py`:
  `derive_explicit_position_ids` delegates ordinary2D masks and image grids to
  Qwen `get_rope_index`; it does not impose a singleton-only shape.
  SHA256:`3d61becf1672f0b2cfcbf15d531a498621b1aad067f3d6d55444033546ca793b`.
- Existing `_open_session`, `_build_request`, `_append_exact_prefix` and the
  completed A `run_a.py` provide source/request/literal-prefix patterns.
  Four real cells were inspected read-only to confirm the ragged lengths above.
  Native batching/gradient tolerance at this exact seam is deliberately what
  candidates must qualify; this draft claims feasibility, not prior acceptance.
