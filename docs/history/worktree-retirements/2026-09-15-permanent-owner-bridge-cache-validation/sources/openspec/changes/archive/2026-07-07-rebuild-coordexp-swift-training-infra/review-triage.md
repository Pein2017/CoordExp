# Review Triage

Date: 2026-06-29

Update: 2026-06-30 after critical review of
`agent_verdict/coordexp-swift-claude.md` and
`agent_verdict/coordexp-swift-codex.md`.

Scope: read-only review of the OpenSpec baseline for
`rebuild-coordexp-swift-training-infra`.

Reviewed artifacts:

- `openspec/config.yaml`
- `proposal.md`
- `design.md`
- `tasks.md`
- `specs/**/spec.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- `docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`
- `agent_verdict/coordexp-swift-codex.md`
- `agent_verdict/coordexp-swift-claude.md`

Review lanes:

- Contract/spec auditor
- Qwen/upstream tracer
- Training/loss/runtime auditor
- Artifact/config/smoke auditor

## P0

The first 2026-06-29 review loop reported no P0 findings. The later
2026-06-30 adversarial review classified the MRoPE/FA2 wording and loss
authority gaps as P0-class. Those findings were accepted in substance and
patched in the active OpenSpec specs and roadmap.

## P1 Accepted And Patched

- Pre-backward scalar non-finite handling now requires an all-rank reduced
  decision before any rank calls backward, and unsafe scalar state skips
  backward plus optimizer update on all ranks.
- Effective-batch derivation now fails fast for non-divisible effective batch,
  effective batch smaller than world size, backend accumulation conflicts, and
  incomplete final optimizer-step windows.
- The five-step smoke schedule is aligned across docs and OpenSpec: two
  scheduled `eval.forward` runs, normally explicit `eval.forward.steps: [2, 4]`,
  plus the required final checkpoint at planned step 5. The canonical
  `resolved_step_schedule.json` artifact and event fields are now specified.
- Follow-up audit found `tasks.md` had reported completed planning/review work
  as incomplete. The completed OpenSpec draft, validation, and read-only review
  tasks were then marked complete while later source-study, implementation, and
  smoke tasks were still in flight.
- Follow-up audit found no-resize image compatibility under-specified. The
  data/template/encoding spec now requires processor-derived admissible
  dimensions, explicit raw-pixel and merged-visual-token caps, and typed
  pre-processor/pre-forward validation diagnostics.
- Later cross-agent review found the MRoPE wording could be read as whole-pack
  upstream-helper parity. The packing/forward spec now requires per-segment
  4-row `[text,t,h,w]` Qwen position inputs, reset points matching
  `PackedSegment` boundaries and FA2 cumulative sequence lengths, and explicit
  varlen branch evidence.
- Later cross-agent review found loss semantics weaker than the approved
  decisions. The supervision/loss spec now requires same-segment causal shifts,
  exact group-mass `TokenTypeGateLoss`, explicit protected loss weights,
  resolved token types for every V1 atom, and `segment_balanced` planned-step
  normalization.
- Later cross-agent review found the final-tail policy contradicted the
  approved deterministic tail-fill decision. The config/runtime spec now
  forbids silent final-pack drop and smaller partial final updates, requiring a
  bounded tail-fill receipt for epoch-led runs.
- Later cross-agent review found Qwen no-resize/MRoPE/FA2 studies were not
  first-class tasks. `tasks.md` now includes those source-study/probe tasks and
  a legacy correctness-invariant inventory before old `src/` archival.
- Later cross-agent review found config inheritance and eval data binding were
  weaker than the blueprint. The config/runtime and training/artifact specs now
  define inheritance merge semantics, eval.forward data-source binding, and
  stronger manifest key requirements.
- Later cross-agent review found the vertical smoke was too degenerate to
  exercise accumulation, multi-segment packing, and denominator behavior. The
  smoke spec and roadmap now require at least two fixture examples, default
  single-rank `training.effective_batch_size: 2`, a multi-segment forward
  receipt, and smoke-adjacent checks for MRoPE reset, FA2 split, same-segment
  causal mapping, and `segment_balanced` behavior.

## P2 Accepted And Patched

- Cadence config fields, forbidden aliases, `resolved_step_schedule.json`,
  event lists, fractional milestone rules, dedupe semantics, and minimum event
  fields were promoted into the config/runtime spec.
- Smoke fixture source pinning now covers the preferred exactly-two-object
  source-row path, not only reduced larger rows.
- Earlier non-local cross-review claims in `BLUEPRINT.md` were reworded as
  non-authoritative context; local verdict files remain the auditable evidence.
- Qwen3-VL MRoPE row-shape expectations were made explicit in the packing and
  forward spec.
- Placeholder/grid validation now includes the expanded image-token formula
  based on `image_grid_thw` and `merge_size`.
- DeepSpeed first-smoke status now uses the canonical label vocabulary and
  forbids premature `systems_smoke_verified` or `production_supported`.
- Best-checkpoint selection now excludes update-skipped or unsafe non-finite
  checkpoints by default.
- Resolved-config provenance and run-manifest path/identity requirements were
  strengthened.
- Loss-continuity language now explicitly prevents V1
  `BaseTokenCE + TokenTypeGateLoss` from being interpreted as old production
  coordinate soft-CE or object/role-balanced objective parity.
- Artifact contracts now define minimal required key sets for the run
  manifest, metric events, eval-forward summaries, and checkpoint metadata.
- Random object ordering now requires a run-controlled seed source that is
  reproducible from resolved config or run artifacts.
- The Qwen token identity preflight now includes the `<|im_end|>\n`
  two-token boundary.
- The DoRA source gate now makes public naming and DoRA magnitude-vector
  persistence part of the study/probe, and the special-token embedding gate now
  requires additive-versus-absolute checkpoint semantics.

## Findings Not Adopted As Patches

- A standard-LoRA or base-only pre-smoke was recommended as a de-risking step by
  one reviewer. This was not adopted because the accepted user decision keeps
  the first adapter-enabled acceptance smoke on the DoRA path after its source
  gate. The roadmap now rejects the old `dlora` spelling rather than inventing
  a separate mechanism.
- The optional suggestion to shrink the final module topology was not adopted
  in this pass. The topology remains domain-aligned, but each implementation
  wave must still create a wave-specific Superpowers plan and avoid placeholder
  framework code.

## Wrong Accepted And Patched

- The active data/template spec no longer supports legacy `sorted` object
  ordering. V1 supports `source_order` and deterministic `random`; future
  geometric sorting is reserved under a deliberate name such as
  `geometry_sorted`. The implementation task wording now matches this
  requirement.
- Smoke source wording now uses the current `len12000` source-family naming,
  not the older max-length spelling.

## Duplicate

No material duplicate requirements were reported.

## Stop State

The OpenSpec baseline has no unresolved P0 or P1 review findings after the
accepted patches. This change is ready for user approval as a planning/spec
baseline. It is not approved for source implementation.

Remaining implementation gates:

- DoRA source study and round-trip probe;
- special-token embedding mechanism source study;
- Qwen3-VL processor, MRoPE, no-resize, placeholder/grid, and FlashAttention
  source probes;
- DeepSpeed systems smoke before production support;
- explicit user authorization before source implementation begins.

## Wave 1B Source-Gate Review Round

After the DoRA, special-token embedding, Qwen processor/forward, and FA2 probe
receipts were added, four read-only lanes reviewed task 2.10. They found no P0
issues. Accepted findings and resolutions:

- **P1 accepted and patched:** task 2.7 was overchecked from a thin Qwen MRoPE
  receipt. The Qwen probe now records per-segment 4-row position summaries,
  segment boundaries, reset checks against FA2 splits, and a whole-pack
  `get_rope_index` contrast showing the second segment would start at text
  position 18 instead of 0.
- **P1 accepted and patched:** FA2 segment isolation was proven by a standalone
  utility probe but not attached to the actual Qwen forward path. The Qwen
  forward probe now loads with `attn_implementation="flash_attention_2"`, passes
  explicit packed `cu_seq_lens_q/k` and `max_length_q/k`, monkeypatches the
  installed FA2 import, and records a Qwen-routed `padding_free_varlen` text
  call matching `[0, 21, 43]`.
- **P1 accepted and patched:** the special-token probe did not prove fresh-base
  reload. It now validates metadata, reloads a fresh base model, installs a new
  wrapper pair, loads `shared_embed_delta`, and verifies selected/non-selected
  input and output behavior.
- **P1 accepted and patched:** the roadmap used a singular non-existent
  `special_token_embedding_roundtrip.py` path. It now uses the canonical
  `special_token_embeddings_roundtrip.py` spelling.
- **P1 accepted and patched:** `docs/ARTIFACTS.md` described legacy
  ceil-rounded accumulation without scoping. It now distinguishes historical
  compact-detection artifacts from the CoordExp-Swift fail-fast
  `training.effective_batch_size` contract.
- **P1 clarified:** the legacy correctness-invariant inventory exists as
  `source-studies/legacy-invariant-inventory.md`; it remains a required input
  to task 2.10 review and is now described as the evidence for checked task 2.9.
- **P2 accepted and patched:** probe receipts were too thin as standalone
  evidence. Qwen and special-token receipts now include provenance and explicit
  assertion booleans.
- **P2 accepted and patched:** CPU and CUDA FA2 runs previously overwrote one
  receipt path. They now have distinct per-run receipt directories.
- **P2 accepted and patched:** generated probe `__pycache__` files were removed.

Focused re-review found no unresolved P0/P1 findings. One contract-auditor lane
reported `2.10 can be checked`; one upstream/probe lane reported `probe gaps
resolved`. Task 2.10 is now complete as a source-study/probe/invariant review
gate.

This does not authorize production `src/` implementation by itself. The next
implementation wave still requires the approved OpenSpec/Superpowers execution
roadmap, explicit handling of the old `src/` archive/skeleton step, and the
five-step vertical smoke before V1 acceptance.

## Section 3 Smoke Fixture Review Round

Section 3 fixture pinning/materialization completed after the DoRA naming
correction and source-gate probe round. The permanent fixture now lives under
`tests/fixtures/smoke/qwen3_vl_single_image_pack/` and contains:

- `examples.jsonl` with two canonical `RawExample` records;
- two copied real COCO-derived images under `images/train2017/`;
- `checksums.json` with source row hashes, source row numbers, image hashes,
  copied fixture paths, selected object ids, and rationale;
- self-contained `config.yaml` using `adapter.type: dora`,
  `packing.global_max_length: 12000`, `training.effective_batch_size: 2`,
  `training.max_steps: 5`, and `eval.forward.steps: [2, 4]`;
- a contract-oriented README.

Source rows were selected from
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`:

- row 3 -> `coco2017_train_000000000030__smoke2obj`;
- row 5 -> `coco2017_train_000000000036__smoke2obj`.

Both selected rows are single-image, exactly two-object,
distinct-description, description-safe, existing-image, and
no-resize-compatible with 32-aligned decoded dimensions. No object reduction
was applied.

A read-only contract-auditor lane found no P0/P1 blockers for tasks 3.1-3.4.
Its one accepted P2 finding was stale singular wording in `DECISIONS.md` and
`BLUEPRINT.md`; those docs now describe two fixture examples/images and clarify
that `expected_rendered.json` is deferred until the real renderer path exists.

Local verification after patching:

- structured fixture validation passed for JSONL/YAML parsing, canonical
  `RawExample` shape, local image paths, SHA-256 equality, decoded dimensions,
  32-alignment, two examples, and two objects per example;
- stale wording sweep passed for `one tiny`, `one image/example`, and
  `effective_batch_size: 1`;
- `expected_rendered.json` is intentionally absent;
- `openspec validate rebuild-coordexp-swift-training-infra --strict` passed;
- `git diff --check` passed.

Tasks 3.1 through 3.4 are checked. Task 3.5 remains intentionally pending until
the real renderer can generate and freeze `expected_rendered.json`.

## Section 4 Approval Gate Packet

The next pending implementation task is 4.1:

```text
Move old active src/ intact to reference/legacy_src/ after explicit
implementation approval.
```

## 2026-07-01 Implementation Verdict Review Follow-Up

After implementation kickoff, the agent verdicts in
`agent_verdict/coordexp-swift-claude.md` and
`agent_verdict/coordexp-swift-codex.md` were reviewed against the live rebuilt
`src/` tree. Accepted findings patched in this follow-up:

- **Manifest lifecycle and reports:** `RunArtifactManager` now supports
  `reports/*.json` registration and finalization. The training pipeline writes
  `reports/token_type_vocab.json`, finalizes `run_manifest.json` with
  `status: completed`, and records `completed_at`.
- **Loss-plan receipt depth:** `receipts/losses/loss_plan.json` now records the
  protected loss normalizer, fp32 selected-logit objective policy, token
  vocabulary groups, finite-gate policy, top-level metric definitions, weighted
  loss metrics, and count metrics instead of a thin stub.
- **Best-checkpoint event ordering:** same-step `eval.forward` now runs before
  `checkpoint`, and checkpoint writing can consume the same-step
  `eval.forward/acc_top1` metric for `best_acc_top1` pointer selection.
- **Compact-logits contract wording:** active OpenSpec specs, decision docs,
  blueprint, roadmap, and upstream standards no longer say V1 requires a full
  sequence logits time axis. The active contract is full-vocabulary logits over
  either the full sequence or explicitly selected supervised causal rows, plus
  a physical-position map validated by `LossContext`.
- **Qwen forward override guard:** external `extra_model_kwargs.logits_to_keep`
  remains rejected, but the error now points callers to the owned
  `QwenForwardInputs.logits_to_keep_positions` path rather than falsely
  implying compact selected-row logits are unsupported.
- **Token-type gate diagnostics:** the gate loss term now records configured
  token types and selected atom counts by token type, so `desc_text`, `schema`,
  `coordinate`, and `eos` coverage can be audited from loss artifacts.

Accepted findings intentionally not closed by this patch:

- **FA2 branch proof in vertical smoke remains a launch-critical gate.** Source
  probes show Qwen-routed padding-free/varlen evidence, but the historical
  five-step smoke receipts still have `fa2_varlen.proof: null`. The next smoke
  rerun must capture or attach real branch evidence for the exact training
  forward path; this should not be papered over with synthesized receipts.
- **Checkpoint reload remains a first-class implementation gate.** Current
  checkpoint writing covers adapter and selected special-token embedding delta
  artifacts, but an integrated base-plus-checkpoint reload path must still be
  proven before V1 acceptance.
- **Gradient coverage diagnostics remain incomplete.** Non-finite gates exist,
  but the optimizer surface should still prove expected trainable groups
  received gradients before claiming robust training diagnostics.
- **Multi-rank denominator and DeepSpeed system behavior remain smoke-gated.**
  Single-rank tests do not prove rank-safe denominator reduction or DeepSpeed
  production support.

This follow-up patches correctness and contract drift discovered by the two
verdict files. It does not replace the required five-step vertical smoke or
the later 8-GPU/DeepSpeed systems checks.

This is the first real source cutover and remains gated on explicit user
approval. Current preflight state:

- `src/` exists and `reference/legacy_src/` does not exist;
- `src/` has 395 tracked files and no tracked dirty changes;
- `src/` also contains 184 ignored `.pyc` files under 35 `__pycache__`
  directories;
- `.gitignore` ignores `/outputs/`, `**/__pycache__/**`, and `**/*.pyc`;
- `reference/legacy_openspec_2026-06-29/` remains reference-only and is not the
  target of this move.

Approved cutover plan after the user explicitly approves 4.1:

1. Remove ignored Python cache artifacts from `src/` so generated `.pyc` files
   are not archived as legacy source.
2. Move the old active `src/` tree to `reference/legacy_src/` without splitting
   or rewriting its tracked files.
3. Add `reference/README.md` stating that `reference/legacy_src/` and
   `reference/legacy_openspec_2026-06-29/` are read-only historical reference,
   not active implementation.
4. Create the new minimal `src/` package skeleton for the approved V1 module
   topology, with no old-source import shims and no placeholder framework code.
5. Mark 4.1 and 4.2 only after the archive move and skeleton verification
   actually pass.

Minimum verification after cutover:

- `git status --short -- src reference`;
- no `*.pyc` or `__pycache__` under `reference/legacy_src/`;
- no imports from `reference.legacy_src` or path hacks keeping the legacy tree
  active;
- `python -m py_compile` over the new skeleton;
- `openspec validate rebuild-coordexp-swift-training-infra --strict`;
- `git diff --check`.
