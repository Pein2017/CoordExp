---
doc_id: docs.architecture.coordexp-swift-blueprint
layer: docs
doc_type: architecture-blueprint
status: draft
domain: architecture
summary: Living implementation blueprint and approval cards for the CoordExp-swift rebuild.
updated: 2026-06-29
---

# CoordExp-Swift Blueprint

This is the living approval surface for the CoordExp-swift rebuild. It is
paired with `DECISIONS.md`: the decision log records resolved principles, while
this blueprint records concrete modules, classes, functionality, vertical
slices, invariants, and approval status.

No important module, class, or functionality should be implemented before its
card is approved.

Approval is intentionally manual and user-guided. Explicit approval is required
for public modules, important classes, cross-module data types, config/receipt
schemas, entrypoints, and public flow-defining functions. Small private helpers
inside an approved module may be implemented without a separate card only when
they do not change public contracts, research meaning, artifacts, or ownership.

Current implementation authorization: blocked pending review convergence and
explicit user approval. In this proposal, `approved` means the card text is
approved as a design target for review and later implementation planning. It is
not standalone authorization to write code. Reviewer timeout, reviewer
disconnection, or an untriaged finding is unresolved, not approval.

Design tradeoffs follow the user's priority order: accuracy and precision
first, training and overall system efficiency second, simplicity to avoid
over-design and redundancy third, and scalability or extension capability
fourth. Cards should preserve this order when local implementation choices
compete.

The current blueprint phase is design grilling plus later multi-agent
review/debate/refinement. After the user's manual module approval and review
rounds converge, implementation should switch to the new OpenSpec workflow and
the appropriate superpower implementation workflow. Do not treat this design
grilling session as authorization to begin coding.

If implementation evidence contradicts an approved card, stop and update this
blueprint or `DECISIONS.md` before continuing. Blocking open questions must be
resolved before coding the affected module; non-blocking open questions may
remain visible in the card. Do not add placeholder classes, TODO modules, fake
implementations, or broad base abstractions before approval.

V1 follows the austerity rule from `DECISIONS.md`: a documented invariant does
not automatically earn a public class, config knob, registry, artifact family,
or extension point. Implement approved modules with the smallest clear machinery:
plain functions, small dataclasses, named JSON receipts, and direct ownership.
Avoid framework-shaped abstractions, plugin systems, broad factories, report
frameworks, and future placeholder modules unless the approved card explicitly
requires them.

V1 is now in scope-freeze review. New V1 cards or public surfaces should be
added only to unblock implementation, fix contradictions, or address accepted
cross-review findings. Future hidden-state losses, rollout supervision,
visual-row capture, persistent caches, richer inference/eval entries, and other
next-version features remain reserved notes unless a payload-specific card is
approved.

The next gate is read-only cross-agent review of this blueprint and
`DECISIONS.md`. Reviewers should criticize the design for blockers,
contradictions, overdesign, under-specified contracts, and implementation risk;
they should not edit the docs directly. After review convergence, implementation
continues module-card by module-card under explicit user approval, with the
vertical smoke as the first milestone.

## Status Values

```text
draft                  # unfinished card text
needs-user-review      # design fork or contract needs explicit user decision
approved               # proposal text accepted; not coding authorization alone
implemented            # code exists for the approved card
verified               # implementation passed the card's verification gate
deferred               # reserved note; no V1 code surface
rejected               # explicitly out of scope
```

## V1 Navigation / Implementation Order

This table is an orientation aid, not a replacement for the cards below.

| Order | Card | First-smoke role | Gate before coding |
| --- | --- | --- | --- |
| 1 | `reference/legacy_src/` move and new `src/` skeleton | mandatory | explicit user approval for the archive/move command |
| 2 | `src/common/errors.py` and package markers | mandatory | keep minimal, no broad exports |
| 3 | `src/config/` plus minimal `src/artifacts/` run-dir/manifest core | mandatory | resolved-config and manifest schema choices fixed |
| 3b | `src/train.py --dry-run` | mandatory | delegates run-dir/manifest writing to artifact helpers |
| 4 | `src/data/` and `src/templates/` | mandatory | assistant suffix/span nesting contracts fixed |
| 5 | `src/qwen/loading.py` and `src/qwen/encoding.py` | mandatory | processor/template parity fixture passes |
| 6 | `src/packing/` and `src/training/stream.py` | mandatory | rank-aware pack-presentation contract fixed |
| 7 | `src/qwen/forward.py` | mandatory | FA2 branch proof defined, not only shape checks |
| 8 | `src/losses/` and `src/supervision/` | mandatory | planned-step denominator semantics fixed |
| 9 | `src/qwen/adapters.py` | adapter acceptance gate | dLoRA source study before `adapter.type: dlora` is accepted |
| 10 | `src/qwen/special_token_embeddings.py` | mandatory for training configs | tied/untied hook contract approved before coding |
| 11 | `src/optim/` and `src/runtime/` | mandatory | optimizer selector and non-finite/all-rank policies fixed |
| 12 | `src/artifacts/`, `src/metrics/`, `src/eval/forward.py` | mandatory | manifest/metric selector schemas fixed |

Early implementation may use a staged smoke ladder, but the final vertical
smoke remains the first milestone that proves train/eval/metric/checkpoint
behavior end to end.

Suggested first-smoke ladder:

```text
Smoke A: archive/skeleton + strict config + fixture render snapshot
Smoke B: Qwen processor parity + encode/pack + forward contract
Smoke C: CE/gate loss + backward + optimizer boundary for planned steps
Smoke D: scheduled eval.forward + metrics + checkpoints/checkpoint-final.json
```

Passing Smoke A/B/C is progress, not V1 acceptance. The first milestone remains
Smoke D with the complete five planned-step train/eval/checkpoint artifact set.

## Cross-Review Round 1 Resolutions

Earlier read-only review lanes audited `DECISIONS.md` and this blueprint before
implementation authorization in the surrounding review context. Those in-thread
review summaries are retained here only as design context; they are not
independently auditable local artifacts and are not coding authorization. The
locally auditable verdict files are listed in the next section. Accepted P1/P2
resolutions from the earlier review context:

- Card `approved` means proposal-text approval, not coding authorization.
- User tradeoff priority is accuracy/precision, then training/system
  efficiency, then simplicity/non-redundancy, then extensibility.
- Assistant suffix handling must prove exactly one `<|im_end|>\n`; assistant
  message content excludes the suffix when Qwen's processor inserts it.
- Rendered spans allow proper nesting and forbid crossing; only leaf spans
  assign token type.
- `segment_balanced` protected losses normalize over the full planned optimizer
  step/effective batch, not pack-local means averaged by accumulation.
- Metric events store split and name separately; `eval.forward/acc_top1:max` is
  selector syntax, not a stored metric name.
- Distributed stream ownership uses global pack-presentation ids and
  deterministic rank sharding.
- Scheduler time advances on the planned-step clock even when a non-finite guard
  skips the optimizer update; all-rank non-finite consensus is required.
- DeepSpeed execution remains a V1 runtime goal but needs its own systems smoke
  before support is claimed.
- dLoRA remains the intended/default recipe but is gated on a source study and
  minimal round-trip probe before `adapter.type: dlora` configs validate.
- Selected-token embedding training uses compact selected-token deltas that
  affect both input lookup and selected output-logit columns; the exact custom
  wrapper versus PEFT trainable-token mechanism is now source-study gated.
- Packed FA2 must be proven by branch-level explicit-varlen assertions, not only
  shape checks.
- `run_manifest.json`, dry-run behavior, runtime identity, resolved-config
  provenance, run-id collision handling, and checkpoint alias locations are now
  minimum contracts.

Remaining gate: implementation is still blocked until the user explicitly
approves the next module/card. If the user wants a pre-dLoRA first smoke using
standard LoRA or no adapter, that is a new decision and is not currently
adopted.

## External Agent Verdict Resolutions

Two later read-only verdicts are preserved locally:

- `agent_verdict/coordexp-swift-codex.md`
- `agent_verdict/coordexp-swift-claude.md`

Accepted resolutions from these verdicts are folded into this blueprint and
`DECISIONS.md`. They are evidence for design hardening, not coding
authorization. Main accepted patches:

- planned-step `LossNormalizers` and backend loss-scaling rules are explicit;
- scalar non-finite handling is split into a pre-backward finite gate and a
  post-backward gradient/overflow gate;
- Qwen no-resize pack cost comes from actual no-resize `image_grid_thw` or a
  proven-equivalent local computation, not helper paths that may apply
  smart-resize semantics;
- Qwen forward validates installed output object shape rather than requiring a
  rote `return_dict=True` kwarg;
- `inputs_embeds` shortcuts are forbidden in V1 because they can bypass Qwen3-VL
  visual replacement or DeepStack behavior;
- dLoRA must be defined against DoRA/`use_dora` or a CoordExp-owned mechanism
  before `adapter.type: dlora` validates;
- special-token embedding implementation must source-study PEFT
  `TrainableTokens` / LoRA `trainable_token_indices` versus custom wrappers and
  avoid full embedding/head saves;
- first-smoke fixture source must be pinned or materialized before it becomes
  an OpenSpec implementation baseline;
- cadence config paths and `resolved_step_schedule.json` event fields are
  canonicalized.

## Card Template

```md
### {Name}

- Status:
- Location:
- Purpose:
- Public interface:
- Inputs:
- Outputs:
- Owned state:
- Invariants:
- Failure modes:
- Non-goals:
- Debug/receipt artifacts:
- Tests or parity checks:
- Open questions:
- Recommendation:
```

## Implementation Slice Cards

### Vertical Smoke Slice

- Status: approved
- Location:
  - `reference/legacy_src/`
  - `src/`
  - `tests/fixtures/smoke/qwen3_vl_single_image_pack/`
  - `docs/architecture/proposals/2026-06-27-coordexp-swift/`
- Purpose: prove the real training path end to end before broad module
  expansion.
- Public interface: `python -m src.train --config ...`
- Inputs:
  - self-contained permanent tiny single-image JSONL smoke fixture
  - `config.yaml` fixture-local training config
  - `examples.jsonl`
  - one copied real local CoordExp-style image
  - pinned source example/image from current training data, chosen to be short,
    valid, single-image, and exactly two-object, or a pinned deterministic
    reduction from a larger valid source row
  - stable fixture metadata and `expected_rendered.json`
  - strict fixture-local YAML config with `global_max_length`, `sample_limit`,
    `effective_batch_size: 1`, `epochs`, `max_steps: 5`, and default
    `eval.forward` cadence
  - fixture-local `config.yaml` is self-contained and does not use `extends`
  - fixture config writes smoke run artifacts under an ignored artifact root
    such as `outputs/smoke/qwen3_vl_single_image_pack/`
  - base Qwen3-VL model from
    `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
  - optional adapter path or adapter-init config
- Outputs:
  - `run_manifest.json`
  - `configs/resolved.yaml`
  - `configs/resolved.json`
  - `debug/qwen_forward_contract.json`
  - `reports/token_type_vocab.json`
  - `reports/pack_plan.json`
  - `reports/loss_plan.json`
  - `resolved_step_schedule.json`
  - metric events and summaries
  - checkpoint metadata, status records, and final checkpoint
- Owned state:
  - run directory
  - deterministic smoke fixture/config under
    `tests/fixtures/smoke/qwen3_vl_single_image_pack/`
  - planned-step schedule for `resolved_max_steps: 5`
- Invariants:
  - move old `src/` to `reference/legacy_src/` before creating the new active
    `src/`
  - create only approved new `src/` package markers/skeleton at first; no TODO
    classes, broad abstract bases, or placeholder implementations
  - fixture is self-contained; smoke does not depend on external dataset paths
  - smoke config lives with the fixture, not under the main `configs/` tree
  - smoke config uses the normal strict `TrainConfig` schema but no `extends`
  - smoke outputs are generated under `run.artifact_root`, not under the fixture
    source directory
  - fixture uses deterministic `object_ordering: source_order`
  - fixture contains exactly two objects with distinct descriptions and boxes
  - source example/image provenance, original source path, source file stat,
    copied fixture path, checksum, and selection rationale are recorded after
    copying into the fixture
  - `expected_rendered.json` is pure pre-tokenization text metadata, not Qwen
    tensor or packed-position metadata
  - `expected_rendered.json` is checked exactly for messages, prompt text,
    supervised response text, spans, object order, and image metadata, with only
    explicitly excluded generated provenance fields ignored
  - standard forward path is one packed sequence per rank/step, not padded
    batches
  - V1 single-image only; video payloads fail fast
  - Qwen forward receives `labels=None`, `use_cache=False`, full logits, and no
    model-side CE path
  - packed FA2 isolation uses `cu_seq_lens_q/k` and `max_length_q/k`, not a 2D
    padding mask
  - shifted-loss-consumption parity proves `TokenAtom.target_position`,
    `LossContext.logits_position`, dense labels, and consumed CE rows agree
  - loss path computes `BaseTokenCE` and `TokenTypeGateLoss` from fp32 selected
    logits
  - the V1 vertical smoke does not run DeepSpeed; it verifies DeepSpeed
    config-conflict handling at setup level while leaving real DeepSpeed
    execution to a later systems smoke
  - adapter-enabled dLoRA smoke requires the dLoRA definition/source study and
    minimal round-trip probe before its config validates
  - base-only or standard-LoRA pre-dLoRA smoke is allowed only after an explicit
    user decision; it is not implied by the default dLoRA recipe
- Failure modes:
  - contract errors use typed exceptions from `src/common/errors.py`
  - failed smoke receipts include bounded rendered-text snippets around failed
    spans/tokens, plus example id and source path when available
  - smoke output written into the fixture source directory is a failure
  - placeholder/grid mismatch fails before Qwen forward
  - ordinary 2D packed FA2 mask with zeros fails fast
  - hidden-state-dependent loss configs fail until the next-version hook exists
- Non-goals:
  - rollout training implementation
  - hidden-state losses in V1
  - video or multi-image support
  - exact optimizer/RNG resume
  - vLLM or inference-loop migration
  - DeepSpeed execution smoke in the first vertical slice
- Debug/receipt artifacts:
  - subsystem-owned JSON receipts linked from `run_manifest.json`
- Tests or parity checks:
  - MS-Swift/Qwen shifted-loss-consumption parity fixture
  - render snapshot and tokenization/alignment snapshot
  - pack construction check
  - Qwen forward contract smoke
  - CE plus token-type gate loss check
  - backward and optimizer-step check
  - config inheritance and unknown-key failure checks
  - DeepSpeed accumulation/batch-size conflict setup check without running a
    DeepSpeed training job
  - vertical smoke: five planned steps plus scheduled `eval.forward` calls,
    metric emission, checkpoint status metadata, and
    `checkpoints/checkpoint-final.json`
  - acceptance requires resolved config, Qwen setup receipt, pack plan, loss
    plan, metrics, scheduled `eval.forward` summaries, checkpoint metadata, and
    `checkpoints/checkpoint-final.json`; unit tests alone are not sufficient
- Open questions:
  - exact package-level public API names per module
- Recommendation: implement this slice first, then expand module coverage only
  after the real path trains, evaluates, emits receipts, and checkpoints.

### Smoke Fixture Contract

- Status: approved
- Location: `tests/fixtures/smoke/qwen3_vl_single_image_pack/`
- Purpose: provide the permanent self-contained regression anchor for the first
  real Qwen3-VL vertical smoke.
- Public interface:
  - `python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml`
- Inputs:
  - one copied real local CoordExp-style image
  - one canonical-shape JSONL example with two objects
  - source selected before OpenSpec implementation baseline from a current
    `len12000` training row, then pinned by source path, row id, image id,
    object ids, reduction rule when applicable, checksum, normalized, and copied
    into the fixture
  - deterministic source-order object ordering
  - self-contained fixture-local `config.yaml` with no `extends`
- Outputs:
  - rendered text snapshot
  - later tokenization snapshot after encoder implementation
- Owned state:
  - `README.md`
  - `config.yaml`
  - `examples.jsonl`
  - `expected_rendered.json`
  - `images/`
  - `checksums.json`
  - later `expected_tokenization.json`
- Invariants:
  - `config.yaml` is the fixture-local training config name
  - `config.yaml` uses the same strict `TrainConfig` schema as normal training
    configs but is self-contained and does not use `extends`
  - `config.yaml` sets `run.artifact_root` to an ignored generated-output
    location such as `outputs/smoke/qwen3_vl_single_image_pack/`
  - smoke outputs are not written under the fixture source directory
  - fixture is self-contained and does not depend on external dataset paths
  - first fixture uses exactly two objects with distinct descriptions and boxes
  - selected source example preferably comes from a real current `len12000`
    training row with exactly two valid objects, short rendered length, ordinary
    no-resize-compatible image dimensions, and descriptions without
    special/control token hazards
  - if no suitable exactly-two-object row exists, a larger valid source row may
    be reduced to exactly two copied objects, with the reduction recorded in
    `checksums.json` selection rationale
  - when reducing a larger source row, choose the first two source-order objects
    that are valid, short, and description-safe
  - before the fixture becomes an OpenSpec implementation baseline, the
    selected row/image/reduction must be materialized in the fixture or pinned
    tightly enough that another agent regenerates the same `examples.jsonl`,
    copied image set, `checksums.json`, and `expected_rendered.json`
  - source selection prioritizes boring validity over representational diversity
  - fixture descriptions are ordinary safe strings copied from source after
    confirming they do not contain Qwen/control/CoordExp token syntax
  - the first vertical smoke does not intentionally include newline, tab,
    control-token, or escaping edge cases in descriptions
  - `examples.jsonl` uses the new canonical `RawExample` shape, not current
    coord-jsonl source-format field names such as `images`, `bbox_2d`, or `desc`
  - current `len12000` coord-jsonl source-format normalization is tested
    separately from the vertical smoke
  - fixture objects store canonical integer coordinate-bin `bbox` values; original
    source `bbox_2d` coordinate-token strings may be recorded under
    `metadata.source` for provenance
  - fixture JSONL image path is fixture-local and relative, for example
    `images/<filename>`; absolute image paths are not used
  - fixture `example_id` preserves the original stable source identifier with a
    fixture suffix such as `<source_example_id>__smoke2obj`; original id is
    recorded in metadata
  - selected object ids preserve original object ids when available; any
    normalization is recorded in metadata
  - source provenance is nested under `metadata.source`, including original
    dataset path, original example id, original object ids, reduction note when
    applicable, and source row number when available
  - fixture JSONL includes image `width` and `height`
  - Qwen encoding validates decoded image size against fixture `width` and
    `height`
  - selected image is copied under `images/`
  - copied image and example record original source path, source file stat,
    copied fixture path, checksum, and selection rationale
  - `checksums.json` is fixture-specific and small: checksum algorithm, one
    `source_example` object, one `images` list, original source paths, copied
    fixture paths, source file stats, checksums, and selection rationale
  - copied fixture image checksum uses SHA-256, recorded once as
    `"algorithm": "sha256"` in `checksums.json`
  - `object_ordering: source_order` is locked for the defining smoke
  - `expected_rendered.json` contains messages, prompt text, supervised
    response text, typed char spans, realized object order, and image metadata
  - `expected_rendered.json` is exact: messages, prompt text, supervised
    response text, spans, object order, and image metadata match byte-for-byte
    except explicitly excluded generated provenance fields
  - `expected_rendered.json` does not contain token ids, Qwen tensors, or packed
    positions
  - `expected_rendered.json` is produced once by the real renderer path, reviewed,
    and committed as a frozen expected artifact rather than hand-written from
    scratch
  - snapshot updates are manual and review-gated; tests may show diffs but must
    not auto-update `expected_rendered.json` or `expected_tokenization.json`
    unless a future explicit developer command is approved
  - snapshot mismatch diff artifacts are written under the ignored smoke/debug
    output root, not beside fixture source files
  - `expected_tokenization.json` is reserved until the Qwen encoder exists and
    should not be hand-authored before then
  - after `QwenExampleEncoder` exists, `expected_tokenization.json` is generated
    through the real encoder path and frozen only after review
  - `README.md` is contract-oriented: purpose, files, invariants, how to run,
    and what failures mean
- Failure modes:
  - external image/data dependency
  - fixture `config.yaml` depends on production `extends`
  - smoke run outputs pollute the fixture source directory
  - generated smoke outputs are treated as permanent golden source artifacts
  - synthetic-only fixture pretending to be the defining Qwen smoke
  - missing checksum or original source provenance for the copied fixture image
  - `checksums.json` grows into a generic provenance manifest
  - larger source row reduction is not recorded in `checksums.json`
  - subjective visually distinct or hard object choice replaces first-valid
    source-order selection
  - first fixture intentionally includes description whitespace/control-token
    edge cases
  - fixture ids discard source traceability without recorded normalization
  - fixture image dimensions are omitted or not validated during Qwen encoding
  - fixture JSONL uses absolute image paths
  - source provenance is added as arbitrary top-level fixture fields instead of
    `metadata.source`
  - copied image checksum uses an unrecorded or configurable algorithm in V1
  - loose semantic comparison for `expected_rendered.json`
  - snapshot tests auto-update expected files without explicit review
  - snapshot diff artifacts are written beside fixture source files
  - fixture `examples.jsonl` uses current coord-jsonl source-format fields
    instead of canonical `RawExample` shape
  - random object ordering in the defining smoke
  - token ids hand-authored before the tokenizer/encoder path exists
- Non-goals:
  - broad dataset coverage
  - random-ordering coverage
  - multiple-image or video coverage
  - tutorial-style documentation
- Debug/receipt artifacts:
  - `expected_rendered.json`
  - future `expected_tokenization.json`
  - `checksums.json`
  - optional README summary pointing to `checksums.json`
- Tests or parity checks:
  - exact render snapshot check
  - snapshot update path is manual/review-gated and does not auto-update
    expected files
  - snapshot mismatch writes bounded diff artifacts under ignored smoke/debug
    output root
  - later tokenization/alignment snapshot check
  - fixture JSONL is canonical-shape while current coord-jsonl source-format
    normalization is covered by a separate loader test
  - fixture ids preserve source traceability and record normalization when
    needed
  - fixture image dimensions are present and validated by Qwen encoding
  - smoke config has no `extends`
  - smoke generated artifacts land under ignored `run.artifact_root`, not under
    the fixture source directory
  - Qwen forward contract smoke
- Open questions:
  - none after the fixture source is pinned; before that pinning, the fixture is
    an approved selection contract rather than a reproducible baseline artifact
- Recommendation: implement when the vertical smoke slice begins, before broad
  trainer work.

## Module Cards

### Archive And Skeleton Setup

- Status: approved
- Location:
  - `reference/legacy_src/`
  - `reference/README.md`
  - `src/`
- Purpose: make the rebuild start from a clean active import root while keeping
  the old implementation readable as reference.
- Public interface: none.
- Inputs:
  - existing old `src/`
  - approved source topology from `DECISIONS.md`
- Outputs:
  - old source moved intact to `reference/legacy_src/`
  - `reference/README.md`
  - minimal new `src/` package skeleton
- Owned state:
  - `reference/`
  - top-level `src/` directory layout
- Invariants:
  - old `src/` is moved intact, not split into topic folders
  - `reference/README.md` explains that `reference/legacy_src/` is read-only
    reference and not active implementation
  - new `src/` does not import from `reference/legacy_src/`
  - no old-source import shims, old-entrypoint aliases, or path hacks
  - new package `__init__.py` files are minimal markers only
  - no TODO classes/functions or broad abstract bases in the skeleton
  - no top-level `src.__version__` in V1; use git/package/run artifact identity
- Failure modes:
  - mixed old/new import paths
  - compatibility shims that keep the old pipeline alive invisibly
  - placeholder code that pretends unimplemented architecture exists
- Non-goals:
  - deleting legacy code
  - refactoring legacy code
  - preserving legacy runtime compatibility
- Debug/receipt artifacts: none.
- Tests or parity checks:
  - import check for the new empty package skeleton when implementation begins
  - residue check that new `src/` does not import `reference.legacy_src`
- Open questions: none.
- Recommendation: perform this as the first implementation step before any new
  module implementation.

### `src/common/errors.py`

- Status: approved
- Location: `src/common/errors.py`
- Purpose: provide a small typed contract-error vocabulary for fail-fast
  debugging.
- Public interface:
  - `CoordExpError`
  - `ConfigContractError`
  - `DataContractError`
  - `TemplateContractError`
  - `EncodingContractError`
  - `PackingContractError`
  - `QwenForwardContractError`
  - `LossContractError`
  - `RuntimeContractError`
- Inputs:
  - short stable error `code`
  - human-readable message
  - optional small serializable `context: dict[str, object]`
  - optional `cause`
- Outputs: typed exceptions.
- Owned state: `code`, `message`, `context`, and optional `cause` fields on
  exception instances.
- Invariants:
  - exceptions identify contract class, not generic control-flow branches
  - messages include enough local context to debug without reopening many files
  - optional context is formatted into readable exception text
  - context remains small and JSON-like enough for future receipt inclusion
  - context may include bounded snippets around failed rendered spans or tokens,
    but does not dump full prompts by default
- Failure modes: over-broad hierarchy or silent fallback would violate the
  purpose.
- Non-goals:
  - recoverable runtime policy
  - retry orchestration
  - warning system
  - rich dataclass hierarchy per error kind
  - Pydantic model per error context
- Debug/receipt artifacts: none directly.
- Tests or parity checks: import check and representative message tests when
  first used.
- Open questions: none.
- Recommendation: implement as the first new source module after archiving old
  `src/`.

### Public Package Surfaces

- Status: approved
- Location: all packages under `src/`
- Purpose: keep the execution flow traceable through narrow, approved public
  APIs.
- Public interface: package-owned exports only after approval.
- Inputs: package-internal helpers and typed contract objects.
- Outputs: stable builder/functions/classes needed by the next layer.
- Owned state: package-local implementation details.
- Invariants:
  - package `__init__.py` files are minimal marker files in V1
  - no broad package-level re-exports until a public API is proven and approved
  - no package `__all__` lists until approved for that package
  - do not export every helper for convenience
  - callers should not casually import file internals
  - public names should align with the decision vocabulary
- Failure modes: helper sprawl, circular ownership, hidden semantic owners.
- Non-goals: a large framework registry or plugin system in V1.
- Debug/receipt artifacts: package-specific where relevant.
- Tests or parity checks: import-boundary checks can be added when package APIs
  stabilize.
- Open questions: none for V1 `__init__.py`/`__all__` policy.
- Recommendation: keep each package public surface small and promote helpers
  only when reuse proves they belong there.

### `src/config/`

- Status: approved
- Location: `src/config/`
- Purpose: own typed train config models, YAML loading, single-parent
  inheritance, strict validation, path resolution, and resolved-config writing.
- Public interface:
  - `load_train_config(path) -> ResolvedTrainConfig`
  - `write_resolved_config_artifacts(resolved_config, run_dir) ->
    ResolvedConfigArtifacts`
- Inputs:
  - root runnable YAML config with `schema_version: 1`
  - optional single-parent `extends` chain
  - inherited fragments such as shared `base.yaml` and research-direction base
    YAMLs
  - optional `REQUIRED` placeholders in inherited fragments for non-trivial
    hyperparameters that must be set by concrete runs
- Outputs:
  - `ResolvedTrainConfig` containing frozen `TrainConfig`, resolved config
    fingerprint, effective `schema_version`, loader/schema version, and resolved
    path metadata
  - `configs/resolved.yaml`
  - `configs/resolved.json`
  - compact fingerprint and path-resolution metadata
  - compact `resolution.sources` metadata for entry config, parent chain, source
    fingerprints, and path-field origins
- Owned state:
  - Pydantic v2 strict config models
  - `ResolvedTrainConfig`
  - `ResolvedConfigArtifacts`
  - YAML loader around PyYAML or existing parser
  - path-resolution and compact resolution metadata helpers
  - `models.py`
  - `loader.py`
  - `resolve.py`
  - `paths.py`
  - `fingerprint.py`
  - `writer.py`
- Invariants:
  - unknown keys fail with path-aware `ConfigContractError`
  - unknown keys fail inside backend-specific subtrees such as
    `runtime.accelerate` and `runtime.deepspeed`; add explicit `extra_args` only
    if a future backend integration proves it is necessary
  - runnable root configs require `schema_version: 1`
  - inherited fragments may omit `schema_version`
  - normal reusable configs live under `configs/base.yaml`,
    `configs/directions/<direction>/base.yaml`, and
    `configs/directions/<direction>/<run>.yaml`
  - smoke fixture config is the V1 exception and lives beside the fixture
  - `extends` is a single parent path per file; parent files may chain
  - `extends` is top-level only
  - parents resolve first, child dictionaries deep-merge over parents
  - lists replace rather than append
  - cycles fail with the full config path chain
  - inherited fragments may be partial and non-runnable
  - data, fixture, cache, and reference paths resolve relative to the YAML file
    that declares them
  - `run.artifact_root` remains literal or cwd-relative by operator choice
  - explicit YAML `null` is valid only for optional fields; it does not delete
    inherited keys
  - `adapter:` is object-shaped when adapter tuning or adapter checkpoint loading
    is expected
  - do not accept `adapter: null`; omit `adapter:` only for explicitly
    base-only, non-training utilities that do not initialize adapter tuning
  - V1 rejects `model.adapter.*`
  - training configs require `model.special_token_embeddings`; read-only
    non-training utilities such as tokenizer setup inspection or base-only
    forward-contract probes may omit it
  - final resolved `TrainConfig` is frozen/immutable
  - initial Pydantic models live together in `models.py` with nested section
    models until real size or ownership pressure justifies a split
  - final resolved config may not contain the string sentinel `REQUIRED`
  - LR groups, effective batch choices, packing/global
    length, dLoRA config, run-length policy, and other non-trivial knobs may use
    `REQUIRED` in base fragments as reminders, but concrete runnable configs
    must resolve them
  - `packing.global_max_length` is the canonical home for the hard physical
    expanded sequence-length budget used by encoding and packing
  - runnable configs expose `training.effective_batch_size`, not
    `training.grad_accum_steps`
  - `training.effective_batch_size` means global packed sequences per optimizer
    update
  - runtime setup computes `resolved_grad_accum_steps` from effective batch size
    and actual world size
  - resolved config records authored `training.effective_batch_size`, not
    runtime-derived `resolved_grad_accum_steps`
  - non-divisible effective batch size versus world size fails before training
  - epoch-led runs use deterministic tail-fill to complete the final
    effective-batch window
  - V1 never silently discards final packs and never performs a smaller partial
    final optimizer update
  - `epochs` means minimum full-pass target plus bounded tail completion, not
    exact presentation count
  - `training.epochs` is a positive integer in V1
  - production run length is authored with `training.epochs` and
    `training.max_steps: null`
  - smoke/debug configs may set `training.max_steps` to a positive integer
  - when `training.max_steps` is set, it overrides `training.epochs` and directly
    defines `resolved_max_steps`
  - when `training.max_steps` is null, `resolved_max_steps` is computed from
    `training.epochs` and resolved packed train dataloader cardinality
  - V1 accepts only `training.mode: supervised`; the field remains in configs
    and artifacts for readability and future rollout compatibility
  - `data.train_order` defaults to `shuffle`; smoke/debug may use `source_order`
  - config schema rejects dormant `template.language`; V1 templates are
    English-only
  - root `configs/base.yaml` defines the backbone/run skeleton and should not
    carry loss-weight placeholders by default
  - direction/run configs supply the neck/head: loss modules, loss weights, and
    research-specific choices
  - path resolution is two-stage: parse/merge raw YAML, validate enough to know
    path fields, resolve paths, then validate final frozen config
  - config validation completes before fixture/data loading, model path opening,
    or model/processor construction
  - config defaults do not depend on inspecting dataset contents
  - config loading does not import torch, Transformers, load models, inspect
    processors, or touch CUDA
  - smoke fixture config uses the same strict train schema as normal training
    configs
  - `configs/resolved.yaml` and `configs/resolved.json` are clean
    machine-authored resolved truth, not comment-preserving rewrites of the
    authored YAML chain
  - resolved config artifacts do not copy authored config files, store the
    inheritance chain, or preserve parent/child relative relationships by default
  - resolved config artifacts do store compact audit metadata for source paths,
    source fingerprints, declaring-file origin for path fields, and final path
    serialization form
  - input paths serialize as resolved absolute paths, with repo-relative
    companions when under the repo; `run.artifact_root` remains operator
    placement while the manifest records the concrete absolute `run_dir`
  - run id is run/artifact setup metadata, not a field inside
    `ResolvedTrainConfig`
- Failure modes:
  - multiple-parent `extends`
  - nested section-level extends or package-specific include keys
  - arbitrary include graph
  - warning-and-ignoring unknown keys
  - permissive unknown keys under backend subtrees
  - `null` used as key deletion syntax
  - surviving `REQUIRED` placeholder in final config
  - authored `training.grad_accum_steps` accepted by config validation
  - implicit rounding of effective batch size to world size
  - runtime-derived `resolved_grad_accum_steps` written back into
    `configs/resolved.yaml` or `configs/resolved.json`
  - incomplete final effective-batch window becomes a partial optimizer update
  - incomplete final packs are silently dropped
  - tail-fill is nondeterministic or unrecorded
  - fractional `training.epochs` accepted in V1
  - example train order is conflated with `template.object_ordering`
  - authored production config with a non-null `training.max_steps`
  - missing `training.epochs`
  - non-null `training.max_steps` outside smoke/debug contexts without an
    explicit operator choice
  - dormant `template.language` accepted by config validation
  - loss-weight placeholders hidden in root `configs/base.yaml`
  - heuristic rewriting of arbitrary strings as paths
  - leaving input paths unresolved until module runtime
  - loading fixture/data/model paths before final config validation
  - caller-instantiated Pydantic models from raw YAML dictionaries
  - config loading that performs model/GPU probing
  - resolved config artifacts that require the original authored inheritance
    chain for replay or inspection
  - path fields depend on process cwd after resolution
  - copied/linked authored source configs treated as default run truth
- Non-goals:
  - Hydra/OmegaConf composition
  - comment-preserving YAML round trip
  - special mini-schema for smoke
  - public merge/loader helper surface in V1
  - one large `config.py`
  - one file per config section in V1
  - one giant flat `TrainConfig`
- Debug/receipt artifacts:
  - `configs/resolved.yaml`
  - `configs/resolved.json`
  - config content fingerprints
  - path-resolution receipt
- Tests or parity checks:
  - single-parent inheritance chain load
  - top-level-only `extends` validation
  - unknown-key failure
  - backend-subtree unknown-key failure
  - list replacement behavior
  - optional-only `null` behavior
  - `REQUIRED` placeholder failure in final config
  - `training.grad_accum_steps` unknown-key failure
  - non-divisible `training.effective_batch_size` versus runtime world size fails
    in setup
  - runtime receipts record world size, effective batch size, and
    `resolved_grad_accum_steps`
  - epoch-led run with non-divisible pack presentations tail-fills
    deterministically and records `tail_fill_pack_count`
  - no partial optimizer step occurs for the final epoch-led window
  - no final packs are silently discarded
  - fractional `training.epochs` validation failure
  - default `data.train_order` resolves to `shuffle`
  - `data.train_order` does not affect object order inside a rendered example
  - production config with `max_steps: null` resolves steps from epochs and pack
    cardinality
  - smoke config with `max_steps: 5` resolves exactly five planned steps
  - `template.language` unknown-key failure
  - root base with non-trivial knob placeholders and direction/run overrides
  - cycle detection
  - three-level inheritance fixture:
    `configs/base.yaml -> configs/directions/<direction>/base.yaml ->
    configs/directions/<direction>/<run>.yaml`
  - YAML-relative input path resolution
  - `run.artifact_root` literal/cwd-relative behavior
  - frozen resolved config
  - `ResolvedTrainConfig` includes config, resolved config fingerprint,
    `schema_version`, loader/schema version, and resolved path metadata
  - three-level inheritance fixture records source fingerprints and path origins
    without copying authored YAML files into the run
  - resolved config does not preserve authored comments
  - resolved config does not require the authored inheritance chain for replay or
    inspection
  - two-stage path resolution
  - no torch/Transformers import during config loading
- Open questions: none for V1 config model file split or
  `ResolvedTrainConfig` top-level contents.
- Recommendation: implement after archive/skeleton setup and before data/template
  modules, because every later surface depends on typed resolved config slices.

### `src/data/`

- Status: approved
- Location: `src/data/`
- Purpose: load strict V1 JSONL records into typed `RawExample`s, resolve image
  references, validate coordinate-token geometry, and preserve lightweight
  provenance before rendering.
- Public interface:
  - `load_raw_examples(config_slice) -> Iterable[RawExample]`
  - exact function names may be adjusted at implementation approval, but the
    public surface should stay narrow
- Inputs:
  - JSONL dataset path from resolved config
  - declaring config/JSONL root for relative image references
  - V1 canonical records containing `example_id`, `image`, `objects`, and
    optional `metadata`
  - current main-branch `len12000` coord JSONL rows such as
    `public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`, without
    requiring regeneration
- Outputs:
  - typed `RawExample`s
  - source path/row metadata
  - lightweight raw-record provenance or hash
  - resolved image path
- Owned state:
  - `types.py` containing immutable `RawExample`, `RawObject`, and `ImageRef`
    runtime records
  - `geometry.py` containing strict coordinate-bin parsing and validation
    helpers
  - JSONL row/provenance helpers
- Invariants:
  - JSONL only in V1
  - canonical data records are frozen dataclasses or dataclass-like immutable
    values; Pydantic is primarily for config validation, not per-example runtime
    records
  - canonical `RawExample` shape is `example_id`, `image`, `objects`, and
    optional `metadata`
  - canonical `ImageRef` shape is `path`, optional `width`, and optional
    `height`
  - canonical `RawObject` shape is `object_id`, `description`, `bbox`, and
    optional `metadata`
  - `RawExample.objects` is stored internally as `tuple[RawObject, ...]`
  - raw JSONL rows instantiate canonical records only through narrow
    loader/factory functions
  - loader/factory functions own strict validation, id normalization, bbox
    parsing, and bounded provenance capture
  - unknown top-level fields fail; extra provenance belongs under `metadata`
  - current coord-jsonl source-format normalization is narrow: support only the
    known current `len12000` coord-jsonl source fields approved in
    `DECISIONS.md`
  - canonical image field is `image: {path, width, height}` with width/height
    optional
  - width and height are optional in the general `ImageRef` type but required
    by the current Qwen3-VL supervised training loader path and smoke fixture
  - `data.train_order` is the example-order policy across the training dataset;
    it is separate from `template.object_ordering`
  - default `data.train_order` is `shuffle`
  - `data.train_order: source_order` is allowed for smoke/debug reproducibility
  - shuffle order is deterministic from the global seed-derived data-order seed
  - current coord-jsonl source rows with `images: ["..."]`, top-level `width`,
    `height`, `image_id`, and `file_name` normalize into canonical `image` and
    provenance
  - every example has exactly one image reference
  - canonical `example_id` is a string; numeric source identifiers are
    stringified during normalization
  - `example_id` is unique within each loaded JSONL file or dataset source
  - explicit `example_id` is preferred; stable current-source identifiers such as
    `image_id` or `file_name` may normalize into `example_id`
  - every object has stable `object_id`, description text, and bbox
  - canonical `object_id` is a string; numeric source identifiers such as
    `coco_ann_id` are stringified during normalization
  - object fields normalize to canonical `object_id`, `description`, and `bbox`
  - current-source object `desc` maps to `description`
  - canonical field name is `description`, not source-dialect `desc`
  - current-source object `bbox_2d` coordinate-token strings map to integer
    coordinate-bin `bbox`
  - coordinate-token parsing for current-source `bbox_2d` is strict: accept only
    exact canonical `<|coord_N|>` strings with decimal `N` in `[0, 999]`
  - reject whitespace, raw integer strings, floats, signs, malformed wrappers,
    non-canonical zero padding, split token fragments, and out-of-range
    coordinate-token values
  - explicit `object_id` is preferred; stable current-source ids such as `coco_ann_id`
    may normalize into `object_id`
  - bbox flow is source coordinate-token strings -> canonical integer bins ->
    rendered coordinate-token strings; raw source token strings do not bypass
    canonical geometry validation
  - object ids are not auto-generated by index
  - object ids are unique within one `RawExample`; global object identity is
    `(example_id, object_id)`
  - descriptions are non-empty strings after stripping
  - canonical field name is `bbox`, not `bbox_2d`
  - `RawObject.bbox` is stored internally as `tuple[int, int, int, int]`
  - JSON/YAML fixture serialization may use arrays, but Python canonical
    geometry is immutable
  - bbox format is `[x1, y1, x2, y2]` in integer coordinate-bin space `[0, 999]`
  - teacher-forced supervised training bbox values satisfy `x1 < x2` and
    `y1 < y2`; zero-area boxes are schema-invalid for this training path
  - bbox validation happens in `data/` before template rendering; templates may
    assert validity but are not the first geometry gate
  - invalid geometry is not clamped, repaired, warned through, or converted from
    pixels in V1
  - original current-source `bbox_2d` coordinate-token strings may be preserved
    under metadata or bounded debug provenance when useful, but never as the
    canonical geometry field
  - source `category_id` and `category_name` are preserved under
    `RawObject.metadata.source` when present, but do not affect template
    rendering or loss by default
  - future rollout/offline-inference decoded geometry must also validate and
    invalid decoded outputs must be reported, not trusted
  - image references are resolved in data loading and recorded, but images are
    not opened in `data/`
  - `data/` checks image path existence but does not decode image bytes
  - image identity fingerprinting uses path plus file stat by default; stricter
    paths such as smoke fixtures may add checksums
  - `metadata` is JSON-serializable nested primitives/lists/dicts only, bounded,
    and non-authoritative
  - raw provenance includes source path, row number, source format name, stable
    source ids when available, optional byte offset, and a raw-line or
    canonical-record hash
  - full raw dicts are not stored inside every `RawExample` by default
  - templates receive canonical `RawExample` objects only, never raw JSON rows
  - `sample_limit` counts accepted valid examples after sequential validation;
    invalid rows encountered before the limit still fail
  - trainer/packer do not own sample limiting
  - `qwen/` opens images later for processor conversion
  - `templates/` never resolves or opens images
  - invalid records fail fast in V1; no skip/count policy
- Failure modes:
  - missing or duplicate example id
  - zero-image or multi-image record
  - missing object id
  - mutable object list exposed as canonical runtime state
  - invalid bbox length, dtype, range, or ordering
  - loose current-source coordinate-token parsing such as whitespace, raw
    integer strings, floats, malformed wrappers, or zero-padded token numbers
  - zero-area bbox in teacher-forced supervised training
  - empty description after stripping
  - canonical object keeps source-dialect `desc` instead of `description`
  - non-JSON-serializable metadata
  - oversized or semantic-bearing metadata
  - arbitrary top-level fields
  - unsupported source-format alias outside the approved current `len12000`
    coord-jsonl source surface
  - templates depending on source JSONL dialects instead of canonical records
  - image path missing or unresolved
  - parser receives non-JSONL input
- Non-goals:
  - dataset registry
  - JSON array parsing
  - live image loading
  - template rendering
  - bbox pixel-to-token conversion
  - invalid-row skip policy
- Debug/receipt artifacts:
  - data fingerprint
  - optional raw-index cache metadata
  - source row/hash provenance in downstream debug receipts
- Tests or parity checks:
  - valid two-object JSONL fixture
  - current `len12000` coord-jsonl source-format normalization fixture
  - current-source `bbox_2d` strict parser accepts exact coordinate-token strings
    and rejects malformed or loose alternatives
  - unsupported source-format alias failure
  - unknown top-level field failure
  - factory-only construction path covers canonical JSONL and current
    coord-jsonl source rows
  - numeric source ids normalize to string canonical ids
  - canonical record shape check for `RawExample`, `ImageRef`, and `RawObject`
  - `RawExample.objects` is a tuple in Python while JSON/YAML fixtures serialize
    it as an array
  - duplicate `example_id` failure
  - duplicate object id within one example failure
  - missing `object_id` failure
  - invalid bbox failures for length, dtype, range, and ordering
  - canonical `RawObject.bbox` is a tuple in Python while JSON/YAML fixtures
    serialize it as an array
  - templates receive only already-validated bbox tuples and are not the first
    geometry validation boundary
  - zero-area bbox failure
  - empty-description failure
  - metadata JSON-serializability failure
  - metadata does not alter training semantics
  - `category_id` and `category_name` are retained in object metadata but do not
    alter rendering or default losses
  - image path resolution relative to declaring root
  - image path existence failure
  - path-plus-stat image fingerprint behavior
  - `sample_limit` counts valid accepted examples while invalid pre-limit rows fail
  - invalid record fail-fast behavior
  - template renderer rejects raw JSON rows and accepts canonical `RawExample`
- Open questions:
  - exact helper/type names inside `src/data/`
- Recommendation: implement after config loading and before template rendering.

### `src/templates/`

- Status: approved
- Location: `src/templates/`
- Purpose: render validated `RawExample`s into Qwen chat messages,
  supervised assistant text, typed character spans, realized object order, and
  provenance without touching tokenization or image processing.
- Public interface:
  - `render_example(raw_example, template_config) -> RenderedExample`
  - exact function/class names may be adjusted at implementation approval, but
    rendering should stay package-owned and narrow
- Inputs:
  - `RawExample`
  - template config with `object_field_order`, `object_ordering`, and
    `assistant_format`
- Outputs:
  - `RenderedExample`
  - Qwen chat messages
  - renderer-owned prompt text
  - `supervised_response_text`
  - typed character spans over `supervised_response_text`
  - realized object order and object/span provenance
- Owned state:
  - `RenderedExample`
  - span record types
  - object renderer variants
  - description/control-token validation helpers
- Invariants:
  - the first canonical `assistant_format` is `object_box_closed`
  - default `object_field_order` is `desc_first`
  - V1 templates are English-only and expose no `template.language` knob
  - prompt text is short, fixed, English, and owned by `src/templates/`
  - default `object_ordering` is `source_order`, preserving validated source
    JSONL object order
  - legacy `sorted` is rejected by V1 config validation; migration tooling must
    rewrite old configs to canonical values before validation
  - future geometric sorting must be named `geometry_sorted` with an explicit
    key definition
  - `object_ordering: random` requires deterministic seed behavior and recorded
    realized order
  - canonical assistant object format is structural and compact, not free prose
    and not JSON-like text
  - object schema segments are concatenated without inserted separators
  - actual V1 object wrapper tokens are `<|object_ref_start|>` and
    `<|object_ref_end|>`
  - actual V1 box wrapper tokens are `<|box_start|>` and `<|box_end|>`
  - conceptual names such as object start/end are not new literal token strings
  - logical bbox coordinate order is `x1,y1,x2,y2`
  - coordinate targets render as tokenizer-visible strings such as
    `<|coord_123|>`, not raw integers
  - the four bbox coordinate-token strings are adjacent with no comma, space, or
    other separator
  - image placeholder is placed in the user message before task prompt text and
    never creates supervised `TokenAtom`s
  - rendered messages and `supervised_response_text` are stored separately
  - `RenderedExample` stores Qwen chat messages and the supervised assistant
    response; `qwen/encoding` owns exact processor text/full chat-template string
    materialization
  - assistant message content excludes the terminal `<|im_end|>\n` suffix when
    Qwen's processor/chat template inserts that suffix
  - `supervised_response_text` still ends with exactly one synthetic
    `<|im_end|>\n` suffix for alignment, and `qwen/encoding` maps that suffix to
    the single processor-inserted suffix in the full model text
  - duplicate assistant `<|im_end|>\n` suffixes fail before tokenization/packing
  - `templates/` owns human-readable assistant text and character spans;
    `qwen/encoding` owns tokenization, token ids, token spans, image-token
    expansion, and model-ready tensors
  - template fingerprint includes exact prompt text/version and renderer config
  - `RenderedExample` preserves source object ids, realized order, object spans,
    description spans, schema wrapper spans, bbox spans, coordinate-token spans,
    `eos_transition` spans, and `ignored_text` spans for trivial rendered suffix
    text such as the post-`<|im_end|>` newline
  - V1 rendered span kinds include at least `assistant_content`, `description`,
    `schema_token`, `coordinate_token`, `object`, `eos_transition`, and
    `ignored_text`
  - `RenderedSpan` minimally records `kind`, `char_start`, `char_end`, `text`,
    optional `object_id`, optional `field`, and source provenance
  - `RenderedSpan` offsets are zero-based Python string half-open character spans
    `[char_start, char_end)`
  - `RenderedSpan.text` is stored and must equal the supervised-response slice
  - nested spans are allowed for parent provenance; crossing spans are forbidden
  - every loss-bearing character is covered by exactly one token-type-bearing
    leaf span; parent spans such as `assistant_content` and `object` do not
    assign competing token types
  - descriptions are stripped at the boundary and internal newlines/tabs are
    normalized to single spaces before control-token validation
  - descriptions cannot contain Qwen special/control tokens or CoordExp
    wrapper/coordinate-token syntax unless a future escaping policy is approved
  - unsafe description content fails rendering; it is not stripped silently
  - `supervised_response_text` ends exactly with `<|im_end|>\n`
  - `<|im_end|>` is typed as `eos_transition`
  - trailing suffix newline is typed as `ignored_text`
  - templates do not tokenize, open images, call processors, create
    `TokenAtom`s, or build Qwen tensors
- Failure modes:
  - legacy `sorted` accepted by V1 config validation
  - hidden geometric sort under `sorted`
  - coordinate values rendered as raw integers
  - image placeholder placed in assistant supervision
  - unsafe Qwen/control tokens in descriptions
  - dormant `template.language` accepted by config validation
  - missing, crossing, or duplicate leaf typed spans
  - suffix omitted, duplicated, or appended only by hidden chat-template behavior
- Non-goals:
  - tokenizer alignment
  - Qwen processor calls
  - image loading
  - full Qwen chat-template string or processor-text materialization
  - `TokenAtom` creation
  - object detection schema generalization beyond approved V1 object format
  - arbitrary user-defined prompt templating language
- Debug/receipt artifacts:
  - `expected_rendered.json` for the smoke fixture
  - rendered example inspection output
  - template fingerprint
- Tests or parity checks:
  - `assistant_format: object_box_closed` rendering fixture
  - source-order rendering fixture
  - legacy `sorted` value rejected by config validation
  - no inserted separator between object schema segments
  - no inserted separator between bbox coordinate-token strings
  - coordinate-token string rendering
  - object/box wrapper span coverage
  - rendered messages and supervised response text stored separately
  - exact prompt text included in template fingerprint
  - `RenderedExample` does not store full Qwen chat-template/processor text
  - minimal `RenderedSpan` fields have source provenance and no token positions
  - `RenderedSpan.text` equals the half-open character slice
  - nested spans accepted and crossing spans rejected
  - token-type-bearing leaf spans are unique for each supervised character
  - description whitespace normalization before safety validation
  - image placeholder prompt-side only
  - unsafe description token failure
  - exact single `<|im_end|>\n` suffix and token-type spans; duplicate suffix
    fails before training
- Open questions:
  - none for the V1 template-rendering skeleton
- Recommendation: implement after `src/data/` and before Qwen encoding.

### `src/qwen/loading.py`

- Status: approved
- Location: `src/qwen/loading.py`
- Purpose: load and validate Qwen3-VL model components through one transparent
  setup entry before encoding, optimizer construction, or training begins.
- Public interface:
  - `load_qwen_components(config) -> QwenComponents`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - resolved model/config slices
  - `model.base_model`
  - optional `adapter.path`
  - adapter tuning config
  - `model.special_token_embeddings` for training configs
  - Qwen processor/tokenizer/model files
- Outputs:
  - frozen `QwenComponents`
  - loaded model
  - processor
  - tokenizer
  - model/tokenizer/processor identity records
  - adapter setup handle or identity
  - special-token embedding handle
  - setup receipt records
- Owned state:
  - `QwenComponents`
  - Qwen setup phase orchestration
  - model/tokenizer/processor identity helpers
  - tokenizer and processor preflight records
- Invariants:
  - `load_qwen_components(...)` is the public Qwen setup entry
  - `train.py` and `SupervisedTrainer` do not manually wire Qwen setup phases
  - setup phases are named and auditable
  - phase order is base model/processor/tokenizer load, tokenizer and processor
    identity validation, processor default resize-policy recording, call-time
    `do_resize=False` and single-image policy validation, adapter load/init,
    special-token embedding setup, setup receipt preparation, then optimizer
    construction
  - optimizer construction happens only after all trainable Qwen surfaces exist
  - `QwenComponents` is frozen/immutable after setup
  - `QwenComponents` includes model, processor, tokenizer, model config,
    tokenizer/vocab identity, processor policy, base identity, optional adapter
    identity, special-token embedding handle, token-vocab resolver or handle,
    and setup receipt handles
  - adapter config is read only from top-level `adapter:`; `model.adapter.*` is
    not a V1 config surface
  - adapter config may be omitted only by explicitly base-only, non-training
    utilities that do not initialize adapter tuning
  - training configs require `model.special_token_embeddings`; read-only
    non-training utilities may omit it because they are not constructing the
    trainable selected-token embedding surface
  - required coordinate, wrapper, image/control, `<|im_start|>`, and
    `<|im_end|>` tokens must already exist and resolve unambiguously
  - setup never mutates tokenizer vocab and never resizes model embeddings in V1
  - raw Qwen3-VL checkpoints without CoordExp coordinate-token vocabulary fail
    before data encoding or training
  - receipts are linked from `run_manifest.json`
- Failure modes:
  - tokenizer or processor identity cannot satisfy V1 contract
  - missing, split, or ambiguous required token
  - call-time `do_resize=False` path or no-resize image-grid admissibility cannot
    be verified
  - tokenizer mutation or embedding resize is attempted
  - adapter setup happens after optimizer construction
  - special-token embedding setup is deferred to runtime/trainer
  - downstream code mutates `QwenComponents`
- Non-goals:
  - data loading
  - template rendering
  - Qwen example encoding
  - optimizer group construction
  - TrainRuntime/device/distributed preparation
- Debug/receipt artifacts:
  - `reports/qwen_setup.json`
  - links to adapter and special-token embedding receipts when enabled
- Tests or parity checks:
  - local coordexp Qwen path loads with required token identity
  - raw Qwen3-VL base without coordinate tokens fails
  - `QwenComponents` is frozen after setup
  - setup phase order places adapter and special-token embedding setup before
    optimizer construction
  - tokenizer mutation and embedding resize are absent
  - setup receipt records `processor_default_do_resize`, `call_do_resize`,
    `patch_size`, `merge_size`, processor class/config identity, and no-resize
    policy
  - `do_resize=False` validation failure is explicit
- Open questions:
  - exact `QwenComponents` field names
  - exact setup receipt fields
- Recommendation: implement before `src/qwen/encoding.py` and before
  `src/optim/`.

### `src/qwen/adapters.py`

- Status: approved
- Location: `src/qwen/adapters.py`
- Purpose: discover and inject LoRA/dLoRA adapters into approved Qwen3-VL
  towers while making model surgery auditable before optimizer grouping.
- Public interface:
  - called by `load_qwen_components(...)`
  - exact helper/type names require the dLoRA source study before
    implementation
- Inputs:
  - loaded Qwen model
  - tokenizer/model identity
  - adapter config with `type`, optional `path`, `target_towers`,
    `target_modules`, `rank`, `alpha`, `dropout`, and `bias`
  - optional existing adapter checkpoint path
- Outputs:
  - model with adapter parameters installed or loaded
  - adapter setup handle/identity
  - `adapter_targets.json` inputs
  - trainable parameter names for optimizer validation
- Owned state:
  - Qwen tower target discovery
  - LoRA/dLoRA injection helpers
  - adapter load/init policy
  - adapter target receipt records
- Invariants:
  - adapter construction and injection live in `src/qwen/adapters.py`
  - `src/optim/` consumes adapter trainables but does not inject adapters
  - `training/` and `runtime/` do not own adapter model surgery
  - dLoRA is first-class behind explicit `type: dlora` only after the source
    study defines whether repo `dlora` maps to upstream DoRA/`use_dora`, a
    CoordExp-owned decomposed or dynamic LoRA mechanism, or another explicitly
    described variant
  - no silent fallback from dLoRA to standard LoRA
  - if `path` is present, load an existing adapter
  - if tuning is enabled and `path` is absent, initialize a fresh adapter
  - if inference/eval-only mode requests adapter type without path, fail fast
  - adapter fields live under top-level `adapter:` using `type`, `path`,
    `target_towers`, `target_modules`, `rank`, `alpha`, `dropout`, and `bias`
  - `model:` remains for base-model loading, tokenizer/processor policy, and
    Qwen identity; `model.adapter.*` is rejected
  - target towers are explicit and use `vision`, `aligner`, and `language`
  - `target_modules: all_linear` means all matched linear modules inside the
    explicit target towers
  - `target_modules: all_linear` excludes `lm_head` and output embedding/head
    modules by default; selected output-token behavior is owned by
    `src/qwen/special_token_embeddings.py`
  - no target exclusions in V1
  - any accepted adapter tower is a support promise and must be discoverable,
    injectable, trainable, optimizable, checkpointable, and recorded
  - dLoRA implementation requires a source study over MS-Swift, Transformers,
    PEFT, DoRA/`use_dora`, and Qwen module naming before implementation
  - before that source study and minimal round-trip probe are accepted,
    `adapter.type: dlora` fails validation rather than silently becoming LoRA or
    a placeholder implementation
  - a pre-dLoRA base-only or standard-LoRA first smoke is a separate user
    decision; the default adapter-enabled smoke remains blocked on the dLoRA
    definition/source-study gate
  - config exposes only verified adapter type/tower combinations; unverified
    combinations fail fast as unsupported
- Failure modes:
  - adapter type is unknown
  - dLoRA silently becomes standard LoRA
  - dLoRA is accepted without defining whether it means DoRA/`use_dora` or a
    CoordExp-owned mechanism
  - dLoRA config is accepted before the source-study gate is satisfied
  - target tower is accepted but no modules are found
  - adapter target discovery includes `lm_head` under ordinary
    `target_modules: all_linear`
  - unverified adapter type/tower combination runs as a best-effort match
  - adapter checkpoint is missing, incompatible, or ambiguous
  - adapter injection happens after optimizer construction
  - receipt omits matched modules or trainable parameter names
- Non-goals:
  - optimizer group construction
  - special-token embedding setup
  - full base-parameter training
  - adapter target exclusions in V1
- Debug/receipt artifacts:
  - `reports/adapter_targets.json` when LoRA or dLoRA is enabled
  - cross-reference to `optimizer_groups.json`
- Tests or parity checks:
  - dLoRA source-study gate rejects `adapter.type: dlora` until the accepted
    study, exact definition, and minimal round-trip probe exist
  - language dLoRA default setup after the source-study gate
  - LoRA and dLoRA target discovery for `language`, `aligner`, and `vision`
  - language `all_linear` target discovery records `lm_head` as excluded by
    policy while matching decoder/language-model internal linear modules
  - one forward/backward smoke for each supported tower/adapter type
  - optimizer group coverage for injected adapter parameters
  - adapter checkpoint metadata check
  - all-tower composition smoke for `[vision, aligner, language]`
- Open questions:
  - exact dLoRA definition and implementation mechanics, pending source study
  - exact adapter receipt fields
- Recommendation: perform the dLoRA source study first, then implement this
  card before the vertical smoke.

### `src/qwen/special_token_embeddings.py`

- Status: approved
- Location: `src/qwen/special_token_embeddings.py`
- Purpose: install selected-token full embedding trainables for coordinate and
  wrapper tokens while keeping base embedding/head matrices frozen.
- Public interface:
  - called by `load_qwen_components(...)`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - loaded Qwen model
  - tokenizer
  - base embedding/head parameters
  - resolved `model.special_token_embeddings`
  - fixed token groups `coordinate_tokens` and `wrapper_tokens`
- Outputs:
  - model-side special-token embedding trainables installed before optimizer
    construction
  - special-token embedding handle
  - trainable parameter names for optimizer validation
  - `special_token_embeddings.json` inputs
- Owned state:
  - selected-token embedding trainables
  - tied/untied detection
  - base-row initialization
  - checkpoint delta key mapping
  - special-token embedding receipt records
- Invariants:
  - setup initializes all selected-token trainables from the loaded base model's
    corresponding embedding/head rows
  - this initialization is preprocessing/model surgery, not TrainRuntime logic
  - after setup, TrainRuntime treats these as ordinary trainable model
    parameters
  - base embedding/head matrices remain frozen
  - no full embedding/head matrix is directly unfrozen in V1
  - no masked-gradient full-matrix training is used as the default mechanism
  - this is selected-token full embedding tuning, not LoRA
  - this config remains under `model.special_token_embeddings`; it is not moved
    under `adapter:` and V1 does not add a generic `trainables:` section
  - training configs require this surface; read-only non-training utilities may
    omit it because they do not construct trainable model-side parameters
  - all listed special tokens must already exist in the tokenizer
  - tokenizer vocab is not mutated and model embeddings are not resized
  - tied/untied status is detected from loaded model parameter identity and
    shape, not exposed as a user knob
  - tied models use one shared selected-token trainable and save
    `shared_embed_delta`
  - untied models use input/output selected-token trainables and save
    `input_embed_delta` and `output_embed_delta`
  - tied selected-token trainables affect both input embedding lookup and
    corresponding output-logit columns through model-side Qwen setup
  - implementation requires a source-study decision comparing a custom Qwen
    input/output wrapper pair, PEFT `TrainableTokensConfig` /
    `TrainableTokensModel`, and LoRA `trainable_token_indices` when LoRA is
    already present
  - currently preferred custom hook uses a small wrapper pair: input embedding
    wrapper adds selected-id deltas after frozen base embedding lookup; output
    head wrapper calls frozen base `lm_head` and scatter-adds selected-column
    correction `hidden @ delta.T`
  - tied models share `shared_embed_delta` across both wrappers in the custom
    path; untied models use separate input/output deltas
  - if PEFT trainable tokens are selected, the implementation must still prove
    tied input/output behavior and compact checkpoint payloads, and must avoid
    automatic full embedding save behavior caused by base/live vocab mismatch
  - public optimizer group is `token_embeddings`
  - checkpoint payload is compact special-token embedding delta state relative
    to validated base rows, not a full merged embedding/head export
  - `token_type_vocab.json` and
    `special_token_embeddings.json` remain separate receipts
- Failure modes:
  - missing, split, or ambiguous selected token
  - tokenizer mutation or embedding resize is attempted
  - selected-token trainables are not initialized from base rows
  - base embedding/head matrices become broadly trainable
  - tied model affects input lookup but not output-logit columns
  - output head wrapper mutates or saves full `lm_head.weight`
  - PEFT trainable-token path saves full embedding/head matrices through
    `save_embedding_layers` auto behavior
  - TrainRuntime has special-case token embedding logic
  - checkpoint load token identity differs from current tokenizer
- Non-goals:
  - LoRA/dLoRA adapter injection
  - optimizer group construction
  - TrainRuntime logic
  - full model merge/export
  - user-authored token-list overrides in V1
- Debug/receipt artifacts:
  - `reports/special_token_embeddings.json`
  - checkpoint special-token embedding delta metadata
  - cross-reference to `optimizer_groups.json`
- Tests or parity checks:
  - selected tokens resolve to expected single ids
  - source-study decision records custom wrapper versus PEFT trainable-token
    mechanism and why the chosen path satisfies the contract
  - trainables initialize from base embedding/head rows
  - base embedding/head matrices remain frozen
  - tied model shares one trainable surface for input and output behavior
  - perturbing one selected token delta changes that input lookup and output
    logit column, while non-selected tokens and frozen base matrices stay
    unchanged
  - untied model path records separate input/output delta keys
  - optimizer group sees only compact selected-token trainables
  - smoke fixture produces nonzero gradient coverage for coordinate and wrapper
    token groups
  - checkpoint save/load validates token strings and ids
- Open questions:
  - exact special-token embedding receipt fields
- Recommendation: implement during Qwen setup before `src/optim/` and keep all
  runtime surfaces unaware of token-embedding special cases.

### `src/qwen/encoding.py`

- Status: approved
- Location: `src/qwen/encoding.py`
- Purpose: convert pure `RenderedExample`s into validated Qwen-local
  `EncodedExample`s with token ids, visual processor payload, and exact
  supervision alignment.
- Public interface:
  - `QwenExampleEncoder.encode(rendered: RenderedExample) -> EncodedExample`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - `RenderedExample`
  - `QwenComponents` from `src/qwen/loading.py`
  - model config fields including `global_max_length`
- Outputs:
  - `EncodedExample`
  - local full-sequence `TokenSequence`
  - span-to-token alignment diagnostics
  - typed `QwenProcessorPayload` and grid metadata
- Owned state:
  - `EncodedExample`
  - `QwenProcessorPayload`
  - `ProcessorTextPlan`
  - piecewise rendered-chat to processor-text offset map
  - image-token expansion records
  - tokenizer special-token validation receipt
  - alignment diagnostics
- Invariants:
  - defining smoke/default examples use
    `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
  - raw Qwen3-VL checkpoints without CoordExp coordinate-token vocabulary are
    invalid V1 bases
  - encoder validates user-authored model paths by tokenizer preflight once at
    Qwen setup or encoder construction, before iterating examples
  - token preflight writes a compact token-identity section in `reports/qwen_setup.json`
  - image processing happens inside `qwen/`, not `data/` or `templates/`
  - if `RawExample.image.width` or `RawExample.image.height` is declared, decoded
    image size must match during Qwen encoding; mismatch fails fast with
    declared and actual size
  - `EncodedExample.input_ids` is a plain CPU `list[int]`; tensor materialization
    happens later in packing/forward builders
  - visual processor fields are stored in typed `QwenProcessorPayload`, not loose
    HF kwargs
  - `QwenProcessorPayload` stores CPU tensors or arrays for fields such as
    `pixel_values` and `image_grid_thw`, plus explicit shape and dtype metadata
  - the encoder calls the HF image processor for visual payloads, reconstructs
    processor text locally, and calls the tokenizer directly for ids/offsets
  - parity tests compare the local split path against `Qwen3VLProcessor.__call__`
    for the defining smoke fixture and a small targeted set
  - parity covers `input_ids`, `image_grid_thw`, image-placeholder expansion, and
    relevant masks/fields
  - V1 is single-image only
  - exactly one image reference and exactly one prompt-side image placeholder
    are required
  - actual image-processor calls use `do_resize=False`; processor default
    `do_resize` may be true and is recorded rather than trusted
  - no-resize image-grid admissibility is checked before processor call,
    including decoded image size, optional declared image size, `patch_size`,
    `merge_size`, and effective divisibility by `patch_size * merge_size` unless
    an installed-processor probe approves a broader rule
  - V1 assumes images are offline prepared for the no-resize Qwen path, such as
    the existing `rescale_32` data family; raw arbitrary-size images are out of
    scope until a pad-or-reject data policy is approved
  - no-resize preflight includes an approved pixel or visual-token budget, since
    Qwen smart-resize max-pixel clamping is bypassed when `do_resize=False`
  - visual token counts and pack admission use actual no-resize
    `image_grid_thw` or a local computation proven equivalent to that payload;
    helper paths that may apply smart-resize semantics are forbidden as the
    source of truth unless an installed-version probe approves them
  - incompatible no-resize image dimensions fail with typed
    `EncodingContractError` before an opaque upstream reshape/runtime error
  - required wrapper/control/image tokens map to exactly one tokenizer id
  - the required object wrapper literals are `<|object_ref_start|>` and
    `<|object_ref_end|>`; nearby aliases such as `<|object_start|>` and
    `<|object_end|>` are rejected because they split into ordinary text tokens
  - all V1 coordinate tokens `<|coord_0|>` through `<|coord_999|>` map to
    exactly one tokenizer id
  - coordinate-token validation runs before example encoding
  - processor offsets are interpreted in processor-text space after Qwen
    image-placeholder expansion
  - `ProcessorTextPlan` stores rendered chat text, processor text, image-token
    expansion records, and offset conversion helpers
  - offset conversion is represented by piecewise range mappings plus explicit
    inserted/repeated image-token expansion records
  - rendered assistant spans are first placed into full rendered-chat text,
    then converted through the rendered-chat to processor-text offset map
  - span conversion verifies slice equality, whole-token boundaries, and
    exact decode round-trip with `skip_special_tokens=false` and tokenizer
    cleanup disabled
  - fallback alignment, if needed, is deterministic and exact over processor
    text; approximate decoded-string matching is forbidden
  - local token positions are zero-based over the full Qwen input sequence,
    including prompt, image placeholder tokens, assistant tokens, and suffix
  - `TokenAtom.target_position` is local full-sequence target position; loss
    code owns causal shifting
  - `EncodedExample` preserves the complete
    `RenderedSpan -> TokenSpan -> TokenAtom` trace
  - one `EncodedExample` must not exceed `global_max_length`
  - V1 raises `EncodingContractError` rather than truncating, skipping, or
    splitting an over-length encoded example
  - dense `labels` are not stored by default and are derived only for parity or
    debugging
  - `expected_tokenization.json`, when present, stores token ids/spans,
    placeholder counts, `image_grid_thw`, payload shapes/dtypes, and optional
    image or payload hashes, but never full `pixel_values`
- Failure modes:
  - raw Qwen3-VL base without coordinate-token vocabulary
  - declared image width/height mismatch after decoding
  - tokenizer path without single-id coordinate tokens
  - tokenizer preflight delayed until after dataset iteration begins
  - processor visual payload represented as an untyped HF kwargs dict
  - `EncodedExample.input_ids` materialized as a GPU tensor before packing
  - tokenizer offsets aligned against unexpanded rendered chat text
  - processor-text reconstruction diverges from `Qwen3VLProcessor.__call__`
  - partial-token span boundary
  - decoded span text differs from processor-text slice
  - missing `RenderedSpan -> TokenSpan -> TokenAtom` trace
  - zero-image, multi-image, or multi-placeholder examples
  - processor image-grid/placeholder mismatch
  - image/video payload shape outside V1 single-image support
  - encoded example exceeds `global_max_length`
  - expected-tokenization fixture dumps dense `pixel_values`
- Non-goals:
  - packing
  - model forward
  - loss computation
  - dense labels as canonical supervision
  - video or multi-image support
- Debug/receipt artifacts:
  - compact token-identity section in `reports/qwen_setup.json` with coordinate
    range/hash checks and sampled ids by default
  - processor-text expansion diagnostics
  - alignment diagnostics
  - future `expected_tokenization.json` in the smoke fixture with no full
    `pixel_values`
- Tests or parity checks:
  - local model
    `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
    has single-id wrappers and coordinate tokens
  - alias literals such as `<|object_start|>` and `<|object_end|>` are not
    accepted as schema-wrapper tokens
  - token preflight runs before dataset iteration
  - `EncodedExample.input_ids` remains a CPU `list[int]`
  - `QwenProcessorPayload` is typed and records shape/dtype metadata
  - adjacent coordinate tokens align as four single-token spans
  - no-resize pack-cost logic rejects use of Qwen helper token-count paths that
    can apply smart-resize assumptions
  - local split image-processor/tokenizer path matches `Qwen3VLProcessor.__call__`
    on smoke and targeted fixtures
  - declared image width/height mismatch fails during Qwen encoding
  - compatible no-resize fixture image succeeds with `do_resize=False`
  - incompatible synthetic image fails through the local typed contract before
    upstream processor reshape
  - processor offsets work for a no-expansion image case
  - processor offsets work for an expanded image placeholder case through the
    rendered-chat to processor-text offset map
  - `expected_tokenization.json` omits full `pixel_values`
  - `RenderedSpan -> TokenSpan -> TokenAtom` trace is present for supervised
    spans
  - exact `RenderedSpan` to token-span inversion checks
  - exact decoded span round-trip with cleanup disabled
  - `EncodedExample` over `global_max_length` fails
  - zero-image, multi-image, and multi-placeholder examples fail
  - optional dense-label parity derives from `TokenSequence`
- Open questions:
  - none for the V1 encoding skeleton
- Recommendation: implement after `src/templates/` and before `packing/`.

### `src/packing/`

- Status: approved
- Location: `src/packing/`
- Purpose: combine validated `EncodedExample`s into no-padding physical
  `PackedSequence`s while preserving segment isolation, supervision
  traceability, and pack diagnostics.
- Public interface:
  - `pack_examples(encoded_examples, packing_config) -> Iterable[PackedSequence]`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - ordered stream of `EncodedExample`s
  - packing config with `global_max_length`
- Outputs:
  - `PackedSequence`
  - `PackedSegment` table
  - physical-position `TokenSequence`
  - local-to-packed mapping records
  - pack diagnostics and summary inputs
- Owned state:
  - `PackedSequence`
  - `PackedSegment`
  - `PackedLayout`
  - local-to-packed/inverse mapping helpers
  - pack plan summary records
- Invariants:
  - `PackedSequence` is model-agnostic physical layout, not a Qwen forward kwargs
    dict
  - `PackedSequence` contains concatenated `input_ids`, remapped
    physical-position `TokenSequence`, segment table, local-to-packed mappings,
    visual payload references or concat metadata, and provenance
  - the V1 supervision-ledger role is carried by `TokenSequence` plus
    `TokenAtom`/`TokenSpan`; do not add a parallel `SupervisionLedger` class
    unless future implementation proves a concrete need
  - any future materialized `SupervisionLedger` must be a compiled view of
    `TokenSequence`, not an independent source of truth
  - packing changes physical positions, not semantic ownership
  - packing may remap and validate supervision, but must not invent semantic
    supervision or infer token meaning from packed offsets
  - loss terms consume packed positions and supervision metadata through
    `LossContext`, not dense labels or raw example order
  - packing uses greedy streaming admission
  - the packer never sorts, shuffles, samples, or otherwise reorders examples
  - upstream sampler/data stream owns example order
  - `global_max_length` counts full encoded `input_ids` length after Qwen
    image-placeholder expansion
  - prompt tokens, image placeholder tokens, assistant tokens, and
    `<|im_end|>\n` suffix all count toward `global_max_length`
  - assistant-only length, supervised-token count, and raw pre-expansion text
    length are not pack-admission budgets
  - each `PackedSegment` records physical `start`, `end`, `example_id`, local
    encoded length, epoch index when training-stream context is available,
    realized data-order metadata when available, image placeholder ranges, and
    provenance
  - `cu_seq_lens_q/k` are derived later from the segment table
  - supervision remapping adds segment start offset to every
    `TokenAtom.target_position` and every `TokenSpan` boundary
  - remapped supervision validates inverse mapping back to the source
    `EncodedExample`
  - Qwen visual payloads are preserved in segment order and remain traceable
    through placeholder ranges and `image_grid_thw`
  - trainer code does not concatenate arbitrary HF processor keys
  - dense labels are not canonical and are derived only for parity/debugging
- Failure modes:
  - over-length single `EncodedExample`
  - hidden example reordering
  - semantic supervision invented during packing
  - loss meaning inferred from packed offsets or dense labels
  - separate supervision-ledger object diverges from `TokenSequence`
  - remapped supervision cannot invert to source example/local position
  - segment table and concatenated `input_ids` disagree
  - visual payload order diverges from segment order
  - missing image placeholder range or `image_grid_thw` provenance
- Non-goals:
  - Qwen `position_ids`
  - FlashAttention varlen tensors
  - model forward kwargs assembly
  - best-fit or bucketed packing
  - random packing
  - dense labels as canonical supervision
- Debug/receipt artifacts:
  - `reports/pack_plan.json`
  - optional verbose pack JSONL trace
- Tests or parity checks:
  - greedy append then commit behavior
  - no hidden reorder from ordered encoded stream
  - `global_max_length` uses full expanded `input_ids` length
  - segment table derives expected starts/ends
  - local-to-packed and packed-to-local inverse mapping
  - supervision positions and spans remap exactly
  - packed `TokenSequence` remains the source of loss semantics after remapping
  - optional dense labels derived from packed `TokenSequence` match parity
    expectations but do not own semantics
  - visual payload references stay in segment order
  - `reports/pack_plan.json` includes summary counts such as `packs_per_epoch`,
    requested and actual pack presentations, `tail_fill_pack_count`,
    `resolved_max_steps`, `effective_batch_size`, world size,
    `resolved_grad_accum_steps`, utilization summary, order policies, seed and
    fingerprint inputs
  - verbose smoke/debug pack traces include per-pack example ids, segment
    starts/ends, token lengths, supervised-token counts by role, image-token
    counts, utilization, and fingerprint inputs
- Open questions:
  - exact internal helper/type names
- Recommendation: implement after `src/qwen/encoding.py` and before
  `src/qwen/forward.py`.

### `src/qwen/forward.py`

- Status: approved
- Location: `src/qwen/forward.py`
- Purpose: convert model-agnostic packed physical layout into typed Qwen3-VL
  forward inputs, call the HF Qwen model, and return typed outputs for
  CoordExp loss computation.
- Public interface:
  - `build_qwen_forward_inputs(packed_sequence, qwen_components, forward_requirements) -> QwenForwardInputs`
  - `run_qwen_forward(model, qwen_inputs) -> ModelOutputs`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - `PackedSequence`
  - `PackedSegment` table
  - Qwen visual payloads and `image_grid_thw`
  - loaded `QwenComponents`
  - aggregated forward requirements derived from the enabled loss plan
- Outputs:
  - `QwenForwardInputs`
  - typed `ModelOutputs` with full logits
  - `qwen_forward_contract.json` inputs
- Owned state:
  - `QwenForwardInputs`
  - Qwen position-id construction helpers
  - FlashAttention varlen metadata helpers
  - forward requirement validation helpers
  - forward contract receipt records
- Invariants:
  - `QwenForwardInputs` is typed, not a generic HF kwargs dict
  - it contains packed `input_ids`, visual payloads, `image_grid_thw`, explicit
    `position_ids`, FA2 varlen metadata, Qwen-only kwargs, and trace metadata
  - V1 training forward uses batch size 1 with one long packed row
  - `input_ids` shape is `[1, seq]`
  - multiple examples are represented as isolated packed segments, not padded
    batch rows
  - Qwen `position_ids` are built under `src/qwen/` from `PackedSequence`,
    `PackedSegment` boundaries, expanded `input_ids`, and `image_grid_thw`
  - HF packed-training position inference is not used
  - MS-Swift's Qwen-VL packing/position-id flow is a reference for per-segment
    MRoPE computation followed by concatenation; it is not imported as a
    training dependency
  - production forward passes `labels=None`
  - non-`None` labels are rejected at the wrapper boundary
  - `use_cache=False`
  - full logits are requested; `logits_to_keep=1` is not used in V1 training
  - wrapper asserts the returned output object/dataclass shape; do not require a
    rote `return_dict=True` kwarg when the installed Qwen3-VL forward signature
    does not declare it
  - V1 uses the `input_ids` plus visual payload path, not a precomputed
    `inputs_embeds` shortcut; the installed Qwen3-VL path injects DeepStack
    visual features through model internals, and shortcuts are unsupported until
    a dedicated hook proves equivalent behavior
  - hidden states, attentions, vision states, and other optional outputs are
    disabled unless an enabled approved loss declares them through the aggregated
    forward requirements
  - text-token activations are named `text_hidden_states`
  - first approved text activation capture is `lm_head_input`, meaning the
    post-final-norm hidden state consumed by `lm_head`
  - future pre-final-norm decoder activation capture is named
    `decoder_layer_post` and requires explicit `layer: -1` or `layer: N`
  - visual activations are reserved as `vision_activations` and are unsupported
    in V1
  - unsupported `vision_activations` requirements fail during loss setup or
    preflight before training starts
  - image placeholder token positions are not treated as visual-row positions
  - Qwen forward does not rely on vague `final`/`last` activation selector names
  - `decoder_layer_post` requires an explicitly implemented and smoke-tested
    wrapper hook
  - V1 protected defaults `BaseTokenCE` and `TokenTypeGateLoss` require logits
    only
  - manual config flags such as `output_hidden_states` are not the primary way
    to satisfy loss requirements
  - returned `ModelOutputs` must satisfy the declared requirements; missing
    outputs fail immediately with the requesting loss name and missing
    requirement
  - FA2 segment isolation is derived from `PackedSegment` boundaries as
    `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`
  - varlen metadata dtype, shape, start/end values, and Python-int max lengths
    are validated before forward
  - packed training forward passes `attention_mask=None` for segment isolation
    and relies on explicit varlen metadata plus position ids
  - forward-contract smoke captures the actual Transformers/FlashAttention
    branch and proves the explicit varlen path was used
  - ordinary dense 2D padding masks do not encode segment boundaries
  - position resets alone are insufficient for segment isolation
- Failure modes:
  - batch shape is padded or multi-row in V1 training
  - labels reach the HF model
  - model-side loss is produced or consumed
  - `use_cache=True` in training forward
  - full logits are missing
  - implementation relies on `return_dict=True` instead of validating the actual
    installed output object
  - precomputed `inputs_embeds` bypasses Qwen visual replacement or DeepStack
    behavior
  - declared forward requirement is missing from `ModelOutputs`
  - Qwen position ids are inferred globally by HF for packed training
  - FA2 varlen metadata disagrees with segment boundaries
  - a stray dense `attention_mask` routes packed training through a padded/unpad
    branch instead of explicit varlen metadata
  - dense 2D attention mask encodes segment isolation
  - hidden-state capture is assumed without an approved hook
  - hidden states or attentions are enabled by manual config drift rather than
    loss requirements
  - visual activation requirement reaches forward after preflight instead of
    failing early
  - visual losses treat language-sequence placeholder positions as vision
    feature rows
  - hidden-state selector uses regex or ambiguous names such as `final`/`last`
  - `decoder_layer_post` is requested before a wrapper hook exists
- Non-goals:
  - loss computation
  - trainer loop
  - hidden-state loss implementation
  - visual embedding replacement reimplementation
  - generation/rollout decoding
- Debug/receipt artifacts:
  - `debug/qwen_forward_contract.json`
- Tests or parity checks:
  - `[1, seq]` packed-row input shape
  - `labels=None` at model boundary and non-`None` labels rejected
  - `use_cache=False` and no `past_key_values`
  - full-logit shape matches `input_ids.shape[:2]`
  - explicit `position_ids` shape is accepted by HF Qwen
  - packed MRoPE fixture proves per-segment position ids are computed before
    concatenation and reset at `PackedSegment` boundaries
  - `cu_seq_lens_q/k` and `max_length_q/k` derive from `PackedSegment`
    boundaries and have valid dtype/shape
  - branch-level FA2 probe or monkeypatch proves actual `attention_mask is None`,
    explicit `cu_seq_lens_q/k`, Python-int max lengths, and no padded/unpad
    branch
  - no dense 2D padding mask with zeros is used for segment isolation
  - visual placeholder counts/ranges and `image_grid_thw` agree
  - contract receipt records input/logit shapes, labels/cache status,
    position-id shape, segment boundaries, FA2 metadata, visual payload shapes,
    image placeholder counts/ranges, `image_grid_thw`, and disabled/deferred
    features
- Open questions:
  - exact helper/type names
- Recommendation: implement after `src/packing/` and before `src/losses/`.

### `src/losses/`

- Status: approved
- Location: `src/losses/`
- Purpose: compute all repo-owned training objectives from Qwen full logits and
  packed physical supervision without using HF model-side loss.
- Public interface:
  - `LossRunner.compute(model_outputs, packed_sequence, loss_config, normalizers) -> LossBundle`
  - `build_loss_context(packed_sequence, model_outputs) -> LossContext`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - `ModelOutputs` with full logits
  - `PackedSequence` with physical-position `TokenSequence`
  - resolved loss config
  - tokenizer/vocab metadata and segment provenance
- Outputs:
  - `LossBundle`
  - per-term `LossTermResult`s
  - scalar metrics and diagnostics
  - `loss_plan.json` inputs
- Owned state:
  - `LossRunner`
  - `LossContext`
  - `ForwardRequirements` or equivalent aggregated model-output requirement
    contract
  - `LossBundle`
  - `LossTermResult`
  - `BaseTokenCE`
  - `TokenTypeGateLoss`
  - `TokenVocabGroups`
  - loss selection/reduction helpers
- Invariants:
  - `LossRunner` is the only aggregate loss executor
  - trainer code does not manually call individual loss functions
  - HF/model-side loss is disabled and ignored
  - `LossContext` owns logits reference, packed `TokenSequence`,
    tokenizer/vocab metadata, segment/provenance maps, selection helpers, fp32
    selected-logit helpers, and metric helpers
  - `PackedSequence` stores inspectable CPU-side supervision records and
    provenance; `LossContext` builds the tensorized hot-path view for the
    current step
  - `LossContext` owns device placement, index-tensor construction, and compact
    tensor/ragged views needed by loss terms or metrics
  - extending hidden-state, coordinate, object-level, or rollout-derived losses
    should happen by extending `LossContext` selections and compiled views, not
    by making `packing/` aware of each new loss
  - enabled loss terms declare model-output requirements before forward
  - loss setup aggregates term requirements into a single forward-requirements
    contract consumed by `src/qwen/forward.py`
  - V1 protected defaults `BaseTokenCE` and `TokenTypeGateLoss` require logits
    only
  - hidden states, attentions, vision states, and other optional outputs are
    opt-in and requested only by enabled approved losses
  - activation requirement names distinguish `text_hidden_states` from future
    `vision_activations`
  - text hidden-state activation selectors use semantic capture names plus
    explicit layer indices when needed, not regex
  - first approved text hidden-state capture is `lm_head_input`
  - `lm_head_input` means the post-final-norm text hidden state consumed by
    `lm_head`
  - future `decoder_layer_post` capture requires explicit `layer: -1` or
    `layer: N` and a smoke-tested Qwen wrapper hook
  - prefix/exact-name matching is the V1 default for optimizer/LR groups,
    freeze/trainable params, and checkpoint extra-state keys
  - regex-style matching is allowed only for adapter target discovery when the
    adapter source study or PEFT/MS-Swift parity requires it, and is not used
    for loss activation selectors
  - `vision_activations` is a reserved future source and unsupported in V1
  - losses requiring unsupported `vision_activations` fail during loss setup or
    preflight before training starts
  - future visual activation support must name the capture point explicitly,
    such as `vision_pre_projector`, `vision_post_projector`, or
    `post_scatter_language_embedding`
  - missing required model outputs fail with the loss name and missing
    requirement, not as a zero/skipped loss
  - `LossContext` caches compiled index/mask tensors for the current step on
    `logits.device`
  - cached compiled tensors may include target/logits positions, token-type ids,
    segment ids, span ids, object ids, and span/object grouping maps
  - selected fp32 logits are not cached in V1
  - loss terms are pure consumers of `LossContext`; they do not mutate
    `PackedSequence`, `LossContext`, `ModelOutputs`, model parameters, or other
    loss terms
  - metrics reuse `LossContext` selections and detached views instead of
    recomputing independent masks or alignment
  - `loss_plan.json` records activation requirements by source, including
    logits, text hidden-state selectors, and whether vision activations were
    unsupported or requested
  - `loss_plan.json` records both semantic activation selector and
    implementation capture path
  - invalid selector names, unsupported captures, out-of-range layer indices,
    and empty selected positions fail before training
  - use `position` for sequence-token indices and `span` for contiguous token
    groups; reserve `row` for a true JSONL row or a real vision feature row
  - individual loss terms expose `name`, `required_inputs`,
    `select_atoms(context)`, and `compute(context) -> LossTermResult`
  - each `LossTermResult` exposes numerator, denominator, scalar loss,
    selected count, skipped count, dtype, and diagnostics
  - `LossRunner` applies configured weights centrally
  - default token-wise reduction is `segment_balanced`
  - `segment_balanced` computes mean loss over eligible atoms per segment, then
    mean over eligible segments
  - under gradient accumulation, `segment_balanced` is computed over the full
    planned optimizer step/effective batch, not independently per `MicroStep`
  - protected V1 token-wise losses receive planned-step denominator metadata
    before backward, so uneven segment counts across micro-steps do not become
    pack-balanced objective weighting
  - `LossNormalizers` is a planned-step-window object with immutable
    denominator/count metadata for protected V1 terms before backward
  - distributed denominator counts needed by protected terms are reduced across
    ranks before any rank starts backward for the planned step
  - `LossBundle.total_loss` is already the planned-step-normalized micro
    contribution for protected terms; trainer/runtime must not apply a blind
    `resolved_grad_accum_steps` divisor
  - `BaseTokenCE` and `TokenTypeGateLoss` use the same `segment_balanced`
    denominator by default
  - one global mean over all eligible packed tokens is a diagnostic view by
    default, not the optimized scalar
  - for a given term, segments with zero eligible atoms are excluded from that
    term's denominator
  - protected default terms `BaseTokenCE` and `TokenTypeGateLoss` fail fast on
    zero eligible atoms
  - optional future auxiliary terms may explicitly allow zero-eligible no-op
  - future object-level losses default to object-balanced reduction:
    per-object loss, mean over objects, then mean over segments
  - V1 first smoke does not expose broad denominator-mode config; term-level
    denominator configs require a later concrete experiment
  - `LossTermResult` denominator detail includes selected atom count, eligible
    segment count, skipped segment count, denominator mode, numerator,
    denominator, raw scalar, and weighted scalar
  - metrics may report token-weighted diagnostic means for MS-Swift/HF
    comparison, but diagnostic denominator names must be explicit
  - token-type vocabulary groups are resolved during loss setup from tokenizer
    identity and approved token families
  - losses consume resolved `TokenVocabGroups`, not hard-coded numeric ranges
  - `BaseTokenCE` uses full-vocabulary CE
  - `TokenTypeGateLoss` is a separate protected default term, not CE internals
  - `TokenTypeGateLoss` applies to every supervised `TokenAtom` with a resolved
    target token type
  - V1 target token types are `desc_text`, `schema`, `coordinate`, and `eos`
  - `TokenVocabGroups` contains those target groups plus blocked Qwen
    special/control groups
  - gate math is group-mass CE:
    `logsumexp(all_logits) - logsumexp(target_group_logits)`
  - blocked Qwen/control groups stay in the gate denominator as always-negative
    probability mass
  - gate loss uses the same length-invariant, segment-balanced reduction as
    other token-wise losses
  - gate metrics include aggregate loss, per-type loss, selected counts,
    target-type mass, off-type mass, blocked-special mass, and top-1 type
    accuracy
  - protected expected target types with zero eligible atoms fail unless a future
    approved config declares that type optional
  - auxiliary terms may use narrowed vocab selections only when approved
  - selected differentiable logits are upcast to fp32 through `LossContext`
  - no V1 selected-logit cache
  - `LossContext` reserves future selected-hidden-state helper surfaces, but V1
    rejects hidden-state-dependent loss configs until the wrapper hook exists
  - metrics detach selected logits and may use model dtype for order-only
    metrics such as top-k accuracy
  - bad sample/step conditions are recorded as warnings or diagnostics on the
    current planned step; they do not retime eval/checkpoint/logging schedules
  - unsafe non-finite losses or gradients do not produce corrupted optimizer
    updates and are not silently zeroed
  - scalar finite status is checked before backward; unsafe scalar loss skips
    backward for the planned-step window, clears accumulated gradients, records
    diagnostics, and advances schedule state according to the planned-step
    policy
  - gradient finite/overflow status is checked after backward and before
    clipping or optimizer stepping
  - V1 baseline is not old production-objective parity: prior IoU/CIoU-aware
    coordinate soft-CE and object/role/image-balanced reductions are known
    future auxiliary-term candidates, not implemented merely because base CE
    and token-type gate pass
- Failure modes:
  - protected CE or gate has zero eligible atoms
  - expected protected token type has zero eligible atoms without explicit
    optional status
  - term tries to consume logits directly without `LossContext` helper
  - term builds its own token-position alignment instead of using `LossContext`
  - term mutates context, model outputs, or another term's state
  - metric path recomputes supervision masks independently from `LossContext`
  - term performs its own weighting/logging policy
  - token-wise loss silently uses global token-weighted mean as the optimized
    scalar
  - already-reduced micro-step means are averaged by
    `resolved_grad_accum_steps`, making the objective pack-balanced instead of
    segment-balanced
  - backend gradient accumulation scales an already planned-step-normalized
    scalar a second time
  - CE and gate use different denominators without an approved experiment
  - zero-eligible segments are counted as zero loss
  - token-weighted diagnostic metrics are reported without explicit diagnostic
    names
  - broad denominator config appears in first smoke before a concrete experiment
  - gate loss is merged into base CE internals
  - hidden-state term is enabled before approved capture hook exists
  - loss requirement/output mismatch is treated as a skipped term
  - hidden-state activation selector uses regex or ambiguous `final`/`last`
    names
  - invalid hidden-state selector warns and proceeds
  - token-type group is missing or hard-coded ad hoc
  - blocked Qwen/control tokens are omitted from gate accounting
  - non-finite loss is silently zeroed or treated as a successful update
  - non-finite scalar loss reaches backward before the central finite gate
  - bad-step diagnostics mutate the planned schedule
- Non-goals:
  - model forward
  - backward/optimizer stepping
  - checkpoint writing
  - hidden-state loss implementation in V1
  - arbitrary user-defined groupby expressions for metrics
- Debug/receipt artifacts:
  - `reports/loss_plan.json`
  - `reports/token_type_vocab.json`
  - non-finite loss diagnostics when triggered
- Tests or parity checks:
  - `LossRunner.compute(...) -> LossBundle`
  - `LossContext` target-position to logits-position shift stays within segment
  - `BaseTokenCE` consumes exact shifted full-logit rows
  - `TokenTypeGateLoss` uses resolved vocab groups and reports per-type metrics
    for `desc_text`, `schema`, `coordinate`, and `eos`
  - gate loss computes group-mass CE in fp32 selected-logit math
  - blocked Qwen/control probability mass is included in diagnostics
  - protected zero-eligible terms fail fast
  - protected expected target types with zero eligible atoms fail unless
    explicitly optional
  - optional zero-eligible auxiliary no-op behavior is explicit
  - segment-balanced denominator behavior computes per-segment eligible-atom
    means then averages eligible segments
  - two-micro-step planned-step fixture with unequal segment counts matches the
    explicit mean over all eligible planned-step segments, not the mean of pack
    means
  - same denominator fixture verifies raw single-process and Accelerate paths do
    not differ by an extra accumulation divisor; DeepSpeed gets the equivalent
    check before execution support is claimed
  - injected non-finite scalar before backward produces no backward call, clears
    window gradients, records diagnostics, and emits an unsafe planned-step
    status
  - CE and gate use the same segment-balanced denominator
  - zero-eligible segments are excluded from term denominator
  - token-weighted diagnostic means are reported with explicit diagnostic names
    and do not affect optimized scalar
  - `LossTermResult` records selected atom count, eligible segment count,
    skipped segment count, denominator mode, numerator, denominator, raw scalar,
    and weighted scalar
  - fp32 selected-logit math for differentiable terms
  - no selected-logit cache in V1
  - hidden-state-dependent loss config is rejected while reserved helper
    surfaces remain non-runnable
  - top-1/top-5 metrics detach logits
  - `reports/loss_plan.json` records enabled terms, weights, eligible counts,
    denominator modes/details, token-role counts, vocab groups,
    dtype/upcast status, bad-step diagnostic policy, and metric definitions
  - loss-continuity note names deferred coordinate soft-CE and object-balanced
    reduction candidates so first-smoke success is not misread as old
    production detection-objective parity
- Open questions:
  - exact internal helper/type names
- Recommendation: implement after `src/qwen/forward.py` and before
  `src/training/`.

### `src/optim/`

- Status: approved
- Location: `src/optim/`
- Purpose: construct optimizer and scheduler from explicit trainable-surface
  groups while proving every trainable parameter has an intentional LR and
  weight-decay assignment.
- Public interface:
  - `build_optimizer(model, trainable_records, optimizer_config) -> OptimizerBundle`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - model after Qwen loading, adapter injection, and special-token embedding
    setup
  - small trainable parameter records or compact setup object from Qwen,
    adapter, and special-token embedding setup
  - resolved optimizer config
  - Qwen semantic group maps from `src/qwen/optim_groups.py`
- Outputs:
  - optimizer
  - scheduler
  - `OptimizerBundle`
  - `optimizer_groups.json` inputs
  - trainable parameter validation summary
- Owned state:
  - `OptimizerBundle`
  - optimizer group resolver
  - decay/no-decay splitter
  - trainable parameter validator
  - optimizer group receipt records
- Invariants:
  - optimizer construction lives in `src/optim/`, not `training/` or `qwen/`
  - do not introduce a public `TrainableRegistry` class in V1; plain typed
    trainable records are enough unless multiple real trainable surfaces prove
    that a registry abstraction is needed
  - Qwen-specific group maps live near Qwen and are consumed by `src/optim/`
  - public trainable-surface taxonomy is `vision`, `aligner`, `language`,
    `adapter`, `token_embeddings`, and reserved `auxiliary_modules`
  - Qwen-internal names such as `vit`, `mlp`, and `llm` may appear in mapping
    metadata but are not public config vocabulary
  - optimizer/LR groups, freeze/trainable parameter sets, and extra checkpoint
    state keys use prefix and exact-name selectors in V1
  - regex match rules are limited to adapter target discovery when needed and
    are recorded in adapter receipts
  - every trainable parameter must match exactly one semantic group
  - every trainable semantic group must have explicit LR and weight decay in
    the final resolved config
  - no global/default LR or weight-decay fallback is used for trainable
    parameters
  - unmatched trainable parameters fail before optimizer construction
  - duplicate-matched trainable parameters fail before optimizer construction
  - declared trainable groups that match no parameters fail unless they
    correspond to an explicitly non-trainable/disabled surface
  - frozen parameters are excluded from optimizer groups but summarized in the
    receipt
  - tied/shared parameters are deduplicated by object identity
  - model setup validates tied input-embedding/lm-head state by runtime tensor
    identity or pointer equality, not config assumption
  - tied/untied embedding-head status is recorded in the model/setup receipt
  - semantic groups are split into decay/no-decay buckets after semantic
    assignment
  - bias and normalization parameters are excluded from decay by the approved
    decay splitter
  - `token_embeddings` covers compact selected-token embedding trainables
    initialized from base rows, not full base embedding/head matrices
  - `token_embeddings` includes coordinate tokens plus the four schema wrapper
    tokens
  - selected special-token embeddings are fully trainable embedding deltas, not
    LoRA over the embedding/head
  - tied models use `shared_embed_delta`; untied models use
    `input_embed_delta` and `output_embed_delta`
  - special-token embedding deltas are saved as safetensors, not full
    embedding/head matrices
  - for tied models, special-token embedding trainables affect input lookup and
    corresponding output-logit columns through Qwen setup
  - V1 does not support full base-model parameter fine-tuning
  - trainable base weights outside approved adapter modules and approved
    special-token embedding deltas fail unless a future design card adds that
    mode
  - default adapter training profile is language-tower dLoRA, but all exposed
    LoRA/dLoRA target choices for `vision`, `aligner`, and `language` must be
    discoverable, trainable, groupable, and recorded
  - optimizer schema exposes training-sensitive AdamW knobs such as beta1, beta2,
    epsilon, and optimizer-specific kwargs rather than only LR and weight decay
  - scheduler schema exposes scheduler name, warmup ratio/steps, and
    scheduler-specific kwargs
  - optimizer/scheduler schema is source-studied against MS-Swift,
    Transformers, and adapter tooling before implementation; local names may
    differ, but important performance knobs should not be silently omitted
  - scheduler is built by optimizer config and its behavior is recorded against
    the planned step schedule through the trainer/runtime order
  - `OptimizerBundle` is not device/distributed prepared; `TrainRuntime` owns
    preparation
- Failure modes:
  - implicit global LR or weight-decay fallback
  - trainable parameter with no semantic group
  - trainable parameter matched by more than one group
  - declared group matching no intended trainables
  - public config uses Qwen-internal `vit`/`mlp`/`llm` names instead of semantic
    group names
  - regex match rule affects adapter trainables but is absent from adapter
    receipts
  - tied embedding/head status inferred from config without runtime identity
    validation
  - AdamW beta/epsilon or scheduler kwargs unavailable despite being needed for a
    training-performance reproduction
  - adapter tower accepted by schema but not discoverable or recorded
  - special-token embedding trainable treated as LoRA, full-matrix unfreezing,
    or TrainRuntime logic
  - full base-model parameter is trainable in V1 without an approved design card
  - special-token embedding delta checkpoint saves full embedding/head matrices
  - scheduler/eval/checkpoint behavior is driven by an unrecorded successful-step
    counter instead of the planned step schedule
  - optimizer bundle prepared before `TrainRuntime.prepare(...)`
- Non-goals:
  - adapter injection or target discovery
  - Qwen model surgery
  - loss computation
  - runtime/distributed wrapping
  - checkpoint schema ownership
- Debug/receipt artifacts:
  - `reports/optimizer_groups.json`
  - trainable parameter summary
  - cross-reference to `adapter_targets.json` when adapters are enabled
  - cross-reference to `special_token_embeddings.json`
- Tests or parity checks:
  - each trainable parameter appears in exactly one optimizer group
  - unmatched and duplicate-matched trainables fail
  - missing LR and missing weight decay fail
  - unused trainable group fails unless explicitly disabled/non-trainable
  - frozen parameters are absent from optimizer groups but appear in summaries
  - tied parameter identity is deduplicated
  - input-embedding/lm-head tied status is validated by tensor identity/pointer
    equality and recorded
  - trainable base weights outside approved adapters and special-token
    embeddings fail in V1
  - decay/no-decay split excludes bias and normalization parameters
  - optimizer selector tests accept prefix/exact-name forms and reject regex
    forms outside adapter target discovery
  - public semantic groups remain `vision`, `aligner`, `language`, `adapter`,
    `token_embeddings`, and `auxiliary_modules`
  - language dLoRA default profile builds optimizer groups after the dLoRA
    source-study gate
  - configured vision/aligner/language LoRA and dLoRA targets build and record
    groups when supported
  - special-token embedding trainables receive the `token_embeddings` group
  - special-token embedding deltas are saved as safetensors, not full
    embedding/head matrices
  - AdamW beta1/beta2/epsilon and optimizer kwargs round-trip into the optimizer
    construction receipt
  - scheduler kwargs and warmup policy round-trip into the scheduler receipt
  - scheduler object is present but not stepped during optimizer construction
  - scheduler behavior is reported against planned step ids
  - `reports/optimizer_groups.json` records group name, LR, weight decay,
    parameter names, shapes, dtypes, counts, trainable status, match rule,
    trainable source, adapter type, special-token embedding status, and
    validation summary
- Open questions:
  - exact internal helper/type names
  - exact optimizer receipt fields
- Recommendation: implement after Qwen adapter/special-token embedding setup
  and before `src/training/`.

### `src/artifacts/`

- Status: approved
- Location: `src/artifacts/`
- Purpose: own the small run artifact surface: run directory creation,
  manifest writing, compact receipt registration, and rank-safe artifact writes.
- Public interface:
  - small `RunArtifactManager`-style object or equivalent functions
  - `CheckpointWriter` from `src/artifacts/checkpoints.py`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - resolved run config
  - generated run id
  - runtime rank/main-process helpers
  - approved boundary receipt paths and optional summaries
  - checkpoint/eval/metric artifact references
- Outputs:
  - `run_dir`
  - top-level `run_manifest.json`
  - registered receipt paths and optional summaries
  - rank-safe artifact write helpers
- Owned state:
  - run directory layout
  - structured manifest state
  - atomic JSON write helper
  - receipt registration helper
  - checkpoint writer module for checkpoint schema, payload layout, and pointer
    files
- Invariants:
  - artifact surface stays small; no event bus, plugin registry, database, or
    broad artifact/receipt framework in V1
  - runtime receipts exist only for fragile or high-value boundaries, not for
    every subsystem by default
  - required V1 receipts are `configs/resolved.yaml`, `configs/resolved.json`,
    `run_manifest.json`, `resolved_step_schedule.json`,
    `reports/qwen_setup.json`, `reports/pack_plan.json`,
    `reports/loss_plan.json`, `reports/token_type_vocab.json`,
    `reports/special_token_embeddings.json`, `reports/optimizer_groups.json`,
    `eval/forward/step-<planned_step_id>.json` when `eval.forward` runs, and
    smoke/failure/explicit-debug `debug/qwen_forward_contract.json`
  - `src/train.py` and `SupervisedTrainer` do not manually invent artifact
    layout or manifest mutation policy
  - checkpoint schema ownership lives in `src/artifacts/checkpoints.py`, not
    `src/runtime/` or `src/training/`
  - `SupervisedTrainer` calls the checkpoint writer; it does not write
    checkpoint payloads inline
  - `TrainRuntime` provides rank-safe save helpers and unwrap/prepare utilities,
    but it does not own checkpoint metadata or pointer-file semantics
  - `run_manifest.json` is a structured current-run index, not a metric event
    log
  - `run_manifest.json` links important artifacts; it does not embed full
    configs, full receipts, package inventories, git changed-file inventories,
    or environment ledgers in V1
  - minimum manifest top-level sections are `run_id`, `run_name`, `run_dir`,
    `status`, `created_at`, `updated_at`, `configs`, `resolution`,
    `runtime_identity`, `schedule`, `receipts`, `metrics`, `checkpoints`,
    `eval`, `runtime`, and `warnings`
  - manifest status values are `initializing`, `dry_run`, `running`, `failed`,
    and `completed`
  - paths inside the run directory are stored run-dir-relative; external paths
    are stored absolute
  - compact `runtime_identity` records repo/package label, git commit when
    available, dirty/unknown status, Python version, and selected dependency
    versions for PyTorch, Transformers, Accelerate, DeepSpeed, PEFT,
    safetensors, and flash-attn when installed
  - resolved config fingerprint covers behaviorally meaningful resolved config
    content and excludes `run_id`, timestamps, package versions, git state,
    runtime world size, and other launch-context metadata
  - manifest is rewritten atomically after major lifecycle events
  - boundary owners own receipt content and register path plus short status or
    summary in the manifest when useful
  - setup receipts live under `run_dir/reports/`
  - execution-contract and failure-local diagnostics live under
    `run_dir/debug/`
  - `run_manifest.json` links configs, receipt artifacts, debug artifacts, metric
    files, checkpoints, eval summary paths, and enabled cache identities when
    present
  - `eval.forward` summaries live under `run_dir/eval/forward/` and are linked
    run-dir-relatively from the manifest
  - `run_manifest.json` may store tiny latest/best eval scalar snapshots, but it
    does not embed full eval summaries
  - `run_manifest.json` links `resolved_step_schedule.json`, planned-step
    checkpoint/eval events, optimizer-update status, warning/non-finite status,
    checkpoint aliases, and checkpoint-to-eval links when available
  - `resolved_step_schedule.json` is one precomputed schedule artifact with
    separate event lists for `eval.forward`, checkpoint, logging, and final
    events
  - resolved schedule events record `planned_step_id`, `event`,
    `trigger_reasons`, `source_config_path`, `deduped_from`, and `required`
  - canonical cadence config paths are `checkpoint.every_fraction`,
    `checkpoint.steps`, `checkpoint.save_final`,
    `eval.forward.every_fraction`, `eval.forward.steps`,
    `training.logging.every_fraction`, and `training.logging.steps`; V1 does
    not add `save_steps`, `eval_steps`, or `logging_steps` aliases
  - rank safety is applied through the artifact layer using `TrainRuntime`
    rank/main-process helpers
  - individual subsystems do not each invent rank-writing policy
  - named setup receipts may be atomically replaced by the same subsystem during
    setup
  - metric event history and checkpoint directories are append-only in ordinary
    operation
  - no global overwrite knob
  - run directory creation is atomic; same-second timestamp collisions append a
    short suffix and retry rather than overwriting or failing nondeterministically
  - no generic report base class, formal JSON Schema set, mandatory shared report
    header, or `src/reports/` framework in V1
  - receipts are typed by their owning Python modules and verified by smoke/tests
  - receipts stay bounded and summary-first; large details go to explicit debug
    sidecars in smoke, failure, or debug mode
  - cache receipts are conditional; uncached V1 smoke runs do not need
    `reports/cache.json`
  - if a future cache stage is enabled, its subsystem registers a receipt or
    manifest link through the artifact manager
- Failure modes:
  - manifest only written at the end
  - metrics embedded wholesale into manifest
  - every subsystem writes artifacts on every rank
  - artifact manager serializes raw subsystem internals instead of registering
    boundary-owned receipts
  - all JSON dumped at run root
  - run id collision overwrites an existing run or fails nondeterministically
  - broad artifact framework introduced before need
  - every subsystem is forced to emit a receipt by default
  - receipt grows into a full trace dump by default
- Non-goals:
  - metric event schema
  - checkpoint payload schema outside `src/artifacts/checkpoints.py`
  - receipt contents owned by boundary modules
  - cache storage implementation
  - external artifact database or service
- Debug/receipt artifacts:
  - `run_manifest.json`
  - artifact write diagnostics when useful
- Tests or parity checks:
  - run directory creation follows approved layout
  - frozen timestamp or parallel-create test produces unique run directories
    without overwrite
  - manifest atomic rewrite behavior
  - receipt path plus optional summary registration
  - rank guard allows ordinary artifact writes only on main process
  - setup receipt replacement is atomic
  - metric events and checkpoints are not overwritten by manifest refresh
  - manifest records checkpoint/eval status for warning-only and
    update-skipped planned steps
  - smoke schedule fixture with `resolved_max_steps: 5` and explicit
    `eval.forward.steps: [2, 4]` materializes two eval event ids, while
    checkpoint cadence plus `checkpoint.save_final` materializes the required
    final checkpoint event at step 5; all events record trigger reasons and
    deduplicate collisions without duplicate writes
  - manifest contains required top-level sections and compact identity fields
    without full package inventories, git diffs, changed-file inventories, or
    environment dumps
  - one resolved schedule file contains separate event lists for eval,
    checkpoint, logging, and final events
  - `eval.forward` summary path such as `eval/forward/step-4.json` exists when
    scheduled eval runs and is linked from `run_manifest.json`
  - uncached runs are valid with no cache receipt
  - no generic receipt/report framework is required for the first smoke
  - future cache receipts can be linked from manifest when a cache stage exists
- Open questions:
  - exact helper/type names
- Recommendation: implement before `src/training/` so trainer wiring uses the
  same artifact surface from the first smoke.

### `src/eval/forward.py`

- Status: approved
- Location: `src/eval/forward.py`
- Purpose: run teacher-forced packed forward evaluation through the same
  supervised path as training, without backward or optimizer updates.
- Public interface:
  - `run_forward_eval(...) -> EvalForwardResult`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - optional `data.eval_path`
  - `eval.forward` config and optional sample limit
  - same render, encode, pack, Qwen forward, and loss components used by train
  - planned step id and optional linked checkpoint/update status
- Outputs:
  - metric events with split `eval.forward`
  - compact eval summary at `eval/forward/step-<planned_step_id>.json`
    linked from `run_manifest.json`
  - checkpoint-to-eval links when scheduled near a checkpoint
- Owned state:
  - forward-eval loop
  - eval summary records
  - eval metric/result aggregation
- Invariants:
  - `eval.forward` is teacher-forced packed forward evaluation, not generation
  - it uses the same render, encode, pack, Qwen forward wrapper, `LossContext`,
    and `LossRunner` path as training under `torch.no_grad()`
  - it uses the same packing implementation and `global_max_length`
  - eval may set its own sample limit but does not use padded batches or a
    one-example-only fallback by default
  - if `data.eval_path` is absent, scheduled `eval.forward` is disabled unless
    an explicit smoke fixture provides eval data
  - train JSONL is not implicitly reused or runtime-split for eval
  - metric events use split `eval.forward`
  - `eval.forward` emits the same core supervised metrics as train wherever the
    same logits/supervision context exists: `loss/total`, `loss/base_ce`,
    `loss/token_type_gate`, top-level `acc_top1`, top-level `acc_top5`,
    `pack/supervised_tokens`, supervised atom count, contributing segment
    count, physical length, and effective pack cost when available
  - train-only runtime metrics such as LR, grad norm, accumulation state,
    backward timing, and optimizer-step details are not eval metrics
  - linked checkpoint/update status may be recorded as eval event context
  - each eval summary path uses an unpadded planned step id, such as
    `eval/forward/step-4.json`
  - eval summaries contain planned step id, split, eval dataset identity, sample
    count, pack count, aggregate metrics, metric event refs, linked checkpoint
    pointer or directory when present, warning/update status context, and
    bounded timing
  - `run_manifest.json` links eval summary paths and may store a tiny
    latest/best scalar snapshot, but does not embed full eval summaries
  - cadence/final schedule collisions are de-duplicated; the eval event may
    record multiple trigger reasons
  - warning-only and update-skipped planned steps still run scheduled
    `eval.forward`, with status recorded in the summary/context
  - no large per-token logits or prediction dumps are written by default
  - per-pack or per-example eval metrics are not written by default; debug mode
    may write bounded example ids or failed-pack diagnostics
  - `eval.inference` stays offline and separate from `eval.forward`
- Failure modes:
  - eval bypasses packing or uses padded HF batches
  - eval silently reuses train data when `data.eval_path` is absent
  - eval emits only `acc_top1` and drops the train-aligned metric set
  - eval summary paths use zero-padded step ids or omit planned step ids
  - eval summaries are embedded wholesale into `run_manifest.json`
  - cadence/final collisions run duplicate eval passes for the same planned step
  - update-skipped planned steps silently skip scheduled eval
  - eval writes full logits/predictions by default
  - generation/inference decode is mixed into `eval.forward`
  - eval owns a separate loss implementation from `LossRunner`
- Non-goals:
  - generation/inference evaluation
  - rollout evaluation
  - per-token prediction dumps by default
  - checkpoint selection policy beyond exposing metrics and summaries
- Debug/receipt artifacts:
  - compact forward-eval summary such as `eval/forward/step-4.json`
  - metric events under split `eval.forward`
- Tests or parity checks:
  - `eval.forward` uses the same `LossRunner` path as train
  - eval metrics include the train-aligned core supervised metric set
  - missing `data.eval_path` disables scheduled eval unless a smoke fixture
    explicitly supplies eval data
  - eval uses packed sequences and does not create padded batches
  - no backward/optimizer update happens during eval
  - compact eval summary links to planned step and checkpoint status when present
  - `run_manifest.json` links eval summary paths without embedding full summaries
  - cadence/final schedule collision produces one eval summary with multiple
    reasons when needed
  - update-skipped planned step still writes scheduled eval summary with status
- Open questions:
  - exact eval summary JSON schema fields
- Recommendation: implement after `src/losses/` and `src/metrics/`, then wire
  into `SupervisedTrainer` scheduled eval hooks.

### `src/metrics/`

- Status: approved
- Location: `src/metrics/`
- Purpose: record small typed metric events and summaries for train,
  `eval.forward`, and future `eval.inference` without becoming a logging
  framework.
- Public interface:
  - `MetricEvent`
  - `MetricSink`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - scalar metrics from loss, trainer, optimizer, runtime, eval, and receipts
  - runtime rank/main-process helpers
  - artifact manager paths
- Outputs:
  - `metrics/events.jsonl`
  - `metrics/summary.json`
  - optional secondary TensorBoard or CSV exports after the JSON source of truth
- Owned state:
  - metric event schema
  - JSONL event writer
  - summary accumulator/writer
  - optional secondary export adapters when approved
- Invariants:
  - canonical metrics storage is `metrics/events.jsonl` plus
    `metrics/summary.json`
  - TensorBoard and CSV are optional secondary exports, not source of truth
  - `MetricEvent` minimally contains planned-step `step`, `split`, `name`,
    `value`, `scope`, optional `counts`, optional `context`, and timestamp
  - events are keyed by planned step id, not successful optimizer-update count
    or wall-clock timestamp
  - events may record `optimizer_update_applied`, warning/non-finite status, and
    event kind
  - canonical splits are `train`, `eval.forward`, and future `eval.inference`
  - `eval.forward` events include the triggering planned step and linked
    checkpoint/update status when available
  - metric scope distinguishes aggregate, token type, role, span kind,
    optimizer group, runtime, and other approved slices
  - core metric names use a stable slash-name registry: `loss/total`,
    `loss/base_ce`, `loss/token_type_gate`, top-level `acc_top1`, top-level
    `acc_top5`, `lr/<group>`, `grad_norm/<group>`, `pack/utilization`, and
    `pack/supervised_tokens`
  - eval events use ordinary metric names with `split="eval.forward"` or future
    `split="eval.inference"`; strings such as `eval.forward/acc_top1:max` are
    selector expressions over `split/name:mode`, not stored metric names
  - ordinary stored loss metrics contain weighted optimized scalar values only
  - do not emit routine `loss/<term>/raw` and `loss/<term>/weighted` metric
    pairs
  - raw/unweighted term values may remain in `LossTermResult`, `loss_plan.json`,
    or targeted diagnostics, but they are not standard metric events
  - token-level top-k accuracy is not named `acc_top1/base_ce` or
    `acc_top5/base_ce`; it is a global supervised-token health metric
  - `acc_top1` and `acc_top5` remain top-level core metric names
  - token-weighted diagnostic loss views use explicit suffixes such as
    `loss/base_ce/token_weighted_diag` and
    `loss/token_type_gate/token_weighted_diag`
  - type-sliced scalar names use `/by_type/<token_type>`, for example
    `loss/base_ce/by_type/coordinate`, `acc_top1/by_type/coordinate`, and
    `loss/token_type_gate/by_type/schema`
  - reducer diagnostics use explicit names such as `loss/base_ce/segment_mean`,
    `loss/base_ce/segment_count`, and `loss/base_ce/token_weighted_diag`
  - named counters use the `count/` namespace, including
    `count/supervised_atoms`, `count/eligible_segments`,
    `count/skipped_segments`, `count/packs`, and `count/examples`
  - metric names never contain dynamic `example_id`, `object_id`, or arbitrary
    span text
  - events stay scalar and compact; tensor dumps are debug artifacts, not metric
    events
  - allowed event values are scalar numbers, booleans, short strings/enums, and
    bounded small dictionaries for structured status
  - mandatory token metrics when logits/supervision exist include CE loss,
    `acc_top1`, `acc_top5`, supervised atom count, contributing segment count,
    and effective pack cost
  - modules return typed metric results or compact metric dictionaries;
    `MetricSink` owns writing, summary update, rank safety, and optional
    secondary TensorBoard/CSV exports
  - rank safety is applied through `MetricSink` using `TrainRuntime`
    rank/main-process helpers
  - no arbitrary user-defined groupby expressions in V1
  - metric history is append-only during a run
- Failure modes:
  - arbitrary metric dicts from each subsystem
  - only TensorBoard metrics with no JSONL source of truth
  - giant manifest-embedded metric history
  - every rank writes duplicate metric events
  - metric events carry tensors or large token-wise dumps
  - metric events carry rendered text, token traces, arrays, or unbounded
    dictionaries
  - metric framework grows plugin routing before need
  - raw/unweighted and weighted loss metric pairs are emitted routinely
  - token-weighted diagnostic loss is named like an optimized scalar
  - type slices rely only on hidden context and have no scan-friendly slash name
  - dynamic example ids, object ids, or span text appear in metric names
- Non-goals:
  - loss computation
  - receipt content ownership
  - checkpoint selection policy beyond exposing scalar metrics
  - external experiment tracking service
- Debug/receipt artifacts:
  - `metrics/events.jsonl`
  - `metrics/summary.json`
- Tests or parity checks:
  - metric event schema validation
  - JSONL append behavior
  - summary update behavior
  - `acc_top1` and `acc_top5` are accepted as top-level metric names and
    `acc_top1/base_ce` or `acc_top5/base_ce` are rejected
  - rank guard prevents duplicate ordinary metric writes
  - mandatory CE/top-k metric emission when loss context exists
  - stored loss metrics use weighted optimized scalar values
  - raw/unweighted loss values do not appear as routine metric keys
  - token-weighted diagnostic names use `/token_weighted_diag`
  - type-sliced metric names use `/by_type/<token_type>`
  - named counters use the `count/` namespace
  - dynamic example/object/span identifiers are rejected in metric names
  - `eval.forward` metric event includes planned-step and checkpoint/update
    context when triggered by the training schedule
  - best-checkpoint selector resolves `eval.forward/acc_top1:max` from
    `(split="eval.forward", name="acc_top1")` events without storing a
    double-prefixed metric name
  - optional TensorBoard/CSV export does not replace JSON source of truth
- Open questions:
  - exact summary JSON schema
  - exact optional secondary export policy
- Recommendation: implement before `src/training/` and keep the first version
  deliberately small.

### `src/training/stream.py`

- Status: approved
- Location: `src/training/stream.py`
- Purpose: build the deterministic supervised train stream that bridges data
  order, rendering/encoding, greedy packing, preflight pack counting, tail-fill,
  and planned optimizer-step grouping.
- Public interface:
  - internal builder consumed by `build_supervised_trainer(config)`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - resolved train config
  - raw-example loader or encoded-example builder
  - `data.train_order`
  - `training.epochs`
  - `training.max_steps`
  - `training.effective_batch_size`
  - `resolved_grad_accum_steps`
  - global seed-derived data-order seed
  - `src/packing/` greedy packer
- Outputs:
  - deterministic stream of `MicroStep`s
  - preflight count summary for `reports/pack_plan.json`
  - planned optimizer-step grouping metadata
  - tail-fill summary
- Owned state:
  - `MicroStep` type, kept in `stream.py` unless a tiny `types.py` becomes
    clearer during implementation approval
  - epoch/example order iterator
  - preflight pack-count pass
  - tail-fill planner
  - optimizer-step grouping view
  - compact stream receipts
- Invariants:
  - this module owns example-order policy across epochs; `src/packing/` only
    consumes ordered encoded examples
  - this module owns planned optimizer-step grouping and assigns
    `planned_step_id` plus `micro_step_id`
  - `src/packing/` does not define `MicroStep` and does not know optimizer-step
    grouping or accumulation
  - default `data.train_order` is `shuffle`; `source_order` is allowed for
    smoke/debug
  - shuffle order is deterministic from the global seed-derived data-order seed
  - `data.train_order` never changes `template.object_ordering`
  - epoch-led production runs perform a deterministic preflight pack-count pass
    before optimizer state changes
  - preflight uses the same strict data/render/encode/pack validation as training
  - preflight may stream and discard heavy payloads, but it is not a cheap
    estimator and must exercise real length/supervision-affecting boundaries,
    including image open/processor behavior, tokenizer identity,
    processor-text expansion, span alignment, Qwen placeholder counts, and
    greedy packing when those stages exist for the training path
  - bad samples during preflight fail before training and are not skipped
  - preflight computes `packs_per_epoch`, requested pack presentations, actual
    pack presentations, `tail_fill_pack_count`, `resolved_max_steps`, and
    schedule inputs
  - tail-fill consumes from the next deterministic epoch stream using the same
    `data.train_order` policy and next seed/order state
  - packing may cross epoch boundaries, including tail-fill
  - every packed segment records epoch index, source example id, realized
    data-order metadata, and pack-local span metadata
  - V1 does not force pack commits at epoch boundaries
  - `training.max_steps` mode consumes a deterministic continuous pack stream
    until `max_steps * effective_batch_size` global pack presentations are
    consumed
  - stream grouping is defined first in global pack-presentation space, then
    deterministically partitioned by rank
  - each planned step owns exactly `effective_batch_size`
    `global_pack_presentation_id`s across all ranks
  - rank-local sharding is deterministic: each rank receives exactly
    `resolved_grad_accum_steps` `MicroStep`s per planned step
  - `MicroStep` records include `planned_step_id`, `rank`, `world_size`,
    `rank_local_micro_step_id`, `global_pack_presentation_id`, compact pack id
    or fingerprint, and replay/tail-fill status
  - unmarked duplicate global pack-presentation ids within a planned step fail
  - the trainer consumes the produced `MicroStep` stream and does not own data
    order, epoch replay, preflight counting, packing, or tail-fill semantics
  - `MicroStep` may contain `PackedSequence`, `QwenForwardInputs`, schedule
    context, and trace metadata
  - `MicroStep` does not contain `ModelOutputs`, `LossBundle`, optimizer state,
    raw config, raw examples, or historical execution dumps
  - normal runs do not serialize every `MicroStep`; only compact pack/step
    summaries are emitted, with full micro-step dumps limited to explicit debug
    sidecars
- Failure modes:
  - trainer owns example order or tail-fill
  - trainer assigns planned step or micro-step ids
  - packer shuffles, samples, or owns epoch semantics
  - packer defines `MicroStep` or owns optimizer-step grouping
  - every rank consumes the same local stream and duplicates global pack
    presentations silently
  - rank-local streams diverge in planned-step boundary or micro-step count
  - bad samples are skipped in preflight
  - preflight is a loose estimator instead of the real strict pipeline
  - preflight only counts raw rows or approximate text lengths and misses Qwen
    image expansion, tokenizer/span alignment, or placeholder failures
  - tail-fill uses a separate sampler or nondeterministic order
  - pack commits are forced at epoch boundaries without approval
  - detailed pack traces become mandatory for every production run
  - every `MicroStep` is serialized as a standard artifact
- Non-goals:
  - raw JSONL schema validation
  - renderer semantics
  - tokenizer/processor semantics
  - greedy pack admission internals
  - Qwen forward construction
  - loss computation
- Debug/receipt artifacts:
  - `reports/pack_plan.json`
  - optional verbose pack JSONL trace for smoke/debug
  - optional debug-only micro-step dumps
- Tests or parity checks:
  - default shuffle is deterministic from seed
  - `source_order` smoke stream is deterministic and inspectable
  - object order inside examples is unchanged by `data.train_order`
  - preflight count matches training stream count
  - epoch-led tail-fill completes the final full effective batch
  - tail-fill consumes the next deterministic epoch stream
  - packing can cross epoch boundaries while preserving segment epoch metadata
  - bad preflight sample fails before optimizer setup changes state
  - two-rank deterministic stream fixture with `effective_batch_size=4` and
    `resolved_grad_accum_steps=2` proves each planned step has four unique global
    pack presentations across ranks
  - `reports/pack_plan.json` records summary counts without requiring full token
    dumps
  - `MicroStep` ids are assigned by `src/training/stream.py`, not by
    `SupervisedTrainer`
  - `MicroStep` excludes `ModelOutputs`, `LossBundle`, optimizer state, raw
    config, raw examples, and execution history
  - standard run artifacts do not contain full serialized `MicroStep` objects
- Open questions:
  - exact internal helper/type names
- Recommendation: implement after `src/packing/` and before
  `SupervisedTrainer`, because schedule materialization depends on the stream
  count.

### `src/train.py`

- Status: approved
- Location: `src/train.py`
- Purpose: provide the thin professional training entrypoint without becoming
  the orchestration brain.
- Public interface:
  - `python -m src.train --config PATH`
  - `python -m src.train --config PATH --dry-run`
- Inputs:
  - authored runnable YAML config path
  - optional `--dry-run`
- Outputs:
  - process exit status
  - run directory
  - `run_manifest.json`
  - `configs/resolved.yaml`
  - `configs/resolved.json`
  - available dry-run plan receipts when `--dry-run` is used
- Owned state:
  - argument parser for `--config` and `--dry-run`
  - `main(argv: Sequence[str] | None = None) -> int`
  - `if __name__ == "__main__": raise SystemExit(main())`
- Invariants:
  - config-first launch surface; research settings live in YAML, not CLI flags
  - no V1 CLI config overrides such as `--set key=value`
  - no separate `--validate-only`; `--dry-run` is the only launch-level preflight
    mode
  - `train.py` parses args, loads/resolves config, creates `run_dir`, writes the
    resolved config artifacts and initial manifest, then delegates to
    package-owned builders
  - `train.py` does not wire model, optimizer, stream, loss, metric, checkpoint,
    runtime, or eval internals inline
  - `--dry-run` validates cheap path/schema/setup contracts and writes available
    plan receipts that do not require CUDA or model forward when possible
  - `--dry-run` exits before optimizer/model mutation and is not a fake training
    run
  - dry-run manifest uses `status: dry_run`
  - heavy/model-mutating receipts are represented as `skipped` entries with
    short reasons rather than silently omitted
  - dry-run writes no checkpoints, training metric events, eval summaries,
    adapter weights, or special-token embedding deltas
  - domain errors fail fast with clear exception types and stack traces; no broad
    catch-all hides debugging context in V1
- Failure modes:
  - stable CLI flags for LR, loss weights, model path, run root, max steps,
    packing, or runtime settings
  - Hydra-style CLI override syntax in V1
  - separate `--validate-only`, `--plan`, or `--inspect` mode on `train.py`
  - production `DryRunTrainer`, `FakeSupervisedTrainer`, or `StubTrainer`
  - dry-run creates fake checkpoints, fake metric events, or fake eval summaries
  - `train.py` manually wires the full object graph inline
  - resolved config artifacts are written only after heavy setup succeeds
  - broad exception swallowing or automatic phase retries
- Non-goals:
  - general command dispatcher
  - `eval.py`, `infer.py`, or `visualize.py`
  - config inheritance tracing UI beyond delegating to config helpers or future
    `trace_config.py`
  - fake training execution
- Debug/receipt artifacts:
  - `run_manifest.json`
  - `configs/resolved.yaml`
  - `configs/resolved.json`
  - dry-run plan receipts when available
- Tests or parity checks:
  - argv parsing accepts `--config` and optional `--dry-run`
  - CLI config overrides are rejected
  - `--validate-only` is rejected
  - resolved config artifacts are written before trainer construction
  - normal launch delegates to `build_supervised_trainer(config).train()`
  - dry-run path exits before trainer training or optimizer/model mutation
  - dry-run manifest has `status: dry_run`, required config links, skipped heavy
    receipt entries, and no checkpoint/eval/training-metric artifacts
- Open questions: none for V1 launch surface.
- Recommendation: implement after `src/config/` and artifact run-directory
  helpers, then wire to `build_supervised_trainer(config)` after the trainer card
  is ready.

### `src/training/`

- Status: approved
- Location: `src/training/`
- Purpose: run the supervised optimization loop over streamed packed
  `MicroStep`s while delegating data, Qwen forward, objective semantics,
  runtime mechanics, metrics, checkpointing, and evaluation to their owning
  components.
- Public interface:
  - `build_supervised_trainer(config) -> SupervisedTrainer`
  - `SupervisedTrainer.train()`
  - `SupervisedTrainer.evaluate_forward()`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - resolved train config
  - streaming iterator of `MicroStep`s
  - `QwenForwardInputs`
  - `LossRunner`
  - `TrainRuntime`
  - metric, checkpoint, and schedule components
- Outputs:
  - planned-step metric events
  - planned-step diagnostics and optional micro-step diagnostics when useful
  - checkpoint/eval calls
  - run lifecycle artifacts
- Owned state:
  - `SupervisedTrainer`
  - `TrainRuntime` integration
  - train-loop schedule context
  - planned-step diagnostic records
- Invariants:
  - `SupervisedTrainer` owns step iteration, gradient accumulation, backward,
    optimizer/scheduler stepping, eval hooks, checkpoint calls, metric emission,
    non-finite runtime policy, and run lifecycle
  - `train.py` is an entrypoint and does not manually wire the full object graph
    inline
  - `train.py` parses args, resolves config, creates the run/artifact root, and
    delegates object construction to package-owned builders
  - trainer code does not own objective semantics
  - trainer code does not own example-order, tail-fill, or planned-step grouping
    semantics
  - trainer calls `LossRunner` and treats `LossBundle.total_loss` as the
    backward-ready planned-step-normalized contribution for the current
    `MicroStep`
  - trainer does not special-case CE, token-type gate, or auxiliary losses
  - each micro-step consumes one `MicroStep` over one `PackedSequence`
  - `MicroStep` means one physical packed forward/backward micro-step per rank
  - plain `step` in public training docs, metrics, checkpoint cadence, and
    schedules means planned optimizer-update step
  - `planned_step_id` refers to planned optimizer-update id, not micro-step count
  - `MicroStep` contains `planned_step_id`, `micro_step_id`, `PackedSequence`,
    `QwenForwardInputs`, schedule context, and trace metadata
  - `MicroStep` does not contain `ModelOutputs`, `LossBundle`, raw examples, or
    raw config
  - `SupervisedTrainer` may log counts and ids surfaced by `MicroStep` or
    `PackedSequence`, but it does not inspect raw examples or reinterpret segment
    semantics
  - fixed micro-step spine is packed sequence fetch, tensor/device move,
    Qwen forward with `labels=None`, `ModelOutputs`, `LossContext`,
    `LossRunner.compute(...)`, pre-backward finite check, top-level metric computation,
    planned-step-normalized backward, and optimizer-boundary handling when due
  - gradient accumulation controls how many rank-local `MicroStep`s form a
    planned optimizer step; it is not a universal scalar divisor for
    `segment_balanced` protected losses
  - protected token-wise losses use planned-step normalizers supplied before
    backward so segment-balanced objectives remain segment-balanced across the
    effective batch
  - backend accumulation must not divide protected planned-step-normalized loss
    scalars a second time; raw single-process, Accelerate, and later DeepSpeed
    paths each need explicit scaling semantics
  - `resolved_grad_accum_steps` is derived from `training.effective_batch_size`
    and actual runtime world size, not authored directly
  - scheduler steps after planned optimizer-update steps, not after every
    `MicroStep`
  - scheduler time advances on every planned optimizer-step boundary, including
    update-skipped non-finite guard steps, so LR remains planned-step-relative
  - gradient clipping happens after accumulation is ready and before optimizer
    step
  - final optimizer steps are full effective-batch updates; epoch-led runs use
    deterministic tail-fill when needed
  - `training.max_steps` mode consumes a deterministic continuous pack stream
    until `max_steps * effective_batch_size` global pack presentations are
    consumed
  - metrics record optimized planned-step loss, effective batch size, world
    size, denominator/normalizer details, and resolved accumulation context
  - non-finite loss or gradient handling is centralized in
    `SupervisedTrainer`/`TrainRuntime`
  - scalar finite status is checked before backward; unsafe scalar loss skips
    backward for the planned-step window and clears accumulated gradients
  - gradient finite/overflow status is checked after backward and before
    clipping or optimizer stepping
  - bad sample/step conditions are recorded as warnings or diagnostics on the
    current planned step and do not retime schedules
  - unsafe non-finite scalars or gradients do not produce corrupted optimizer
    updates and are not silently zeroed
  - unsafe non-finite state skips the optimizer update on all ranks, clears
    accumulated gradients, advances scheduler time on the planned-step policy,
    and records optimizer/scheduler status
  - scheduled eval/checkpoint events still fire for planned steps with warnings
    or update-skipped guard status, with status recorded in artifacts
  - forward runs under configured precision/autocast policy, defaulting to
    `bf16` when configured and supported
  - differentiable loss math still upcasts selected tensors through
    `LossContext` to fp32
  - no custom scaler is added for bf16 in V1
  - training uses a streaming iterator and does not require materializing all
    packs before training
  - future stage-owned caches may be added only after payload-specific approval
  - trainer does not consume standard padded PyTorch batches
  - `evaluate_forward()` uses the same packed forward/loss path under
    `torch.no_grad()`
- Failure modes:
  - trainer computes loss terms directly
  - `train.py` becomes the real orchestration brain
  - trainer assigns `planned_step_id` or `micro_step_id`
  - trainer inspects raw examples to decide order, grouping, or segment semantics
  - gradient accumulation scaling happens inside `LossRunner`
  - loss is backpropagated without planned-step normalizer metadata and
    explicit runtime backend-scaling policy
  - non-finite loss is replaced with zero silently
  - unsafe scalar loss reaches backward before the pre-backward finite gate
  - non-finite handling is implemented independently inside loss terms
  - scheduler clock becomes successful-update-relative after an update-skipped
    planned step
  - one distributed rank steps optimizer while another rank skips
  - all packs are required in memory before training starts
  - padded batch tensors become the supervised training unit
- Non-goals:
  - data loading/rendering/encoding/packing semantics
  - Qwen forward implementation
  - loss semantics
  - checkpoint metadata schema ownership
  - rollout/inference training
  - exact optimizer/RNG resume in V1
  - placeholder classes for rollout, cache, hidden-state losses, or inference
- Debug/receipt artifacts:
  - step metric events
  - optional micro-step diagnostics
  - non-finite runtime diagnostics
  - schedule/checkpoint/eval artifacts linked from the run manifest
- Tests or parity checks:
  - `SupervisedTrainer` calls `LossRunner.compute(...)` rather than loss terms
    directly
  - planned-step denominator normalizers are applied before backward for
    protected token-wise losses
  - backend scaling fixture proves no extra accumulation divisor is applied to
    planned-step-normalized protected losses
  - optimizer/scheduler step occurs after `resolved_grad_accum_steps`
  - scheduler does not step on micro-steps
  - injected pre-backward non-finite scalar produces no backward call and records
    all-rank skip, gradient clear, scheduler planned-step advancement, and
    unchanged milestone timing
  - gradient clipping happens after accumulation and before optimizer step
  - tail-fill receipts record requested pack presentations, actual pack
    presentations, and tail-fill count
  - bad-step diagnostics do not create a new step counter or retime schedules
  - planned warning/update-skipped steps still trigger scheduled eval/checkpoint
    calls and record status
  - bf16 forward plus fp32 selected-logit loss math
  - streaming iterator path does not pre-materialize all packs
  - `MicroStep` excludes `ModelOutputs`, `LossBundle`, raw examples, and raw
    config
  - `SupervisedTrainer` does not assign `MicroStep` ids or own grouping
  - step metrics include planned step, micro-step id, pack id, segment count,
    physical length, supervised atom count, loss terms, `acc_top1`, `acc_top5`,
    grad norm, LR groups, optimizer-update status, warning/non-finite status,
    accumulation context, and timing
- Open questions:
  - exact internal helper/type names
- Recommendation: implement after `src/runtime/`, then wire through
  `src/train.py`.

### `src/runtime/`

- Status: approved
- Location: `src/runtime/`
- Purpose: provide the shared systems seam for single-GPU, Accelerate, and
  DeepSpeed-backed training without owning CoordExp data, packing, Qwen forward,
  loss, or checkpoint schema semantics.
- Public interface:
  - config surface `runtime.backend: single | accelerate | deepspeed`
  - small config subtree `runtime.accelerate` for CoordExp-owned Accelerate
    settings on the common path
  - small config subtree `runtime.deepspeed`, including `config_path`, for
    DeepSpeed-backed launches
  - `TrainRuntime.prepare(...)`
  - `TrainRuntime.backward(loss)`
  - `TrainRuntime.clip_grad_norm(...)`
  - `TrainRuntime.optimizer_step(...)`
  - `TrainRuntime.scheduler_step(...)`
  - `TrainRuntime.gather_metrics(...)`
  - `TrainRuntime.is_main_process`
  - exact helper/type names may be adjusted during implementation approval
- Inputs:
  - model
  - optimizer
  - scheduler
  - runtime/distributed config
  - optional runtime-owned iterable or tensor handles
- Outputs:
  - prepared model/optimizer/scheduler/runtime handles
  - rank-safe metric/artifact helpers
  - optimizer-update status
  - runtime diagnostics
- Owned state:
  - `TrainRuntime`
  - Accelerate integration
  - DeepSpeed integration through the runtime/Accelerate config surface
  - rank/process guards
  - gradient clipping and optimizer-step helpers
  - scalar gathering helpers
  - compact `OptimizerStepStatus` or equivalent status object
- Invariants:
  - runtime is Accelerate-first
  - both Accelerate and DeepSpeed-backed execution must be supported
  - raw single-GPU, DDP-style distributed, and DeepSpeed execution use the same
    runtime interface
  - DeepSpeed execution is not considered supported by the first vertical smoke;
    it requires a later systems smoke through the same `TrainRuntime` seam
  - DeepSpeed status is reported with explicit labels:
    `schema_accepted`, `conflict_validation_implemented`,
    `systems_smoke_verified`, and `production_supported`; V1 artifacts may
    claim only proven labels
  - `runtime.backend: single` is a friendly single-process config value, but it
    may still route through the same minimal Accelerate-backed `TrainRuntime`
    seam internally so single-GPU and distributed paths do not drift
  - basic Accelerate-backed single-node launches do not require a separate
    Accelerate config file; the V1 path uses typed CoordExp-owned runtime fields
    and instantiates Accelerate from those fields
  - DeepSpeed configuration is passed by `runtime.deepspeed.config_path` to an
    official DeepSpeed JSON/YAML config; CoordExp validates only
    conflict-sensitive fields such as batch-size and gradient-accumulation
    settings rather than re-modeling the full DeepSpeed schema in Pydantic
  - DeepSpeed support does not create a separate trainer substrate
  - DeepSpeed uses the same repo entrypoint,
    `python -m src.train --config ...`; launch wrappers may vary, but the
    training entry does not fork into `train_deepspeed.py`
  - `prepare(...)` owns device/distributed wrapping for model, optimizer,
    scheduler, and runtime-owned iterable/tensor handles
  - runtime does not own raw data loading, rendering, Qwen encoding, packing
    policy, or loss construction
  - gradient clipping happens after accumulation is ready and before optimizer
    step
  - public cadence uses planned step ids materialized before training, not a
    separate `global_step` or successful-update counter
  - bad sample/step warnings do not advance, delay, or rebase eval/checkpoint
    milestones
  - non-finite loss/gradient and backend overflow checks reduce to one all-rank
    optimizer-step decision
  - scalar loss finite status is checked before backward; unsafe scalar loss
    skips backward for the whole planned-step window, clears accumulated
    gradients, and records diagnostics through the same planned-step status path
  - gradient finite and backend overflow status are checked after backward and
    before clipping or optimizer stepping
  - if any rank is unsafe, every rank skips the optimizer update, clears
    accumulated gradients, advances scheduler time on the planned-step policy,
    and records the same global planned-step status
  - `OptimizerStepStatus` records planned step id, per-rank finite flags,
    backend overflow status when available, global update decision,
    `optimizer_update_applied`, `scheduler_step_applied`, gradient-clear status,
    and LR values after the planned scheduler boundary
  - runtime exposes rank guards and scalar gathering
  - rank 0 writes metrics, receipts, manifests, and checkpoints unless a backend
    save helper requires coordinated participation
  - components do not invent independent rank-writing policy
  - Accelerate/DeepSpeed receive the CoordExp-derived
    `resolved_grad_accum_steps`; conflicting user-provided backend accumulation
    settings fail fast
  - runtime explicitly documents and tests backend loss-scaling behavior so
    Accelerate or DeepSpeed gradient accumulation does not apply an extra blind
    divisor to `LossBundle.total_loss`
  - `CheckpointWriter` owns checkpoint schema, adapter payloads,
    special-token embedding deltas, metadata, and aliases
  - `CheckpointWriter` lives under `src/artifacts/checkpoints.py`; runtime does
    not own that module
  - runtime provides rank-safe save helpers and unwrap/prepare utilities but
    does not own checkpoint semantics
  - V1 checkpoints save adapter weights, special-token embedding deltas,
    tokenizer/processor/model identity, resolved config fingerprint,
    loss/pack metadata, enabled cache identity when present, counters, and
    interpretation metadata
  - checkpoint loading is one shared Qwen composition path for base-only,
    base-plus-adapter, and base-plus-adapter-plus-special-token-embedding-delta
    modes across training weight initialization, offline eval, and inference
  - checkpoint metadata records planned step id, `resolved_max_steps`, schedule
    fingerprint, optimizer-update status, warning/non-finite status, linked eval
    outputs, and metric summary links
  - numbered checkpoint directories are keyed by planned step id with unpadded
    HF-like names such as `checkpoint-2`, `checkpoint-4`, and `checkpoint-5`
  - checkpoint directory step ids are not zero-padded
  - checkpoint directories contain `checkpoint_metadata.json`, optional
    `adapter/`, `special_token_embeddings.safetensors`, and
    `special_token_embeddings.json`
  - base model files are not copied into V1 checkpoints
  - adapter payloads live under `adapter/`; PEFT-compatible layouts are
    preserved when supported, and dLoRA may store explicit metadata there
  - special-token embedding tensor deltas use `safetensors`; JSON side metadata
    records token strings/ids, tied/untied mode, tensor keys, tensor
    shapes/dtypes, and base identity checks
  - `checkpoints/checkpoint-final.json` always exists and means final run state;
    it points to the final planned-step checkpoint directory rather than
    duplicating payloads
  - `checkpoints/best_acc_top1.json` may select warning-only checkpoints but excludes
    update-skipped/unsafe non-finite checkpoints unless explicitly configured
  - alias pointer files are JSON, not symlinks or copied checkpoint directories
  - pointer files record target checkpoint directory, planned step id, selection
    reason, status, and selection metric when applicable
  - checkpoint loading fails fast on base model identity, tokenizer/vocab
    identity, special-token strings/ids, coordinate-token mapping, tied/untied
    mode, adapter type/targets, and tensor shape/dtype mismatches
  - V1 checkpoints do not promise optimizer, scheduler, scaler,
    dataloader/iterator, or RNG resume
  - eval/checkpoint cadence is based on the precomputed planned step schedule
- Failure modes:
  - DeepSpeed path requires a different trainer or bypasses `TrainRuntime`
  - raw single-GPU path has behavior drift from distributed runtime path
  - runtime introduces an unrecorded successful-step/global-step clock
  - checkpoint directories are keyed by successful-update count or wall-clock
    save order
  - checkpoint directories use zero-padded planned step ids such as
    `checkpoint-000005`
  - train/eval/infer each invent separate model-plus-adapter loading paths
  - checkpoint aliases are copied payload directories
  - checkpoint aliases are symlinks
  - checkpoint saves copied base model weights
  - special-token embedding deltas are saved as JSON tensor arrays or full
    embedding/head matrices
  - checkpoint loading warns and tries to continue after identity mismatch
  - warning-only checkpoints are suppressed by default
  - `checkpoints/checkpoint-final.json` is missing because the final planned step had
    warnings
  - `checkpoints/best_acc_top1.json` silently selects an update-skipped checkpoint
  - gradient clipping happens per micro-step before accumulation is ready
  - authored `grad_accum_steps` reappears as a public config knob
  - runtime silently rounds or changes effective batch size to fit world size
  - backend gradient accumulation applies an extra hidden divisor to
    `LossBundle.total_loss`
  - DeepSpeed execution is claimed before loss scaling, non-finite consensus,
    and checkpoint save are proven through a systems smoke
  - non-finite/overflow status is decided independently on each rank
  - update-skipped planned step leaves stale accumulated gradients
  - scheduler state follows successful optimizer updates without recording that
    policy
  - DeepSpeed or Accelerate accumulation config conflicts with
    `resolved_grad_accum_steps` and is accepted silently
  - scheduler steps on every micro-step
  - final optimizer step is smaller than `effective_batch_size`
  - final incomplete packs are silently discarded
  - every rank writes metrics or checkpoints independently
  - runtime owns checkpoint artifact schema
  - checkpoint metadata implies exact training resume support in V1
  - separate `train_deepspeed.py` entrypoint appears
- Non-goals:
  - data/render/encode/pack semantics
  - Qwen forward semantics
  - loss semantics
  - checkpoint schema ownership
  - exact optimizer/RNG resume in V1
- Debug/receipt artifacts:
  - runtime/distributed setup receipt
  - planned-step status metrics
  - bad-step diagnostics
  - rank/process metadata in run manifest
- Tests or parity checks:
  - single-GPU path uses `TrainRuntime`
  - `runtime.backend: single` follows the same runtime seam as distributed paths
  - Accelerate path prepares model/optimizer/scheduler through runtime
  - DeepSpeed config path remains available through `runtime.backend` and
    `runtime.deepspeed`
  - basic Accelerate path works without a required external Accelerate config
    file
  - DeepSpeed `config_path` conflicts with CoordExp-derived accumulation or
    batch settings fail before training
  - DeepSpeed path still enters through `python -m src.train --config ...`
  - DeepSpeed systems smoke proves prepare/backward/accumulation/non-finite
    consensus/scheduler/checkpoint/rank-safe artifact behavior before any
    production support claim
  - gradient clipping occurs after accumulation and before optimizer step
  - eval/checkpoint milestones come from `resolved_step_schedule.json`
  - bad-step warnings do not retime planned milestones
  - warning-only and update-skipped planned steps still trigger scheduled
    checkpoint/eval calls with status metadata
  - injected one-rank NaN/Inf in a distributed smoke produces one all-rank skip
    decision, clears gradients, advances scheduler time by planned step, and
    writes synchronized status
  - checkpoint metadata contains planned step id, max steps, schedule
    fingerprint, optimizer-update status, warning/non-finite status, eval links,
    and metric-summary links
  - checkpoint directory names use unpadded planned step ids
  - base-only, base-plus-adapter, and base-plus-adapter-plus-special-token
    embedding-delta loading use the same Qwen composition path for train/eval/infer
  - checkpoint payload layout includes optional `adapter/`, safetensors
    special-token embedding deltas, and JSON metadata
  - `checkpoints/checkpoint-final.json` and `checkpoints/best_acc_top1.json`
    pointer files do not duplicate checkpoint payloads
  - strict identity validation rejects tokenizer/token/adapter/base mismatches
  - rank guard allows only main process artifact writes in ordinary cases
  - checkpoint writer receives unwrapped/prepared save handles without runtime
    owning schema semantics
- Open questions: none for backend config naming.
- Recommendation: implement before `src/training/` so the trainer uses the same
  seam from the first smoke.

## Reserved Next-Version Surfaces

### Hidden-State Losses

- Status: deferred
- Location: `src/qwen/forward.py`, `src/losses/`, `src/supervision/`
- Purpose: support future auxiliary losses over selected text-token hidden
  states, while reserving visual activation support behind a separate explicit
  hook.
- Public interface: future approved wrapper-owned hidden-state capture hook plus
  selected-hidden-state helpers in `LossContext`.
- Inputs: approved `text_hidden_states` requirement from enabled loss config.
- Outputs: selected fp32 text hidden-state tensors for differentiable loss math.
- Owned state: none in V1.
- Invariants:
  - hidden-state capture is activated only through aggregated forward
    requirements from enabled losses
  - no standalone `training.output_hidden_states` knob controls production
    training capture
  - V1 protected CE/gate losses require logits only
  - actual text hidden-state loss implementation may be post-first-smoke
  - first smoke proves unsupported hidden-state/visual requirements fail cleanly
    if requested before their hooks are implemented
  - text hidden states are the first activation surface because they align to
    packed token positions
  - first approved text activation capture is `lm_head_input`, the
    post-final-norm hidden state consumed by `lm_head`
  - avoid ambiguous selector names such as `final` and `last`
  - pre-final-norm decoder activation capture is future `decoder_layer_post`
    with explicit `layer: -1` or `layer: N` and requires a wrapper hook
  - regex is not a hidden-state activation selector language
  - visual activations/visual feature rows are deferred behind
    `vision_activations` and an explicitly named future capture point
  - image placeholder token positions are not visual-row positions
  - next implementation version must use a real Qwen smoke to prove hidden-state
    capture shape and layer semantics
  - hidden-state loss math upcasts selected tensors to fp32
- Failure modes:
  - assuming `output_hidden_states=True` is enough on installed Qwen3-VL
  - assuming `decoder_layer_post` exists before a wrapper hook is implemented
  - using regex, `final`, or `last` for hidden-state activation selectors
  - warning through invalid selectors, unsupported captures, out-of-range layer
    indices, or empty selected positions
  - treating language placeholder token positions as visual feature rows
  - requesting visual activations before a real visual hook exists
  - using vague "vision hidden states" terminology without a capture point
- Non-goals:
  - visual-activation caches
  - generic activation capture framework in V1
  - visual activation capture in the first smoke
- Debug/receipt artifacts:
  - future Qwen hook receipt
  - activation section in `loss_plan.json`
- Tests or parity checks:
  - future Qwen text hidden-state capture smoke
  - first-smoke unsupported visual activation requirement failure
  - selector validation rejects ambiguous, regex, unsupported, out-of-range, and
    empty-position selectors before training
- Open questions:
  - which concrete hidden-state loss and token/span selections should be
    implemented first after the logits path is verified
- Recommendation: keep interface names future-compatible, but do not implement
  until V1 logits path is verified.

### Persistent Pre-Forward Caches

- Status: deferred
- Location: future stage-owned modules first; introduce `src/cache/` only after
  two real cache stages need shared policy.
- Purpose: optionally reuse deterministic pre-model artifacts after the uncached
  V1 path is correct.
- Public interface: future approved stage-specific cache cards. Candidate stages
  are `RawIndexCache`, `RenderedExampleCache`, `EncodedExampleCache`, and
  `PackPlanCache`.
- Inputs: stage payload plus fingerprint axes approved by that stage.
- Outputs: stage-owned replay payload and optional cache receipt/manifest.
- Owned state: none in V1.
- Invariants:
  - uncached V1 training remains the canonical first path
  - V1 does not create an empty reserved `src/cache/` namespace
  - candidate cache names are not V1 implementation requirements
  - cache contents are disposable and never bundled into checkpoints
  - no shared cache modes or invalidation framework until a payload-specific card
    exists
  - random object ordering may use future caches only when seed and epoch policy
    make replay deterministic and fingerprintable
- Failure modes:
  - generic `src/cache/` package becomes a dumping ground before real payloads
    exist
  - rendered or encoded caches silently ignore template, tokenizer, processor,
    object-ordering, image, or pack-policy identity
- Non-goals:
  - V1 storage implementation
  - empty reserved cache package
  - mandatory `reports/cache.json`
  - hidden-state, logits, KV, rollout-output, or visual-activation cache
- Debug/receipt artifacts: none in V1; future cache-bearing runs register a cache
  receipt or manifest link.
- Tests or parity checks: future payload-specific replay and stale-entry tests.
- Open questions: which candidate cache earns implementation first after the V1
  smoke.
- Recommendation: keep as a deferred note only; implement no cache storage in
  the first milestone.

### Visual-Row And Coverage Capture

- Status: deferred
- Location: `src/qwen/forward.py`, `src/losses/`, `src/rollouts/`
- Purpose: reserve the right to support coverage-ledger, visual-row, and
  hidden-state losses without mislabeling same-forward capture as a cache.
- Public interface: future approved Qwen capture hook and loss input request
  surface.
- Inputs: Qwen forward outputs, same-forward hidden states or visual features,
  row/coverage supervision, and rollout provenance when applicable.
- Outputs: differentiable loss inputs or inference-style rollout artifacts.
- Owned state: none in V1.
- Invariants:
  - `output_hidden_states=True` alone is not assumed to satisfy Qwen3-VL capture
  - visual-feature reuse is not approved unless visual/aligner trainables,
    adapter targets, dtype/device, processor identity, `image_grid_thw`,
    placeholder layout, DeepStack features, and gradient policy are all gated
  - faithful row-conditioned rollout may intentionally re-prefill and set
    cross-row cache reuse to false
- Failure modes:
  - persisted activation cache hides gradient breakage
  - final image embeds are cached without Qwen3-VL DeepStack feature rows
  - KV cache semantics are confused with row/coverage supervision semantics
- Non-goals: V1 implementation or generic activation-cache framework.
- Debug/receipt artifacts: future Qwen capture/parity receipt and gradient gate.
- Tests or parity checks: future no-cache vs cached-feature parity probe plus
  frozen/trainable visual-gradient rejection probe.
- Open questions: first approved loss family that needs this hook.
- Recommendation: defer to the next-version hidden-state/coverage work after the
  supervised logits path is proven.

### Rollout Training Readiness

- Status: deferred
- Location: `src/rollouts/`, `src/supervision/`, `src/losses/`, `src/training/`
- Purpose: reserve the ability to add rollout-derived supervision without
  refactoring the supervised pipeline.
- Public interface: future rollout supervision producer reusing `TokenAtom`,
  `TokenSpan`, `PackedLayout`, `LossContext`, and `LossRunner`.
- Inputs: generated examples, rollout provenance, inference-style evaluation
  artifacts.
- Outputs: supervision-compatible token targets and loss inputs.
- Owned state: none in V1 beyond reserved architecture notes/package markers.
- Invariants:
  - supervised training should not know rollout training exists
  - rollout supervision attaches to shared supervision/loss inputs, not a
    trainer-only side channel
  - `eval.inference` remains separate from `eval.forward`
- Failure modes: empty placeholder modules that pretend rollout is implemented.
- Non-goals: rollout training code in V1.
- Debug/receipt artifacts: future rollout provenance receipts.
- Tests or parity checks: future rollout fixture and inference-format checks.
- Open questions: exact rollout example schema and loss families.
- Recommendation: reserve names and constraints only; implement after V1 smoke.
