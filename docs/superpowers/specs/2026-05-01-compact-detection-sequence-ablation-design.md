# Compact Pixel2Seq-Style Detection Sequence Training Infrastructure Design

## Problem

The current Stage-1 detection baseline emits CoordJSON object dictionaries:

```text
{"objects": [{"desc": "...", "bbox_2d": [<|coord_x1|>, ...]}, ...]}
```

This is compatible with current training, inference, parsing, confidence, and
evaluation contracts, but it spends many supervised tokens on JSON keys,
quotes, brackets, and repeated structure. The next ablation asks whether a
more Pixel2Seq-like compact object-row sequence improves Stage-1 SFT and later
eval-time rollout behavior without entering Stage-2 rollout/objective work.
This design is the Phase 1 training-infrastructure slice only. Linear owns the
larger research lifecycle; this repo-local super-power spec owns the code
implementation and smoke verification needed before merging training
infrastructure to `main`.

## Phase 1 Scope

- Stage-1 SFT only.
- Training-data serialization, prompt wiring, token-role handling, cache
  fingerprinting, and Stage-1 smoke verification only.
- COCO 1024 bbox-max60 coord-token data only:
  - `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- Qwen3-VL 2B only.
- Source checkpoint:
  - `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- The implementation must not expand, rewrite, or mutate the source checkpoint
  or tokenizer. Compact rows reuse existing Qwen grounding sentinel tokens.

## Non-Goals

- No Stage-2 rollout-aware training, replay, reward shaping, or objective
  changes in Phase 1.
- No inference pipeline, confidence post-op, detection evaluation, `val200`, or
  benchmark interpretation as a Phase 1 merge gate. These belong to Phase 2
  after production checkpoints exist.
- No new stable CLI flags when a YAML config field can express the run.
- No changes to upstream HF model files such as `modeling_qwen3_vl.py`.
- No runtime bbox re-interpretation; raw JSONL remains canonical `xyxy`
  coord-token input and compactness is only the model-facing assistant target.
- No 4B run in the first comparison.

## Variants

`coordjson` / E baseline keeps the current existing detection template.

Compact variants A-D use newline-delimited object rows. The exact row separator
is `\n`; it is part of the serialization contract and must participate in
renderer tests, parser tests, prompt hashes, encoded-sample-cache fingerprints,
and static-packing fingerprints. A final trailing newline is optional at parse
time but the renderer should omit it for deterministic token counts.

For phase 1, compact descriptions are constrained to the existing sanitized
COCO-80 description surface: no embedded newline, no tab/control character, and
no literal marker-like substring matching `<|object_ref_start|>`,
`<|box_start|>`, `<|coord_`, `<|im_start|>`, or `<|im_end|>`. Any generated row
that violates these grammar rules is dropped with an explicit parse diagnostic
instead of being converted into an invented box.

`compact_full` / A emits one object row:

```text
<|object_ref_start|>{desc}<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>
```

`compact_no_desc` / B removes only the description-start sentinel
`<|object_ref_start|>`; the raw description text remains:

```text
{desc}<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>
```

`compact_no_bbox` / C removes only the bbox-start sentinel `<|box_start|>`; the
description marker, raw description text, and four bbox coordinate tokens remain:

```text
<|object_ref_start|>{desc}<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>
```

`compact_min` / D removes both structural sentinels while preserving raw
description text and four bbox coordinate tokens:

```text
{desc}<|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|>
```

## Architecture

The compact sequence layer should be a narrow serialization adapter around the
existing Stage-1 data and training contracts:

```text
strict JSONL objects
  -> model-facing sequence renderer
  -> Qwen3-VL chat template
  -> Stage-1 SFT
  -> training artifacts and production checkpoints
```

The stable boundary remains canonical objects with `desc` and `bbox_2d`.
Compact variants may omit fields inside the assistant target, but the raw JSONL
and geometry contract do not change.

The shared implementation owner is `src/common/detection_sequence.py`. Parser
helpers may live there for unit-test symmetry and Phase 2 reuse, but Phase 1
only uses the renderer in the training path.

Use one Phase 1 training config surface:

- training: `custom.detection_sequence_format`
- allowed values: `coordjson`, `compact_full`, `compact_no_desc`,
  `compact_no_bbox`, `compact_min`

Do not mix this flat contract with a second nested
`custom.detection_sequence.*` schema during the pilot.

Phase 2 may add or refine `infer.detection_sequence_format` once production
checkpoints exist and inference/eval is the active Linear phase.

Token handling is a separate role contract, not part of the renderer/parser
contract. The compact renderer emits native Qwen marker strings; the training
stack decides which token IDs receive trainable row offsets and which token IDs
participate in coordinate/regression losses.

## Tokenizer/Checkpoint Contract

The real 2B coord-expanded checkpoint has no free embedding rows beyond the
tokenizer length. Gate 0 must therefore treat compact markers as existing
Qwen-native grounding sentinels, not as newly added tokens, tokenizer metadata
replacements, or consumed unassigned rows.

Verified source invariants for the current 2B checkpoint are:

```text
len(tokenizer) == 152670
tokenizer.vocab_size == 151643
config.text_config.vocab_size == 152670
model.language_model.embed_tokens.weight.shape == (152670, 2048)
<|coord_*|> == 151669
<|coord_0|> == 151670
<|coord_999|> == 152669
<|object_ref_start|> == 151646
<|box_start|> == 151648
```

For added-token checks, use `len(tokenizer)` as the total tokenizer length.
`tokenizer.vocab_size` is only the base BPE vocabulary size and does not include
Qwen special tokens or CoordExp coordinate tokens.

The compact sequence format must use the literal native token strings
`<|object_ref_start|>` and `<|box_start|>`. The implementation must not call
`tokenizer.add_special_tokens()` for compact markers, must not resize model
embeddings, and must not rewrite tokenizer metadata. Phase 1 uses the existing
coord-expanded source checkpoint directly.

The coordinate-offset range remains coordinate-only:

```yaml
custom:
  coord_offset:
    ids: { start: 151670, end: 152669 }  # <|coord_0|>.. <|coord_999|>
```

Do not widen this range to include compact marker IDs. The compact marker rows
are sparse IDs and must be resolved separately:

```text
151646: <|object_ref_start|>
151648: <|box_start|>
```

The implementation should refactor trainable embedding/logit-row offsets toward
a token-role resolver with named groups:

```yaml
custom:
  trainable_token_rows:
    enabled: true
    tie_head: true
    groups:
      coord_geometry:
        role: coord_geometry
        start_token: "<|coord_0|>"
        end_token: "<|coord_999|>"
        expected_start: 151670
        expected_end: 152669
      compact_structure:
        role: structural_ce_only
        tokens: ["<|object_ref_start|>", "<|box_start|>"]
        expected_ids:
          "<|object_ref_start|>": 151646
          "<|box_start|>": 151648
```

`custom.coord_offset` remains a backward-compatible coordinate-only alias during
the pilot. Compact marker embeddings must be made trainable like coord-token
embeddings, but through the sparse `compact_structure` group, not by merging
them into the coord span.

Token code lives under a role-aware package hierarchy:

```text
src/tokens/
  qwen_native.py              # native Qwen/CoordExp special-token constants
  roles.py                    # TokenRoleSets and trainable/loss ID separation
  row_offsets.py              # generic trainable embedding/logit row adapter
  coord/
    codec.py                  # <|coord_i|> syntax and coordinate ID lookup
    validator.py              # coord-token JSONL annotation
    template_adapter.py       # Qwen bbox-normalization bypass for coord data
    soft_ce_w1.py             # coordinate-bin loss math
    masks.py                  # coordinate/regression-family masks
  structural/
    compact_markers.py        # compact structural marker constants
```

`src/coord_tokens/` remains as a compatibility wrapper during the pilot. New
code should import from `src.tokens.*`; existing imports continue to work until
a later cleanup pass migrates them.

## Token Role And Loss Boundary

The token roles are deliberately separated:

| Role | Token examples | Trainable row offset | Base assistant CE | Coordinate softCE/W1/regression-family losses |
| --- | --- | --- | --- | --- |
| `coord_geometry` | `<|coord_0|>`..`<|coord_999|>` | yes | yes, according to the active Stage-1 objective | yes |
| `structural_ce_only` | `<|object_ref_start|>`, `<|box_start|>` | yes | yes | no |
| description text | `cat`, `traffic light` | normal model training path | yes | no |

This boundary is a hard invariant. The native compact marker IDs `151646` and
`151648` may be included in the trainable embedding/logit-row adapter, but they
must not be returned by coordinate-token resolvers, coordinate masks,
coord-softCE/W1 losses, bbox regression losses, coordinate diagnostics, or
coordinate validity counters. The coordinate/regression family keeps using only
`<|coord_0|>`..`<|coord_999|>`.

## Phase 1 Artifact Contract

Training infrastructure must emit the normal Stage-1 reproducibility artifacts:

- `config_source.yaml`
- `resolved_config.json`
- `effective_runtime.json`
- `runtime_env.json`
- `train_data_provenance.json`
- `eval_data_provenance.json`
- `run_metadata.json`
- `experiment_manifest.json`
- `logging.jsonl`

The resolved config and runtime artifacts must show:

- `custom.detection_sequence_format`,
- `custom.trainable_token_rows`,
- `training.packing` / `training.packing_mode`,
- static-packing cache settings,
- raw JSONL paths and sample limits.

Inference parse diagnostics are intentionally deferred to Phase 2. Parser
helpers may exist in this branch if they are off by default and low risk, but
Phase 1 does not claim inference/eval readiness or any measured mAP/AP result.

## Phase 1 Merge Gates

0. Contract/render/tokenizer audit:
   - verify the source checkpoint path from the worktree,
   - prove native compact marker token IDs and coordinate token IDs resolve as
     expected,
   - prove tokenizer length, model text vocab size, and embedding row count
     already match without expansion,
   - render compact training assistant targets for at least one train and one
     val sample, with unit coverage for variant grammar,
   - verify cache-fingerprint separation,
   - verify `trainable_token_rows` resolves 1002 trainable row IDs for the
     compact-full pilot: 1000 `coord_geometry` IDs plus two sparse
     `structural_ce_only` IDs,
   - verify coordinate/regression loss masks still contain exactly the 1000
     coordinate IDs and exclude `151646` / `151648`,
   - stop if compact marker rows cannot be made trainable without widening the
     coordinate range or leaking into coordinate losses.
1. Training smoke:
   - use raw COCO JSONLs with deterministic `train_sample_limit` /
     `val_sample_limit` instead of copied temp JSONLs,
   - verify tokenizer/template rendering for the compact-full path,
   - run a short two-GPU Stage-1 train smoke for `compact_full`,
   - confirm training artifacts are complete,
   - confirm static packing builds and uses a format/sample-limit-separated
     cache fingerprint,
   - confirm the smoke does not mutate the source checkpoint.
2. Merge readiness:
   - targeted tests pass,
   - ruff passes on touched Python surfaces,
   - `git diff --check` passes,
   - branch is reconciled with current `main`,
   - super-power plan records Phase 1 evidence and defers Phase 2.

## Phase 2 Handoff

Linear tracks Phase 2 after production checkpoints exist. Phase 2 should own:

- production checkpoint selection,
- inference/eval config creation,
- compact parse diagnostics in inference artifacts,
- confidence post-op scoring policy,
- `val200` benchmark execution,
- final research interpretation in Notion and checked-in `progress/benchmarks/`.

## Workflow Pilot

For this research-management pilot:

- Notion has a global `CoordExp` root, a canonical `Research Units` database,
  and `Experiments & Evidence` as the renamed/optimized migration surface for
  `progress/`.
- The `Compact Detection Sequence Ablation` Research Unit is the canonical
  research-memory record. The standalone `Compact Detection Sequence Pilot`
  page is a supporting brief.
- Linear owns cross-phase execution state: Phase 1 training-infra merge,
  production training launch, Phase 2 inference/eval, `val200`, and final memo.
- Repo-local super-power specs/plans hold branch-specific implementation and
  verification details. They should not try to manage the whole research
  lifecycle, and Linear should not mirror file-level checklists.
- OpenSpec remains available for stable behavior or config-contract changes,
  but this pilot should not abruptly remove OpenSpec from the workflow.
- Final measured results still belong in `progress/benchmarks/`; Notion links
  them, but does not replace repo evidence.
