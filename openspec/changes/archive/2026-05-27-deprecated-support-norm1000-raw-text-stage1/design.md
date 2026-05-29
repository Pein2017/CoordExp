## Context

CoordExp's public-data pipeline already emits three canonical surfaces per
prepared preset:

- pixel-space JSONL: `<split>.jsonl`
- norm1000 integer JSONL: `<split>.norm.jsonl`
- coord-token JSONL: `<split>.coord.jsonl`

The Stage-1 training stack, however, only treats the coord-token surface as a
valid model-facing expression. That global assumption leaks across:

- config parsing (`custom.coord_tokens.enabled must be true`)
- prompt resolution (`coord_mode=coord_tokens` only)
- cache identity (`custom_coord_mode=coord_tokens`)
- documentation and examples

This creates two problems:

1. it blocks the most direct benchmark for testing whether `<|coord_*|>`
   tokens themselves introduce optimization tax,
2. it forces researchers to think of `*.norm.jsonl` as an internal artifact
   even though it is already the correct offline-prepared surface for raw-text
   norm1000 coordinates.
3. it prevents a score-aware raw-text benchmark because the current confidence
   path is specialized around coord-token spans.

The design goal is therefore narrow:

- keep the canonical dataset family unchanged,
- keep bbox semantics unchanged (`xyxy`),
- keep the same Qwen3-VL norm1000 lattice (`0..999`),
- only widen the model-facing expression contract from one mode to two modes,
- and do so only for the pure-CE Stage-1 benchmark surface.

## Goals / Non-Goals

**Goals**
- Support Stage-1 training from canonical `*.norm.jsonl` using raw-text numeric
  coordinates on the norm1000 lattice.
- Make geometry expression an explicit contract dimension across prompt
  building, cache identity, and training startup.
- Preserve benchmark comparability by keeping:
  - canonical `xyxy`
  - canonical preset lineage
  - the same `[0,999]` lattice as coord-token mode
- Keep config changes minimal for operators.
- Recover real score-aware mAP for the raw-text benchmark instead of relying on
  constant-score fallback scoring.

**Non-Goals**
- Changing non-canonical bbox branches (`cxcy_logw_logh`, `cxcywh`).
- Redesigning Stage-2 around raw-text geometry expression.
- Broadening soft-coordinate or bbox-aux objective families to raw-text mode in
  this change.
- Introducing a second numeric lattice or pixel-space raw-text benchmark in
  this change.
- Replacing coord-token mode; the change is additive.

## Decisions

### 1. Use existing `coord_tokens.enabled` as the expression-mode switch

Decision:
- Do not add a new top-level config key for the first implementation slice.
- Use the existing authored signal:
  - `custom.coord_tokens.enabled=true` -> `coord_tokens`
  - `custom.coord_tokens.enabled=false` -> `norm1000_text`

Rationale:
- This keeps the user-facing config delta small.
- The dataset builder already branches on `coord_tokens_enabled`.
- The pure-CE benchmark does not need a third mode yet.

Trade-off:
- The name `coord_tokens` becomes slightly asymmetric when disabled, but the
  operator cost is low and the migration is simple.

### 2. Canonical `*.norm.jsonl` is the authoritative raw-text training surface

Decision:
- Raw-text norm1000 training uses canonical preset artifacts directly:
  - `train.norm.jsonl`
  - `val.norm.jsonl`
- No new `public_data` derivation algorithm is required for V1.

Rationale:
- Those files already carry the correct `xyxy` chart and the correct
  `[0,999]` lattice.
- Reusing them avoids redundant data pipelines and keeps the benchmark about
  expression, not data preparation.

### 3. Dense prompts become mode-aware

Decision:
- Prompt resolution must support two dense geometry-expression variants:
  - coord-token instructions
  - numeric norm1000 instructions
- The dataset-specific policy layer remains shared; only the geometry wording
  changes.
- The assistant output shell remains fixed:
  - top-level `{"objects": [...]}`
  - unchanged object keys and ordering semantics
  - only geometry arrays switch between token literals and numeric JSON values

Rationale:
- We want prompt parity across the benchmark except for the expression
  contract itself.
- This keeps `coco_80`, ordering, and object-field-order behavior stable.

### 4. Cache identity records geometry-expression mode explicitly

Decision:
- Encoded-sample and packing fingerprints must include the resolved geometry
  expression mode, not a hardcoded `coord_tokens` string.

Rationale:
- `train.coord.jsonl` and `train.norm.jsonl` may share the same canonical
  source lineage but are not interchangeable model-facing training surfaces.
- Silent cache reuse across them would invalidate the benchmark.

### 5. Raw-text benchmark inference must be explicit, not auto-resolved

Decision:
- The documented benchmark eval path for norm1000 raw-text mode uses:
  - `infer.mode=text`
  - `infer.pred_coord_mode=norm1000`
  - `infer.bbox_format=xyxy`

Rationale:
- Auto mode is convenient, but it weakens the benchmark contract because
  raw-text `[0,999]` numbers can be ambiguously interpreted by heuristics.
- Explicit mode selection keeps the eval boundary clean and reproducible.
- The implementation may still technically allow explicit/fixed settings rather
  than inventing a separate benchmark-only flag surface.

### 6. Confidence for raw-text benchmark uses numeric bbox spans

Decision:
- Add numeric-span confidence scoring for canonical raw-text `xyxy` runs.
- Use bbox numeric spans recovered from `raw_output_json` and aligned against
  token traces to derive per-object scores.
- Keep the first slice geometry-focused so the benchmark mAP reflects bbox-span
  confidence rather than a new fused desc+geometry scoring family.

Rationale:
- The benchmark question is about geometry expression.
- Raw-text `xyxy` remains canonical, so it should be eligible for true
  score-aware mAP rather than constant-score compatibility scoring.

## Architecture

### A. Expression-mode contract

Introduce a repo-level conceptual split:

- `coord_tokens`
  - source artifact: `*.coord.jsonl`
  - assistant payload: bare `<|coord_k|>` geometry literals
  - prompt geometry wording: coord-token-specific
- `norm1000_text`
  - source artifact: `*.norm.jsonl`
  - assistant payload: standard JSON numeric geometry arrays
  - prompt geometry wording: numeric `[0,999]` specific

Both modes:
- use canonical `bbox_format=xyxy`
- use the same norm1000 lattice
- preserve the same ordering and prompt-variant machinery

### B. Config behavior

The Stage-1 config contract becomes:

- `custom.emit_norm: none` remains required
- `custom.bbox_format: xyxy` remains valid
- `custom.coord_tokens.enabled=true` means coord-token mode
- `custom.coord_tokens.enabled=false` means norm1000 raw-text mode
- `custom.coord_tokens.skip_bbox_norm` is required only when coord-token mode
  is enabled; in raw-text mode it is ignored or tolerated

### C. Dataset/build pipeline

No new data conversion stage is needed.

Runtime behavior becomes:
- if the dataset surface is `*.coord.jsonl`, builder emits coord-token payloads
- if the dataset surface is `*.norm.jsonl`, builder emits numeric JSON payloads
- startup validation ensures dataset suffix and expression-mode intent agree
  strongly enough to prevent accidental mixed runs

### D. Prompt resolution

Prompt resolver inputs expand conceptually from:
- prompt variant
- ordering
- object field order
- coord mode

to a real two-branch expression contract:
- `coord_mode=coord_tokens`
- `coord_mode=norm1000_text`

The fixed base prompt keeps the same ordering and dataset policy structure, but
switches geometry instructions and examples accordingly.

### E. Inference / eval benchmark path

For raw-text benchmark runs:
- standalone inference remains in `text` mode
- predictions are interpreted as `norm1000`
- standardized artifacts still emit canonical pixel-space boxes after
  denormalizing with each sample's image `width/height`
- evaluation and visualization consume only those canonical pixel-space boxes

One additive scoring change is required:
- raw-text `xyxy` benchmark runs use numeric-span confidence scoring derived
  from token traces and numeric bbox spans
- unlike the non-canonical bbox branches, they should not need constant-score
  scoring just to recover canonical geometry

## Risks / Trade-offs

- [Prompt drift between coord-token and raw-text modes]
  -> keep one resolver with explicit mode input and shared variant suffixes.
- [Cache contamination across modes]
  -> fingerprint geometry-expression mode explicitly.
- [Accidental misuse of `.jsonl` pixel-space artifacts for raw-text benchmark]
  -> document and validate that the intended raw-text training surface is
     `*.norm.jsonl`, not pixel-space `*.jsonl`.
- [Residual coord-token assumptions hidden in training code]
  -> fail fast on benchmark startup with explicit mode-aware validation rather
     than allowing silent fallback.
- [Operator confusion over `coord_tokens.enabled=false`]
  -> document it as the minimal Stage-1 raw-text switch for V1.
- [Numeric span alignment can fail on messy text traces]
  -> use stable failure reasons and auditable per-object scoring outputs rather
     than silently fabricating scores.

## Migration Plan

1. Add the geometry-expression capability and update specs to distinguish
   `coord_tokens` from `norm1000_text`.
2. Relax coord-token-only config validation and make prompt building
   mode-aware.
3. Update cache fingerprints and runtime validation to include the resolved
   geometry-expression mode.
4. Add a canonical pure-CE Stage-1 raw-text config that points to
   `train.norm.jsonl / val.norm.jsonl`.
5. Add numeric-span confidence scoring for raw-text canonical `xyxy` runs.
6. Add a smoke workflow proving:
   - training startup succeeds on norm1000 raw text
   - infer/eval succeeds with explicit text/norm1000 settings
