## ADDED Requirements

### Requirement: Rollout-matching trainer is YAML-gated
The system SHALL provide an opt-in rollout-matching training mode (alias: `stage_2`) that is enabled via YAML by setting:
- `custom.trainer_variant: rollout_matching_sft`.

When enabled, rollout-matching training SHALL be driven by YAML configuration and SHALL NOT require adding new hyperparameter CLI flags.

#### Scenario: Rollout-matching enabled via trainer_variant
- **GIVEN** a training config sets `custom.trainer_variant: rollout_matching_sft`
- **WHEN** `python -m src.sft --config <yaml>` is executed
- **THEN** training uses the rollout-matching trainer implementation
- **AND** the baseline training behaviour (alias: `stage_1`) is not used for that run.

### Requirement: Single-path training constructs one canonical teacher-forced target sequence
When rollout-matching training is enabled, the trainer SHALL implement a **single** training path that is expressed as:
- one canonical assistant token sequence per sample (`Y_train`), and
- one forward pass on that sequence, with per-token supervision masks.

There SHALL NOT exist separate training “paths” (e.g., “reordered-GT SFT” vs “self-context”); all supervision SHALL be expressed as per-token loss masks on a single teacher-forced forward pass.

The trainer SHALL construct the canonical assistant target sequence as:
- `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

where:
- `Y_rollout_prefix` is a **prefix** of the model’s rollout assistant `response_token_ids` produced by autoregressive generation.
  - The trainer MAY perform **suffix-only trimming** to define `Y_rollout_prefix` so it is safe for append:
    - treat `<|im_end|>` as a hard stop and strip it (and any suffix after it) when present,
    - drop any trailing incomplete / invalid suffix tokens beyond the last complete predicted object boundary (e.g., when rollout is truncated mid-object),
    - drop the final top-level JSON closing brace `}` so the prefix ends in an **open** JSON object ready for append.
    - NOTE: Some tokenizers fuse closing punctuation (e.g., a single token may decode to `}}` or `}},`).
      - In those cases, the desired suffix cut can fall **inside** the final token.
      - The trainer MAY realize the cut by replacing ONLY the final token with a shorter tokenization that decodes to a strict prefix of that token’s decoded text (e.g., `}}` → `}`), while keeping all earlier token IDs unchanged.
  - The trainer SHALL NOT edit or re-tokenize any token **before** the cut boundary used for the prefix (no decode+re-encode; no pretty-printing; no key sorting).
  - Failure behavior: if the rollout does not contain an opening `{` OR the prefix cannot be made append-ready via suffix-only trimming, the trainer SHALL treat the rollout prefix as empty and use `Y_rollout_prefix = "{"` (no prefix supervision; all GT objects become FN and are appended).
- `FN_gt_objects` are the GT objects that are unmatched after matching (see matching requirement).
- `SerializeAppend(FN_gt_objects)` emits GT objects in the project’s JSON-only assistant schema (object-index JSON mapping `object_{n}` → `{desc, geometry}`) as an **append fragment**:
  - it SHALL emit **only** comma-separated `"object_{n}": {...}` entries (no outer `{}` wrapper),
  - it SHALL decide whether to emit a leading comma based on the last non-whitespace character of the decoded `Y_rollout_prefix`:
    - if the last non-whitespace character is `{` or `,`, it SHALL NOT emit a leading comma,
    - if the last non-whitespace character is `}`, it SHALL emit a leading `, `,
    - otherwise it SHALL error (the prefix is not append-ready).
  - it SHALL assign keys `object_{n}` starting from `n = max_object_index_in_prefix + 1` (or `n = 1` when no valid object index exists in the prefix),
    - `max_object_index_in_prefix` is the maximum integer `n` observed in any key matching the pattern `object_{n}` in the rollout prefix (best-effort; invalid/malformed keys are ignored),
  - it SHALL terminate by emitting the single top-level JSON closing brace `}` so `Y_train` is a valid JSON object before `EOS`.

There SHALL be exactly ONE forward pass per sample on the canonical encoding (same chat template as generation).

#### Scenario: Training uses one sequence and one forward pass
- **GIVEN** a batch under rollout-matching training
- **WHEN** the trainer executes one training step
- **THEN** it performs exactly one forward pass per sample on `Y_train`
- **AND** it computes exactly one total loss from that forward pass by applying per-token supervision masks.

### Requirement: Rollout generation returns token IDs (no grad) and selects one response
When rollout-matching training is enabled, the trainer SHALL perform an autoregressive rollout (generation) for each training sample:
- using the current model parameters,
- with gradients disabled during rollout, and
- producing both a decoded assistant response string and the corresponding assistant `response_token_ids` sequence (which defines `Y_rollout`).

The rollout decoding mode SHALL be configurable (at minimum: greedy and beam).

If decoding uses beam search, the trainer SHALL select exactly one rollout response for training: the best beam (highest logprob). Other beams MAY be logged for debugging, but SHALL NOT affect training.

#### Scenario: Beam search selects exactly one rollout response
- **GIVEN** rollout-matching training is enabled
- **AND** rollout decoding is configured as beam search
- **WHEN** a sample produces multiple beams
- **THEN** the trainer uses only the single best beam as `Y_rollout`
- **AND** all subsequent parsing, matching, and loss masking are computed from that selected beam only.

### Requirement: Mandatory FN append (recall recovery)
Unmatched GT objects (false negatives, `FN_gt_objects`) SHALL ALWAYS be appended to the end of `Y_rollout_prefix` to form `Y_train`.

Rationale (normative): GT annotations may be incomplete, but they do not hallucinate. Recall MUST be recovered via FN append, rather than suppressing unmatched GT.

#### Scenario: All GT is appended when no valid matches exist
- **GIVEN** a sample where rollout parsing yields zero usable predicted objects (or all pairs are gated out)
- **WHEN** `Y_train` is constructed
- **THEN** `FN_gt_objects` equals the full GT object set for that sample
- **AND** `SerializeAppend(FN_gt_objects)` is appended to `Y_rollout_prefix` before EOS.

### Requirement: Predicted order is defined by raw rollout text appearance (no silent reordering)
“Predicted order” SHALL be defined as the appearance order in the raw rollout string (the assistant response text decoded from `response_token_ids`).

Parsers/utilities used for rollout-matching training SHALL NOT sort JSON keys or re-serialize/pretty-print the rollout prefix in any way that changes tokenization.

If predicted objects are represented as a dict-like JSON object, the object order SHALL be defined by the order of appearance of each object’s key/value span in the raw rollout text (NOT lexicographic key order).

#### Scenario: Dict keys are not sorted for matching order
- **GIVEN** a rollout response string whose object keys appear as `"object_10": {...}, "object_2": {...}`
- **WHEN** predicted objects are enumerated for matching
- **THEN** the predicted order is `object_10` followed by `object_2` (appearance order)
- **AND** the trainer does not reorder them lexicographically.

### Requirement: Strict parsing drops invalid predicted objects (no repair)
The trainer SHALL require strict, schema-conformant predicted objects.

If a predicted object is malformed or violates schema (e.g., missing brackets, extra commas, nested/unexpected keys, wrong coord count, non-coord tokens in coord arrays, invalid geometry key, missing/empty `desc`), that object SHALL be marked invalid and DROPPED:
- invalid objects SHALL NOT participate in matching, and
- invalid objects SHALL NOT contribute to self-context supervision.

The trainer SHALL NOT perform token-inserting “JSON repair” for rollout-matching training (no adding braces/quotes, no filling missing tokens, no re-serializing JSON).

However, the trainer MAY perform **suffix-only trimming** of the rollout tokens to:
- drop trailing incomplete text when rollout is truncated, and/or
- strip a terminal `<|im_end|>` token, and/or
- drop the final top-level `}` to enable FN append under the canonical `Y_train` construction.

All tokens **before** the suffix-trim boundary SHALL remain unchanged; “dropping” affects only (a) the parsed object list and loss masks and (b) the chosen prefix cut for `Y_rollout_prefix`.

#### Scenario: One malformed object does not block training
- **GIVEN** a rollout response containing 3 object entries in appearance order
- **AND** the middle object entry is malformed (e.g., wrong coord count)
- **WHEN** strict parsing runs
- **THEN** exactly that object is marked invalid and excluded
- **AND** the other valid object entries remain eligible for matching and supervision
- **AND** training still proceeds because FN append provides teacher-forced supervision for unmatched GT.

#### Scenario: Truncated rollout tail is trimmed (suffix-only) and training proceeds
- **GIVEN** a rollout response that is truncated mid-object (no balanced JSON object is available)
- **WHEN** the trainer constructs `Y_rollout_prefix` via suffix-only trimming
- **THEN** it drops the incomplete trailing suffix and keeps only the prefix up to the last complete object boundary (or just `{` when none)
- **AND** it appends `SerializeAppend(FN_gt_objects)` and `EOS` to form a valid `Y_train`
- **AND** the training step completes without crashing.

### Requirement: Coord-slot token indices are derived from token-aligned parsing (object-level validity)
Self-context supervision SHALL NOT rely on searching for repeated `<|coord_k|>` patterns in the text.

Coord-slot token indices SHALL be obtained deterministically from parsing the rollout token sequence (or a token-aligned parse), producing for each VALID predicted object:
- `bbox_2d`: exactly 4 coord-token indices (for `[x1, y1, x2, y2]`), and
- `poly`: exactly `2N` coord-token indices with `N >= 3` and even length.

If coord-slot indices for an object are not uniquely determined / not trusted, that object SHALL be excluded from self-context supervision (object-level exclusion). The sample SHALL still train via mandatory FN append in the tail.

#### Token-aligned parsing / prefix trimming algorithm (normative sketch)
The trainer SHOULD implement a single streaming pass over the rollout assistant token IDs to determine:
- (a) a safe `Y_rollout_prefix` cut boundary (append-ready), and
- (b) per-object geometry kind + coord-token indices in **appearance order**.

Normative algorithm sketch (no string-search for coord patterns; structure-aware only):
1. **Precompute coord-token IDs**:
   - Build a `coord_id_set` from `get_coord_token_ids(tokenizer)` (size 1000).
2. **Materialize per-token decoded text pieces**:
   - For each assistant token id `t_i` in `response_token_ids`, compute `piece_i = tokenizer.decode([t_i], skip_special_tokens=False, clean_up_tokenization_spaces=False)`.
   - The trainer MUST NOT decode+re-encode the entire sequence to “normalize” spacing or quotes.
3. **Run a streaming JSON structure scanner over `piece_i` in order**:
   - Track JSON state variables across characters:
     - `in_string` + `escape` (to ignore braces/brackets inside JSON strings),
     - `brace_depth` for `{}` and `bracket_depth` for `[]`.
   - Track parse context:
     - current top-level key string (e.g., `"object_17"`),
     - current object index `n` (when inside an `"object_n": {...}` value),
     - current geometry key (`bbox_2d` or `poly`) and whether the scanner is currently inside its array value.
4. **Determine appearance-order objects and coord-token indices**:
   - When a JSON string is parsed at `brace_depth == 1` in a “expecting key” position, and the key matches `object_{n}` where `n` is an integer, start a new predicted object context in appearance order.
   - Within that object’s value dict (`brace_depth == 2`), the scanner MUST locate exactly one geometry key (`bbox_2d` or `poly`) and then enter “capture mode” on the next `[` that begins that geometry’s array.
   - While in capture mode for a geometry array, every assistant token position `i` whose token id is in `coord_id_set` SHALL be appended to that object’s `coord_token_indices` list (even if the coord token is surrounded by JSON quotes).
   - Capture mode ends when the corresponding array `]` is closed (i.e., `bracket_depth` returns to the value it had immediately before the geometry array opened).
   - After capture ends:
     - `bbox_2d` MUST have exactly 4 coord-token indices, else the object is invalid.
     - `poly` MUST have an even number of indices and at least 6 total indices, else the object is invalid.
   - If an object contains multiple geometry keys, nested/unexpected geometry keys, or the geometry array is not fully closed before the chosen prefix cut boundary, the object is invalid and MUST be dropped (no repair).
5. **Determine the append-ready `Y_rollout_prefix` cut boundary**:
   - During the same streaming scan, the trainer MUST record candidate cut positions corresponding to the **end of the last complete predicted object entry** in the top-level JSON object.
     - A candidate cut occurs after a `}` that reduces `brace_depth` from 2 → 1 (end of an `"object_n": {...}` value dict).
     - A candidate cut MAY include a following comma token if it is fused (e.g., token decodes to `},`); `SerializeAppend` MUST handle prefixes whose last non-whitespace char is `{`, `,`, or `}` as specified above.
   - The selected cut boundary SHALL be the last recorded candidate at or before the end of the rollout, after stripping any trailing end-of-turn tokens like `<|im_end|>`.
   - **Fused-suffix handling**: if the last candidate cut falls inside the final token (e.g., a token decodes to `}}` and the cut is after the first `}`), the trainer MAY replace ONLY that final token with a shorter tokenization that decodes to the needed substring (e.g., `}}` → `}`), keeping all earlier token IDs unchanged.

#### Scenario: Ambiguous coord-slot alignment excludes object from self-context supervision
- **GIVEN** a predicted object whose geometry can be parsed but whose coord token indices cannot be uniquely aligned to `response_token_ids`
- **WHEN** the trainer builds self-context supervision masks
- **THEN** that object contributes no self-context coord loss
- **AND** its GT counterpart (if any) is treated as unmatched and included in `FN_gt_objects` for tail append.

### Requirement: Matching baseline uses Hungarian assignment with dummy augmentation and maskIoU gating
Matching SHALL be done via Hungarian assignment with dummy augmentation to allow FP/FN.

MVP baseline (configurable via YAML, with defaults defined by the trainer):
- Candidate reduction: for each predicted object, the trainer SHALL compute AABB IoU against GT AABBs and select top-k candidates before expensive geometry. If AABB IoU is all zero or candidates are insufficient, a deterministic fallback SHALL be used (e.g., keep top-k by center distance).
- Geometry cost: the trainer SHALL compute `maskIoU` between predicted and GT shapes (bbox/poly rasterized to masks) and define:
  - `cost_geo(i, j) = 1 - maskIoU(i, j)`
  - `maskIoU` SHALL be computed in **norm1000 space** on a fixed virtual canvas of size `R x R` (default `R=256`), by:
    - treating `poly` as a single-ring polygon,
    - treating `bbox_2d` as its quadrilateral polygon,
    - clamping coordinates to `[0, 999]` before projection to the `R x R` canvas.
- Gating (pre-assignment): pairs with `maskIoU < threshold` SHALL be treated as infeasible (equivalently `cost = +INF`) BEFORE assignment, to avoid wrong matches.
- Dummy semantics:
  - `pred -> dummy` represents FP (low penalty / light control only),
  - `dummy -> gt` represents FN and SHALL be handled by mandatory FN append (the GT object is appended, not silently dropped).

The matching output SHALL determine:
- matched pairs eligible for self-context supervision (subject to coord-slot alignment validity), and
- `FN_gt_objects` (all GT objects not matched to a usable predicted object).

#### Scenario: Pre-assignment gating prevents wrong matches and triggers FN append
- **GIVEN** a predicted shape that has `maskIoU < threshold` with every GT shape
- **WHEN** Hungarian matching runs
- **THEN** all `pred -> gt` edges are infeasible (`+INF`) prior to assignment
- **AND** the predicted object is assigned to dummy (FP)
- **AND** all GT objects remain unmatched and are appended via `FN_gt_objects`.

### Requirement: Poly self-context targets use Sinkhorn OT with barycentric projection only
For matched pairs where poly is involved (poly<->poly, bbox_2d<->poly, poly<->bbox_2d), the trainer SHALL construct self-context coord supervision targets using OT + barycentric projection (ONLY barycentric; no mixture):

- Represent both shapes as point sets:
  - `poly`: its vertex points from parsed coord tokens (point count `N >= 3`)
  - `bbox_2d`: MVP point set = 4 corners `(x1,y1),(x2,y1),(x2,y2),(x1,y2)` derived from `[x1,y1,x2,y2]`
- Compute an OT plan `T` via Sinkhorn on a chosen cost (L1 or L2 in norm1000 space), and treat `T` as a stop-grad aligner.
- Use barycentric projection ONLY:
  - `g_hat_i = sum_j ((T_ij / sum_j T_ij) * g_j)`
- Convert each `g_hat_i` into unimodal soft labels `q(x)` and `q(y)` over coord bins.
- Apply token-level coord supervision at the predicted object’s supervised coord token indices (poly vertices, or bbox tokens) using those `q` targets.

The trainer SHALL NOT implement or reference any “mixture” target construction for this OT alignment.

#### Scenario: Poly prediction is supervised by barycentric-projected targets
- **GIVEN** a matched pair where the predicted geometry is `poly`
- **WHEN** OT+barycentric target construction runs
- **THEN** each predicted poly coord token position receives a unimodal soft target derived from barycentric projection onto the GT shape
- **AND** the resulting targets are used for coord-token supervision only (no decoded coordinate regression losses).

### Requirement: Unified loss definition uses token masks (no decoded-coordinate losses/metrics)
The trainer SHALL compute a single total loss from the logits of the ONE forward pass on `Y_train` by applying per-token supervision masks.

At ALL `<|coord_*|>` supervised positions (both: matched coord slots in the rollout prefix AND coord tokens in the appended GT tail), the trainer SHALL compute:

`L_coord = softCE(q, p) + λ * W1(p, q) + λ_gate * GateMassLeak(p_full_vocab)`

where:
- `p` is the coord-bin distribution derived from logits restricted to the coord sub-vocabulary,
- `q` is a unimodal soft target over coord bins (Gaussian-like, configured by σ), and
- `GateMassLeak` penalizes probability mass outside the coord sub-vocabulary at coord positions.

For non-coordinate tokens in the appended GT tail segment, the trainer SHALL compute standard hard CE over the full vocabulary.
For this rollout-matching rollout, the trainer SHALL ignore (mask out) CE supervision for tokens that correspond to the JSON string *value* of `desc` fields in the appended GT tail (i.e., the token span inside `"desc": "<VALUE>"`). JSON structure tokens (braces/quotes/keys/colons/commas) in the appended GT tail remain supervised by CE.

The trainer SHALL NOT compute any of the legacy decoded-coordinate losses or metrics:
- no expectation/argmax/median decoding losses,
- no L1 regression,
- no IoU/GIoU/maskIoU loss terms,
- no polygon mask losses, smoothness losses, or geometry regularizers,
- no IoU/GIoU/maskIoU metric logging.

#### Scenario: Prefix coord tokens can be supervised without supervising prefix text tokens
- **GIVEN** a sample with at least one matched predicted object in the rollout prefix
- **WHEN** losses are computed for that sample
- **THEN** coord tokens at matched coord-slot indices in the prefix contribute `L_coord`
- **AND** non-coord tokens in the rollout prefix contribute neither CE nor coord loss (they are masked out)
- **AND** appended tail non-coord tokens contribute standard CE EXCEPT `desc` value tokens, which are masked out.

### Requirement: Canonical encoding and supervision index sanity checks
The ONE teacher-forced forward pass SHALL use the exact same prompt/messages encoding (chat template + image tokens placement) as rollout generation.

Labels SHALL align to assistant response tokens only; prompt tokens MUST be `ignore_index` (or equivalent).

The trainer MUST implement two engineering sanity checks:
- (a) prompt+image prefix tokenization matches generation (e.g., `len` and/or hash of the prompt token IDs),
- (b) all supervised `coord_token_indices` fall within the assistant-label span (never into the prompt span).

#### Scenario: Supervision indices are validated against assistant span
- **GIVEN** a sample with computed coord token indices for self-context supervision in the rollout prefix
- **WHEN** the trainer builds loss masks for the forward pass
- **THEN** it asserts every supervised coord index lies within the assistant portion of the encoded sequence
- **AND** it errors clearly if any index points into the prompt/image prefix (preventing silent misalignment).

### Requirement: Training-time counters expose parsing/matching health (without geometry metrics)
The trainer SHALL expose counters that record:
- number of predicted objects parsed as valid vs invalid (dropped),
- number of objects excluded due to coord-slot alignment ambiguity,
- match rate (#matched vs #GT),
- number of FN appended objects,
- number of gating rejections,
- rollout decoding mode (greedy/beam) and any truncation flags.

These counters SHALL NOT include IoU/GIoU/maskIoU numeric metric logging (those values are used internally for matching only).

#### Scenario: Parse failures are visible without crashing
- **GIVEN** a batch where some samples have malformed rollout JSON objects
- **WHEN** the trainer processes that batch
- **THEN** invalid objects are dropped and counted
- **AND** the training step completes without crashing due to mandatory FN append supervision in the tail.

## Validation (non-normative)
- OpenSpec validation:
  - `openspec validate 2026-01-15-add-rollout-matching-trainer --strict`
- Unit tests:
  - `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m pytest -q tests/test_rollout_matching_sft.py -q`
- Current rollout behaviour reference (20-sample smoke):
  - `output/infer/rollout_ckpt3106_smoke/pred.jsonl`
  - Raw-output pattern:
    - Many samples end with `<|im_end|>` (strip required for strict JSON parsing).
    - A minority are truly truncated mid-object (typically within poly coord arrays), which motivates suffix-only trimming to the last complete object boundary.
