## MODIFIED Requirements

### Requirement: Single-path training constructs one canonical teacher-forced target sequence
When rollout-matching training is enabled, the trainer SHALL implement a **single** training path that is expressed as:
- one canonical assistant token sequence per sample (`Y_train`), and
- one forward pass on that sequence, with per-token supervision masks.

There SHALL NOT exist separate training “paths” (e.g., “reordered-GT SFT” vs “self-context”); all supervision SHALL be expressed as per-token loss masks on a single teacher-forced forward pass.

The trainer SHALL construct the canonical assistant target sequence as:
- `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

where:
  - `Y_rollout_prefix` is a **prefix** of the model’s rollout assistant `response_token_ids` produced by autoregressive generation, interpreted under the project’s CoordJSON assistant schema (top-level `{"objects": [...]}`).
  - The trainer MAY perform **suffix-only trimming** to define `Y_rollout_prefix` so it is safe for append:
    - treat `<|im_end|>` as a hard stop and strip it (and any suffix after it) when present,
    - drop any trailing incomplete / invalid suffix tokens beyond the last complete predicted record boundary (e.g., when rollout is truncated mid-record),
    - drop the final top-level CoordJSON closing suffix `]}` so the prefix ends inside an **open** `"objects"` array ready for append.
    - NOTE: Some tokenizers fuse closing punctuation (e.g., a single token may decode to `]}` or `}]}`).
      - In those cases, the desired suffix cut can fall **inside** the final token.
      - The trainer MAY realize the cut by replacing ONLY the final token with a shorter tokenization that decodes to a strict prefix of that token’s decoded text (e.g., `}]}` → `}` when the cut boundary is the record-end `}`), while keeping all earlier token IDs unchanged.
  - The trainer SHALL NOT edit or re-tokenize any token **before** the cut boundary used for the prefix (no decode+re-encode; no pretty-printing; no key sorting).
  - Failure behavior: if the rollout does not contain an opening `{` OR the prefix cannot be made append-ready via suffix-only trimming, the trainer SHALL treat the rollout prefix as empty and use the literal append-ready prefix `{"objects": [` as `Y_rollout_prefix` (no prefix supervision; all GT objects become FN and are appended).
    - Consistency requirement: in this failure behavior, the trainer MUST also treat the rollout as having zero valid predicted objects for matching/supervision (i.e., the predicted-record list is empty).
- `FN_gt_objects` are the GT objects that are unmatched after matching (see matching requirement).
- `SerializeAppend(FN_gt_objects)` emits GT objects in the project’s CoordJSON-only assistant schema (`{"objects": [{...}, {...}]}`) as an **append fragment**:
  - it SHALL emit **only** comma-separated record payloads like `{"bbox_2d": [...], "desc": "..."}` or `{"poly": [...], "desc": "..."}` entries (no outer `{}` wrapper, no surrounding `"objects": [` wrapper),
  - it MUST use the canonical CoordJSON formatting rules defined by the CoordJSON output-format contract (single-space separators `, ` and `: `; no indentation; no newlines outside strings; no extra whitespace),
  - it SHALL decide whether to emit a leading comma based on the last non-whitespace character of the decoded `Y_rollout_prefix`:
    - if the last non-whitespace character is `[` or `,`, it SHALL NOT emit a leading comma,
    - if the last non-whitespace character is `}`, it SHALL emit a leading `, `,
    - otherwise the prefix is not append-ready and the trainer MUST apply the failure behavior above (treat `Y_rollout_prefix` as empty and use the literal prefix `{"objects": [`),
  - it SHALL terminate by emitting the top-level CoordJSON closing suffix `]}` so `Y_train` is a valid top-level object before `EOS`.

There SHALL be exactly ONE forward pass per sample on the canonical encoding (same chat template as generation).

#### Scenario: Training uses one sequence and one forward pass
- **GIVEN** a batch under rollout-matching training
- **WHEN** the trainer executes one training step
- **THEN** it performs exactly one forward pass per sample on `Y_train`
- **AND** it computes exactly one total loss from that forward pass by applying per-token supervision masks.

### Requirement: FN append serialization honors configured object field order
When rollout-matching builds `Y_train` via mandatory FN append, each appended record payload SHALL follow `custom.object_field_order`.

Normative behavior:
- `desc_first`: append payload uses `{desc, bbox_2d}` or `{desc, poly}` depending on object geometry type.
- `geometry_first`: append payload uses `{bbox_2d, desc}` or `{poly, desc}` depending on object geometry type.
- Geometry key can be `bbox_2d` or `poly`.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.

This requirement applies only to key order within each appended record payload and MUST NOT alter:
- predicted record appearance-order parsing,
- matching order semantics.

#### Scenario: geometry-first changes only per-record key order in FN append
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B has unmatched GT objects to append
- **WHEN** `SerializeAppend(FN_gt_objects)` is produced
- **THEN** each appended record places its concrete geometry key (`bbox_2d` or `poly`) before `desc`.

#### Scenario: desc-first remains baseline append layout
- **GIVEN** `custom.object_field_order` is omitted or set to `desc_first`
- **WHEN** FN append fragment is serialized
- **THEN** appended record payloads keep `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Predicted order is defined by raw rollout text appearance (no silent reordering)
“Predicted order” SHALL be defined as the appearance order in the raw rollout string (the assistant response text decoded from `response_token_ids`).

Parsers/utilities used for rollout-matching training SHALL NOT re-serialize/pretty-print the rollout prefix in any way that changes tokenization.

When predicted objects are represented as an `"objects"` array, the predicted order SHALL be the array element order as it appears in the raw rollout text.
For rollout-matching training, parsing MUST be anchored to the first top-level CoordJSON container encountered in the decoded rollout text; any content after that container closes (including later containers) SHALL be treated as trailing junk and ignored.

#### Scenario: Array element order is preserved for matching order
- **GIVEN** a rollout response string containing `{"objects": [{"bbox_2d": [...], "desc": "a"}, {"bbox_2d": [...], "desc": "b"}]}`
- **WHEN** predicted records are enumerated for matching
- **THEN** the predicted order is the first array element followed by the second array element
- **AND** the trainer does not reorder them.

### Requirement: Strict parsing drops invalid predicted objects (no repair)
The trainer SHALL require strict, schema-conformant predicted records.

If a predicted record is malformed or violates schema (e.g., missing brackets, extra commas, nested/unexpected keys, wrong coord count, non-coord tokens in geometry arrays, invalid geometry key, missing/empty `desc`, both `bbox_2d` and `poly` present, or record key order that violates the configured `custom.object_field_order`), that record SHALL be marked invalid and DROPPED:
- invalid records SHALL NOT participate in matching, and
- invalid records SHALL NOT contribute to self-context supervision.

The trainer SHALL NOT perform token-inserting “JSON repair” for rollout-matching training (no adding braces/quotes, no filling missing tokens, no re-serializing CoordJSON).

However, the trainer MAY perform **suffix-only trimming** of the rollout tokens to:
- drop trailing incomplete text when rollout is truncated, and/or
- strip a terminal `<|im_end|>` token, and/or
- drop the final top-level `]}` to enable FN append under the canonical `Y_train` construction.

All tokens **before** the suffix-trim boundary SHALL remain unchanged; “dropping” affects only (a) the parsed record list and loss masks and (b) the chosen prefix cut for `Y_rollout_prefix`.

#### Scenario: One malformed record does not block training
- **GIVEN** a rollout response containing 3 record entries in appearance order
- **AND** the middle record entry is malformed (e.g., wrong coord count)
- **WHEN** strict parsing runs
- **THEN** exactly that record is marked invalid and excluded
- **AND** the other valid record entries remain eligible for matching and supervision
- **AND** training still proceeds because FN append provides teacher-forced supervision for unmatched GT.

#### Scenario: Truncated rollout tail is trimmed (suffix-only) and training proceeds
- **GIVEN** a rollout response that is truncated mid-record (no balanced record is available)
- **WHEN** the trainer constructs `Y_rollout_prefix` via suffix-only trimming
- **THEN** it drops the incomplete trailing suffix and keeps only the prefix up to the last complete record boundary (or just `{"objects": [` when none)
- **AND** it appends `SerializeAppend(FN_gt_objects)` and `EOS` to form a valid `Y_train`
- **AND** the training step completes without crashing.

### Requirement: Coord-slot token indices are derived from token-aligned parsing (object-level validity)
Self-context supervision SHALL NOT rely on searching for repeated `<|coord_k|>` patterns in the text.

Coord-slot token indices SHALL be obtained deterministically from parsing the rollout token sequence (or a token-aligned parse), producing for each VALID predicted record:
- `bbox_2d`: exactly 4 coord-token indices (for `[x1, y1, x2, y2]`), and
- `poly`: exactly `2N` coord-token indices with `N >= 3` and even length.

If coord-slot indices for a record are not uniquely determined / not trusted, that record SHALL be excluded from self-context supervision (record-level exclusion). The sample SHALL still train via mandatory FN append in the tail.

#### Token-aligned parsing / prefix trimming algorithm (normative sketch)
The trainer SHOULD implement a single streaming pass over the rollout assistant token IDs to determine:
- (a) a safe `Y_rollout_prefix` cut boundary (append-ready), and
- (b) per-record geometry kind + coord-token indices in **appearance order** (array element order).

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
     - whether the scanner is inside the top-level `"objects"` array value,
     - current record index `i` (when inside an objects-array element dict),
     - current geometry key (`bbox_2d` or `poly`) and whether the scanner is currently inside its array value.
   - The scanner MUST anchor to the first top-level `{"objects": [...]}` container only; after that container closes, scanning for predicted records/cut candidates MUST stop and any later text MUST be ignored as trailing junk.
4. **Determine appearance-order records and coord-token indices**:
   - When the scanner enters a new dict that is a direct element of the `"objects"` array, start a new predicted record context in appearance order (array order).
   - Within that record dict, the scanner MUST locate exactly one geometry key (`bbox_2d` or `poly`) and then enter “capture mode” on the next `[` that begins that geometry’s array.
   - While in capture mode for a geometry array, every assistant token position `p` whose token id is in `coord_id_set` SHALL be appended to that record’s `coord_token_indices` list.
   - While in capture mode, the scanner MUST enforce the CoordTok-only geometry-array contract:
     - any non-CoordTok geometry element (e.g., integer literals like `1`, quoted coord tokens like `"<|coord_1|>"`, or nested arrays like `[[<|coord_1|>, <|coord_2|>], ...]`) makes the record invalid and MUST be dropped,
     - implementations SHOULD enforce this by rejecting any non-CoordTok token content inside the geometry-array span that is not purely structural punctuation/whitespace (e.g., `[`, `]`, `,`, and spaces).
   - Capture mode ends when the corresponding array `]` is closed (i.e., `bracket_depth` returns to the value it had immediately before the geometry array opened).
   - After capture ends:
     - `bbox_2d` MUST have exactly 4 coord-token indices, else the record is invalid.
     - `poly` MUST have an even number of indices and at least 6 total indices, else the record is invalid.
   - If a record contains multiple geometry keys, nested/unexpected geometry keys, unexpected keys, or the geometry array is not fully closed before the chosen prefix cut boundary, the record is invalid and MUST be dropped (no repair).
5. **Determine the append-ready `Y_rollout_prefix` cut boundary**:
   - During the same streaming scan, the trainer MUST record candidate cut positions corresponding to:
     - the point immediately after the `[` that opens the top-level `"objects"` array (valid empty-prefix cut), and
     - the end of the last complete record entry in that `"objects"` array.
   - A record-end candidate cut occurs after a `}` that closes an objects-array element dict while the scanner remains inside the `"objects"` array context.
   - The selected cut boundary SHALL be the last recorded candidate within the first container, after stripping any trailing end-of-turn tokens like `<|im_end|>`.
   - **Fused-suffix handling**: if the selected cut falls inside the final token, the trainer MAY replace ONLY that final token with a shorter tokenization that decodes to the needed substring (e.g., `}]}` → `}` when cutting at record-end), keeping all earlier token IDs unchanged.

#### Scenario: Ambiguous coord-slot alignment excludes record from self-context supervision
- **GIVEN** a predicted record whose geometry can be parsed but whose coord token indices cannot be uniquely aligned to `response_token_ids`
- **WHEN** the trainer builds self-context supervision masks
- **THEN** that record contributes no self-context coord loss
- **AND** its GT counterpart (if any) is treated as unmatched and included in `FN_gt_objects` for tail append.

## REMOVED Requirements

### Requirement: Field-order variation is schema-equivalent for strict parsing
**Reason**: Under CoordJSON + ordering-ablation runs, predicted records are required to follow the configured `custom.object_field_order`; records that violate the configured order are treated as invalid and dropped for rollout-matching training.

**Migration**: Ensure cooked targets are deterministically serialized in the configured order. For rollouts, update parsing/validation so a record is considered invalid when its key order violates the current `custom.object_field_order`.
