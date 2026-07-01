## ADDED Requirements

Define the model-facing assistant output format as CoordJSON (a JSON-like DSL where coord tokens are emitted as bare literals) and define the required conversion to strict RFC 8259 JSON (parseable by `json.loads`) for downstream components that require standards-compliant JSON (e.g., evaluation, visualization, and artifact parsing).

### Requirement: Assistant output is CoordJSON-only with a top-level `objects` array
The system SHALL render each assistant output as CoordJSON text that is consumed by the Qwen3-VL chat template.

Normative contract:
- The assistant message content MUST be CoordJSON-only (no natural-language prefix/suffix, no markdown fences).
- The top-level CoordJSON value MUST be an object with exactly one key: `"objects"`.
- The value of `"objects"` MUST be an array (possibly empty).
- The stable empty form MUST be exactly `{"objects": []}` under the canonical serializer.

#### Scenario: Empty output uses a stable canonical form
- **WHEN** the system needs to represent “no objects”
- **THEN** it emits `{"objects": []}` as the assistant output content
- **AND** no other top-level keys are present.

### Requirement: Each record is closed-schema and uses exactly one geometry field plus `desc`
Each element of the top-level `"objects"` array SHALL be a record.

Normative record schema:
- Each record MUST contain:
  - exactly one geometry key: `bbox_2d` OR `poly` (but not both), and
  - exactly one text key: `desc`.
- Records MUST NOT contain any other keys (e.g., `label`, `score`, `category_id`).
- `desc` MUST be a non-empty string after trimming whitespace.

#### Scenario: Bbox record schema is minimal and closed
- **WHEN** a bbox-mode record is emitted
- **THEN** it contains exactly `bbox_2d` and `desc`
- **AND** it contains no other keys.

#### Scenario: Poly record schema is minimal and closed
- **WHEN** a poly-mode record is emitted
- **THEN** it contains exactly `poly` and `desc`
- **AND** it contains no other keys.

#### Scenario: Mixed-geometry record is invalid
- **WHEN** a record contains both `bbox_2d` and `poly`
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

### Requirement: CoordTok literals are allowed only inside geometry arrays
CoordJSON SHALL support a coord-token literal (CoordTok) with the exact surface form:

`<|coord_k|>` where `k` is a base-10 integer in `[0, 999]`.

Normative placement and validation:
- Bare CoordTok literals MUST appear only as array elements inside `bbox_2d` or `poly` values.
- Bare CoordTok literals MUST NOT appear as object keys.
- Bare CoordTok literals MUST NOT appear as values of non-geometry keys (e.g., `"desc": <|coord_1|>`).
- CoordTok-like substrings inside JSON strings (e.g., inside `desc`) are permitted and are treated as ordinary text; they MUST NOT be interpreted as coordinates and MUST NOT be converted to integers by the CoordJSON → strict-JSON conversion.

#### Scenario: Out-of-range CoordTok is rejected
- **WHEN** a geometry array contains `<|coord_1000|>` (or any `k` outside `[0, 999]`)
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

### Requirement: Geometry arity constraints are enforced
The system SHALL enforce geometry arity constraints for both the CoordJSON records and the strict JSON produced after conversion.

Normative constraints:
- `bbox_2d` MUST be an array of exactly 4 CoordTok literals (CoordJSON) and exactly 4 integers (strict JSON).
- `poly` MUST be a flat array of CoordTok literals (CoordJSON) / integers (strict JSON) with:
  - even length, and
  - at least 6 values (≥ 3 points).

#### Scenario: bbox_2d must have exactly four coords
- **WHEN** a record has a `bbox_2d` array whose length is not 4
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

#### Scenario: poly must have even length and at least three points
- **WHEN** a record has `poly` length < 6 or odd length
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

#### Scenario: Geometry arrays are CoordTok-only in CoordJSON (no quoted tokens)
- **WHEN** a CoordJSON record contains quoted coord tokens inside a geometry array (e.g., `{"bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"], "desc": "cat"}`)
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

#### Scenario: Geometry arrays are CoordTok-only in CoordJSON (no integer literals)
- **WHEN** a CoordJSON record contains integer literals inside a geometry array (e.g., `{"bbox_2d": [1, 2, 3, 4], "desc": "cat"}`)
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

#### Scenario: Poly must be a flat array (no nested vertex-pair arrays)
- **WHEN** a CoordJSON record encodes poly as nested arrays (e.g., `{"poly": [[<|coord_1|>, <|coord_2|>], [<|coord_3|>, <|coord_4|>], [<|coord_5|>, <|coord_6|>]], "desc": "triangle"}`)
- **THEN** strict validation fails fast for cooked SFT/GT
- **AND** salvage parsing drops that record for rollouts.

### Requirement: Canonical serialization order is controlled by `custom.object_field_order`
The system SHALL define a canonical CoordJSON serialization for each record that is controlled by `custom.object_field_order`:
- `geometry_first`: the record MUST serialize the geometry key (`bbox_2d` or `poly`) before `desc`.
- `desc_first`: the record MUST serialize `desc` before the geometry key (`bbox_2d` or `poly`).

Canonical serialization MUST be deterministic and MUST NOT depend on incidental host-language dict ordering.

#### Scenario: geometry-first record serialization is enforced for cooked targets
- **WHEN** cooked SFT/GT targets are serialized with `custom.object_field_order: geometry_first`
- **THEN** each emitted record begins with the geometry key (`bbox_2d` or `poly`)
- **AND** `desc` appears after that geometry key.

#### Scenario: desc-first record serialization is enforced for cooked targets
- **WHEN** cooked SFT/GT targets are serialized with `custom.object_field_order: desc_first`
- **THEN** each emitted record begins with `desc`
- **AND** the concrete geometry key (`bbox_2d` or `poly`) appears after `desc`.

### Requirement: Order validation policy is explicit and centralized
The system SHALL treat `custom.object_field_order` as both:
- a canonical serialization rule for deterministic stages (cooked SFT/GT), and
- a record-level validity constraint for rollout predictions.

Normative policy:
- Cooked SFT/GT strict mode: if a record’s key order violates the configured `custom.object_field_order`, the system MUST fail fast (raise an error) and MUST NOT silently reorder or drop that record.
- Rollout parsing / salvage mode: if a predicted record’s key order violates the configured `custom.object_field_order`, that record MUST be marked invalid and DROPPED; the sample MUST continue using remaining valid records (robustness).
- Rollout-matching strict parsing MUST apply the same record-level drop behavior for order-mismatched predicted records as salvage parsing does.

This policy is intentionally stricter for rollouts than the legacy order-insensitive contract, because field order is an explicit ablation axis and order-mismatched predictions are treated as non-conformant outputs for that run.

#### Scenario: geometry-first run rejects desc-first cooked record and drops desc-first rollout record
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** a cooked SFT/GT record is encountered in the order `{"desc": "cat", "bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>]}`
- **THEN** strict-mode validation fails fast with an error identifying that record.
- **WHEN** a rollout prediction record is encountered in the same (order-mismatched) form
- **THEN** salvage parsing drops that record (no crash) and continues.

#### Scenario: desc-first run rejects geometry-first cooked record and drops geometry-first rollout record
- **GIVEN** `custom.object_field_order: desc_first`
- **WHEN** a cooked SFT/GT record is encountered in the order `{"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>], "desc": "cat"}`
- **THEN** strict-mode validation fails fast with an error identifying that record.
- **WHEN** a rollout prediction record is encountered in the same (order-mismatched) form
- **THEN** salvage parsing drops that record (no crash) and continues.

### Requirement: Canonical CoordJSON formatting is fixed for deterministic stages
Because downstream training uses token-aligned supervision, the exact assistant text emitted by deterministic stages MUST be stable and deterministic.
Because Qwen3/Qwen3-VL tokenization is whitespace-sensitive, separator spaces are part of the contract (not cosmetic).

For cooked SFT/GT assistant outputs, the system SHALL emit canonical CoordJSON formatting with the following rules:
- No whitespace is emitted outside JSON strings, except for the single spaces mandated by separators below.
- No newline characters are emitted outside JSON strings.
- Array element separator MUST be `, ` (comma + single space).
- Object member separator MUST be `, ` (comma + single space).
- Key/value separator MUST be `: ` (colon + single space).
- All keys MUST use double quotes and MUST match the schema exactly (`"objects"`, `"bbox_2d"`, `"poly"`, `"desc"`).
- `desc` MUST be serialized as an RFC 8259 JSON string, including any required escaping.
- Non-ASCII characters in `desc` MUST be preserved as Unicode characters (no ASCII-escaping); this is equivalent to `json.dumps(..., ensure_ascii=False)` behavior.
- CoordTok literals MUST be emitted as bare tokens (no surrounding quotes) inside geometry arrays.
- No trailing commas are permitted.

#### Scenario: Canonical geometry-first bbox output matches the golden string
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** a single bbox record with `desc="cat"` and `bbox_2d=[<|coord_12|>, <|coord_56|>, <|coord_200|>, <|coord_512|>]` is serialized
- **THEN** the emitted assistant text equals `{"objects": [{"bbox_2d": [<|coord_12|>, <|coord_56|>, <|coord_200|>, <|coord_512|>], "desc": "cat"}]}`.

#### Scenario: Canonical desc-first poly output matches the golden string
- **GIVEN** `custom.object_field_order: desc_first`
- **WHEN** a single poly record with `desc="triangle"` and `poly=[<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>, <|coord_5|>, <|coord_6|>]` is serialized
- **THEN** the emitted assistant text equals `{"objects": [{"desc": "triangle", "poly": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>, <|coord_5|>, <|coord_6|>]}]}`.

### Requirement: CoordJSON is converted to strict RFC 8259 JSON for downstream pipeline usage
Before any downstream component uses `json.loads` or schema validation on assistant outputs, the system SHALL convert CoordJSON to strict RFC 8259 JSON (parseable by `json.loads`).

Normative strict JSON representation:
- The strict JSON top-level is an object: `{"objects": [...]}`
- Each strict JSON record contains:
  - `bbox_2d: [int,int,int,int]` OR `poly: [int,...]`, and
  - `desc: "<string>"`
- Geometry integers MUST lie in `[0, 999]`.

#### Scenario: CoordTok literals convert to integer bins
- **WHEN** CoordJSON contains `{"objects": [{"bbox_2d": [<|coord_12|>, <|coord_56|>, <|coord_200|>, <|coord_512|>], "desc": "cat"}]}`
- **THEN** the conversion produces strict JSON whose `bbox_2d` value is `[12, 56, 200, 512]`
- **AND** the output is valid RFC 8259 JSON parseable by `json.loads`.

#### Scenario: Poly CoordTok literals convert to integer bins
- **WHEN** CoordJSON contains `{"objects": [{"poly": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>, <|coord_5|>, <|coord_6|>], "desc": "triangle"}]}`
- **THEN** the conversion produces strict JSON whose `poly` value is `[1, 2, 3, 4, 5, 6]`
- **AND** the output is valid RFC 8259 JSON parseable by `json.loads`.

### Requirement: Conversion supports strict mode (fail-fast) and salvage mode (robust)
The CoordJSON → strict JSON conversion SHALL support two modes:

- **Strict mode** (cooked SFT/GT):
  - Any contract violation MUST raise a fail-fast error.
  - Errors MUST be record-addressable (include `objects[i]` index at minimum).
- **Salvage mode** (rollout predictions):
  - Invalid records MUST be dropped.
  - If the decoded assistant text is truncated and the final record is incomplete, the converter MUST drop only that incomplete tail record and preserve earlier valid records.
  - The converter MUST ignore leading/trailing non-CoordJSON content around the first valid container in rollout predictions:
    - it MUST scan left-to-right to locate the first substring that can be interpreted as a top-level `{"objects": [...]}` container under this contract,
    - it MUST discard any prefix before that container begins,
    - if multiple valid containers are present in one decoded rollout string, it MUST select the first valid container and treat any later text (including later containers) as trailing junk to discard,
    - any brace/bracket scanning used to identify these boundaries MUST be string-aware (i.e., braces/brackets inside `desc` strings do not affect structure depth).
  - If the top-level CoordJSON cannot be interpreted as an object of the form `{"objects": [...]}` (e.g., missing opening `{`, missing `"objects"`, `"objects"` is not an array, or extra top-level keys exist), the converter MUST treat the prediction as a sample-level parse failure:
    - it MUST report the failure (e.g., via a parse-fail flag/counter), and
    - it MUST return a safe empty strict JSON object of exactly `{"objects": []}` (so downstream `json.loads` always succeeds).
    - downstream components MUST treat this sample as having zero valid predicted objects (i.e., skip using rollout content for this sample).
    - rollout-matching training MUST still be able to proceed deterministically for this sample by treating the rollout prefix as empty and FN-appending all GT objects (see rollout-matching / Stage-2 AB delta specs).
  - The converter’s strict-JSON output is a pipeline-facing representation for structural parsing, matching inputs, and metrics.
    - It MUST NOT be assumed to be an append-ready prefix for teacher-forced sequence construction; `Y_rollout_prefix` is defined on raw rollout tokens and is constructed via suffix-only trimming (see rollout-matching delta spec).
    - It MUST NOT be used to derive the predicted-record list for rollout-matching training, because salvage extraction may discard leading junk that is not present in the token-aligned `Y_rollout_prefix`. Training-time matching/supervision MUST derive predictions from the same token-aligned scan/cut boundary that yields `Y_rollout_prefix` (see rollout-matching / Stage-2 AB delta specs).
  - The converter MUST NOT perform token-inserting “repair” (no adding braces/quotes/commas).
    - This “no repair” rule applies to how decoded rollout text is interpreted/salvaged; it does NOT forbid the defined CoordTok→int conversion performed when producing strict JSON output.
    - It MAY discard leading/trailing junk outside the selected container, drop invalid records, and drop an incomplete tail record caused by truncation.
    - It MUST NOT add new structural tokens to the decoded rollout text or alter the non-geometry semantic content of valid records.

#### Scenario: Strict mode fails fast on a single bad record
- **WHEN** cooked SFT/GT conversion encounters a record-level violation (e.g., empty desc or wrong geometry arity)
- **THEN** conversion raises an error that identifies the offending record index
- **AND** the training data build fails fast rather than silently dropping the record.

#### Scenario: Salvage mode preserves valid prefix and drops invalid tail
- **WHEN** a rollout prediction contains a valid first record and a second record that is truncated mid-array
- **THEN** salvage conversion returns strict JSON containing only the first record
- **AND** training/eval proceeds using that valid prefix.

#### Scenario: Salvage mode treats malformed top-level as a parse-fail sample and returns empty objects
- **WHEN** a rollout prediction does not contain a valid top-level `{"objects": [...]}` container (e.g., missing `{` or missing `"objects"`)
- **THEN** salvage conversion reports a sample-level parse failure
- **AND** it returns strict JSON equal to `{"objects": []}`.

#### Scenario: Salvage mode ignores leading/trailing junk around the container
- **WHEN** a rollout prediction contains non-CoordJSON prefix/suffix text around a valid container (e.g., `Answer: {"objects": [{"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>], "desc": "cat"}]}<|im_end|>`)
- **THEN** salvage conversion extracts the container (or a salvageable prefix of it) and drops the surrounding junk
- **AND** it produces strict JSON parseable by `json.loads` with geometry integers `[1, 2, 3, 4]`.

#### Scenario: Salvage mode selects the first valid container when multiple are present
- **WHEN** a rollout prediction contains multiple valid `{"objects": [...]}` containers in one decoded string (e.g., `{"objects": [{"bbox_2d": [<|coord_1|>, <|coord_2|>, <|coord_3|>, <|coord_4|>], "desc": "first"}]}{"objects": [{"bbox_2d": [<|coord_5|>, <|coord_6|>, <|coord_7|>, <|coord_8|>], "desc": "second"}]}`)
- **THEN** salvage conversion uses only the first container and discards the later container as trailing junk
- **AND** it produces strict JSON with a single record whose `bbox_2d` equals `[1, 2, 3, 4]` and whose `desc` equals `"first"`.
