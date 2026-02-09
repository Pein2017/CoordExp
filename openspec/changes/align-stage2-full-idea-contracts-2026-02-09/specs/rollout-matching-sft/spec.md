# rollout-matching-sft Spec Delta

This is a delta spec for change `align-stage2-full-idea-contracts-2026-02-09`.

## MODIFIED Requirements

### Requirement: Single-path training constructs one canonical teacher-forced target sequence
When rollout-matching training is enabled, the trainer SHALL implement a **single** training path that is expressed as:
- one canonical assistant token sequence per sample (`Y_train`), and
- one forward pass on that sequence, with per-token supervision masks.

There SHALL NOT exist separate training "paths" (e.g., "reordered-GT SFT" vs "self-context"); all supervision SHALL be expressed as per-token loss masks on a single teacher-forced forward pass.

The trainer SHALL construct the canonical assistant target sequence as:
- `Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

where:
- `Y_rollout_prefix` is a **prefix** of the model's rollout assistant `response_token_ids` produced by autoregressive generation.
  - The trainer MAY perform **suffix-only trimming** to define `Y_rollout_prefix` so it is safe for append:
    - treat `<|im_end|>` as a hard stop and strip it (and any suffix after it) when present,
    - drop any trailing incomplete / invalid suffix tokens beyond the last complete predicted object boundary (e.g., when rollout is truncated mid-object),
    - drop the final top-level JSON closing brace `}` so the prefix ends in an **open** JSON object ready for append.
    - NOTE: Some tokenizers fuse closing punctuation (e.g., a single token may decode to `}}` or `}},`).
      - In those cases, the desired suffix cut can fall **inside** the final token.
      - The trainer MAY realize the cut by replacing ONLY the final token with a shorter tokenization that decodes to a strict prefix of that token's decoded text (e.g., `}}` -> `}`), while keeping all earlier token IDs unchanged.
  - The trainer SHALL NOT edit or re-tokenize any token **before** the cut boundary used for the prefix (no decode+re-encode; no pretty-printing; no key sorting).
  - Failure behavior: if the rollout does not contain an opening `{` OR the prefix cannot be made append-ready via suffix-only trimming, the trainer SHALL treat the rollout prefix as empty and use `Y_rollout_prefix = "{"` (no prefix supervision; all GT objects become FN and are appended).
- `FN_gt_objects` are the GT objects that are unmatched after matching (see matching requirement).
- `SerializeAppend(FN_gt_objects)` emits GT objects in the project's JSON-only assistant schema (object-index JSON mapping `object_{n}` -> `{desc, geometry}`) as an **append fragment**:
  - it SHALL emit **only** comma-separated `"object_{n}": {...}` entries (no outer `{}` wrapper),
  - it SHALL decide whether to emit a leading comma based on the last non-whitespace character of the decoded `Y_rollout_prefix`:
    - if the last non-whitespace character is `{` or `,`, it SHALL NOT emit a leading comma,
    - if the last non-whitespace character is `}`, it SHALL emit a leading `, `,
    - otherwise it SHALL error (the prefix is not append-ready).
  - it SHALL assign keys `object_{n}` starting from `n = max_object_index_in_prefix + 1` (or `n = 1` when no valid object index exists in the prefix),
    - `max_object_index_in_prefix` is the maximum integer `n` observed in any key matching the pattern `object_{n}` in the retained rollout prefix body,
    - this scan MUST include keys from entries later dropped by strict parse/validation/matching,
    - malformed keys that do not parse as `object_{n}` MUST be ignored,
  - it SHALL terminate by emitting the single top-level JSON closing brace `}` so `Y_train` is a valid JSON object before `EOS`.

There SHALL be exactly ONE forward pass per sample on the canonical encoding (same chat template as generation).

#### Scenario: Training uses one sequence and one forward pass
- **GIVEN** a batch under rollout-matching training
- **WHEN** the trainer executes one training step
- **THEN** it performs exactly one forward pass per sample on `Y_train`
- **AND** it computes exactly one total loss from that forward pass by applying per-token supervision masks.

#### Scenario: Highest retained object key controls FN start even when object is invalid
- **GIVEN** retained rollout prefix contains `object_2` (valid) and `object_9` (invalid object body)
- **WHEN** `SerializeAppend(FN_gt_objects)` assigns new keys
- **THEN** `max_object_index_in_prefix` is `9`
- **AND** the first FN key is `object_10`.
