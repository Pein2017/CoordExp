## MODIFIED Requirements

### Requirement: Semantic description matching
The evaluator SHALL always run description matching via `sentence-transformers/all-MiniLM-L6-v2` when deriving COCO annotations. Predictions whose normalized descriptions are not mapped with cosine similarity ≥ `semantic_threshold` SHALL be dropped (counted in `unknown_dropped`) rather than assigned to synthetic categories.

Legacy configuration keys `unknown_policy` and `semantic_fallback` are unsupported and MUST fail fast if present (no backward/legacy support).

If the semantic encoder cannot be loaded (missing from the HuggingFace cache and download is not possible), the evaluator SHALL fail loudly with a clear error message (it SHALL NOT silently degrade into bucketed or dropped defaults).

#### Scenario: Description matching fails when the encoder is unavailable
- **GIVEN** any evaluation run and `sentence-transformers/all-MiniLM-L6-v2` cannot be loaded from caches or downloads
- **WHEN** the evaluator starts mapping descriptions
- **THEN** it raises a runtime error describing that the encoder is mandatory for evaluation and advising the user to ensure the model is cached/downloadable.

#### Scenario: Deprecated keys fail fast
- **WHEN** evaluation config includes `unknown_policy` or `semantic_fallback`
- **THEN** evaluation fails fast with an actionable error describing that these keys are unsupported and must be removed.
