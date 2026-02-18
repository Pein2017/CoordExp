## ADDED Requirements

### Requirement: Encoded fusion samples include stable join metadata
When the fusion dataset builds an encoded sample, it SHALL attach stable metadata keys to the encoded mapping:
- `sample_id` (int)
- `dataset` (string dataset ID)
- `base_idx` (int index into the dataset’s base record array)

If the system cannot attach this metadata, it MUST fail fast with an actionable error message.

#### Scenario: Encoded sample carries metadata
- **WHEN** `FusionCaptionDataset` encodes a sample for training
- **THEN** the returned encoded mapping includes `sample_id`, `dataset`, and `base_idx`.

#### Scenario: Metadata attachment failure is fatal
- **WHEN** encoded output is not a mutable mapping and metadata cannot be attached
- **THEN** dataset iteration fails fast with an error describing which metadata keys could not be attached.

### Requirement: Prompt injection is restored deterministically
If dataset encoding temporarily overrides template prompts for a specific sample (for example, `template.system`), it MUST restore the original prompt value before encoding the next sample.

Failure to apply the override or restore the original value MUST fail fast to prevent cross-sample prompt leakage.

#### Scenario: No prompt leakage across samples
- **WHEN** a sample is encoded with an injected system prompt
- **THEN** subsequent samples use their configured system prompt, not the injected value
- **AND** failure to restore stops the run with an explicit error.
