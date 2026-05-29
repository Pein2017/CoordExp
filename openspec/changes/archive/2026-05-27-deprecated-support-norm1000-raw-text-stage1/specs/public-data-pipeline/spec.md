## ADDED Requirements

### Requirement: Canonical norm JSONL is a documented benchmark-ready output
The public-data pipeline SHALL treat canonical `*.norm.jsonl` artifacts as a
first-class prepared output, not merely an intermediate file.

Normative behavior:
- canonical prepared presets SHALL continue to emit:
  - `<split>.jsonl`
  - `<split>.norm.jsonl`
  - `<split>.coord.jsonl`
- documentation and examples SHALL identify `<split>.norm.jsonl` as the
  supported Stage-1 benchmark input for norm1000 raw-text geometry
  experiments,
- this benchmark path SHALL reuse the canonical preset root rather than
  requiring a separate derivation branch.

#### Scenario: Operator can launch raw-text benchmark from canonical norm JSONL
- **GIVEN** a canonical prepared preset root
- **WHEN** an operator wants to run a raw-text norm1000 Stage-1 benchmark
- **THEN** the documented dataset surface is `<split>.norm.jsonl`
- **AND** no extra public-data conversion step is required.
