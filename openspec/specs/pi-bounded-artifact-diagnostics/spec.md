# pi-bounded-artifact-diagnostics Specification

## Purpose
Provide compact, source-linked diagnostic views of Pi journals and explicitly selected JSON artifacts without altering the original evidence or exact-read tools.
## Requirements
### Requirement: Bounded journal projections preserve source evidence
The diagnostic reader SHALL provide metadata/usage, tool-call and selected text views of a user-selected session journal. It MUST disclose source line or entry identity, selected range, parsing completeness and display omissions; opaque signatures and encrypted checkpoint content MUST NOT appear in default output. Display limits MUST NOT truncate parsing before the selected range is processed, and the reader MUST NOT write the source.

#### Scenario: Large signature precedes requested usage
- **WHEN** a selected journal contains an oversized encrypted signature before later usage records
- **THEN** the reader parses the selected records, returns accurate later usage and source identities, and omits the opaque bytes from bounded output

#### Scenario: Journal is incomplete or malformed
- **WHEN** a selected journal contains malformed complete lines or a partially appended final line
- **THEN** the reader discloses the affected source positions and incomplete coverage rather than reporting an unqualified complete total

### Requirement: Artifact projection uses explicit selectors
The reader SHALL support explicit JSON field selectors and bounded collection/text display with original selector identities and omission counts. It MUST reject missing or invalid selectors visibly and MUST NOT infer scientific meaning, silently change numeric values, or modify raw data. A file-size admission limit SHALL fail visibly rather than partially parse a JSON document. Numeric representations that cannot be emitted without value loss SHALL be preserved losslessly or visibly rejected before a rounded projection is returned.

#### Scenario: Unsupported numeric precision
- **WHEN** a selected artifact contains an unsafe integer such as9007199254740993 or a decimal that would be rounded by the reader
- **THEN** the reader preserves its value losslessly or rejects the unsupported representation visibly without emitting a different numeric value

#### Scenario: Scientific ledger selected fields
- **WHEN** a caller selects fields or rows of a ledger with exact numeric and boolean values
- **THEN** projected values retain their original types and values and any omitted rows or text are identified

#### Scenario: Generic reads remain exact
- **WHEN** a caller uses Pi's ordinary read tool rather than the explicit projection reader
- **THEN** the new diagnostic behavior does not alter its output or source semantics

