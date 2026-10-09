# pi-cache-usage-breakdown Specification

## Purpose
Distinguish ordinary inference cache reuse from maintenance usage while preserving the authoritative lifetime token and cost accounting of a Pi session.
## Requirements
### Requirement: Categorized usage reconciles with lifetime totals
The Web session statistics SHALL add separate categories for ordinary inference, compaction, branch summary, cache warming and other usage. All input, output, cache-read, cache-write and cost amounts MUST belong to exactly one category, and category token sums MUST equal existing all-file totals exactly. Category monetary sums MUST reconcile within documented USD1e-9 floating-point tolerance, with the existing chronological monetary total retained as authoritative. Unknown usage kinds and billed nested/tool-result usage MUST remain counted. Context edits, branches and compaction MUST NOT erase historical billed usage.

#### Scenario: Compaction lowers pooled cache fraction
- **WHEN** ordinary inference has cached input and a compaction adds uncached input
- **THEN** statistics preserve the lower pooled fraction and separately expose the ordinary fraction and compaction usage

#### Scenario: Unknown usage and tool result
- **WHEN** a session includes unknown usage kinds or usage on a tool-result message
- **THEN** those amounts appear in a defined category exactly once and reconcile to lifetime totals

### Requirement: Cache fractions and maintenance display are unambiguous
The UI SHALL retain the lifetime cache fraction using cache-read divided by input plus cache-read plus cache-write, exclude output from that denominator, and display ordinary and maintenance breakdowns with labels that distinguish these scopes. Empty denominators SHALL be shown as unavailable rather than a fabricated percentage. Incremental updates, reload and reconciliation MUST preserve category totals without duplicate counting or resetting after compaction.

#### Scenario: Streaming and reload agree
- **WHEN** a completed request is observed live and later loaded from the journal after compaction
- **THEN** both views expose the same category amounts without double counting the completed request

#### Scenario: No prompt usage
- **WHEN** a category has no input-class tokens
- **THEN** its cache fraction is unavailable rather than zero or one hundred percent

