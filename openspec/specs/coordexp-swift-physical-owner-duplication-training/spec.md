# coordexp-swift-physical-owner-duplication-training Specification

## Purpose
TBD - created by archiving change add-physical-owner-duplicate-rejection-and-recovery-training. Update Purpose after archive.
## Requirements
### Requirement: Reviewed physical-owner evidence controls training eligibility
The system SHALL admit a duplicate-negative row only when a reviewed image-local physical owner is already represented by a usable accepted prefix row and the later row is trusted to represent that same owner. The system SHALL keep official ground-truth-unmatched predictions neutral unless review establishes an eligible role.

#### Scenario: Confirmed duplicate is admitted
- **WHEN** a later row has trusted category and binding to a physical owner already represented by a usable accepted prefix row
- **THEN** the system admits that row as a covered-owner duplicate negative

#### Scenario: Unmatched or ambiguous prediction remains neutral
- **WHEN** a row is unmatched to official ground truth or has uncertain owner, category, geometry, adjacent-instance binding, or entity existence
- **THEN** the system gives that row no duplicate-negative or new-owner-positive training gradient

#### Scenario: Verified omitted object is positive evidence
- **WHEN** review confirms a real Common Objects in Context 80-category physical owner omitted by official ground truth and not covered by the replay prefix
- **THEN** the system may admit that owner as positive entity evidence without classifying the official mismatch as hallucination

### Requirement: Candidate generation provenance remains distinct from replay provenance
The system SHALL preserve the true candidate-generation prefix identity and the replay-prefix identity for every duplicate training event. It SHALL permit different identities only under typed exact-self-prefix transplant or complete-row-deletion rewrite evidence.

#### Scenario: Historical event retains exact-prefix equality
- **WHEN** an event does not declare duplicate-trajectory evidence
- **THEN** every candidate generation-prefix hash MUST equal the replay-prefix hash as in the existing StateBank contract

#### Scenario: Recovery row is transplanted to an earlier prefix
- **WHEN** a verified downstream recovery row is replayed before one or more duplicate rows
- **THEN** the event records both prefix hashes, the original trajectory and row index, the replay context kind, and the duplicate burst identity without rewriting generation provenance

#### Scenario: Cleaned prefix is constructed by row deletion
- **WHEN** a prefix is derived by deleting confirmed duplicate complete rows
- **THEN** the system records the removed row indices and retains exact stored integer row slices without decode-and-retokenize reconstruction

### Requirement: Local duplicate rejection uses a pure field-balanced complete-row preference
The system SHALL provide an opt-in loss that compares one verified new-owner row and one confirmed covered-owner duplicate row under the same replay prefix. The pairwise score SHALL average description mean log probability and trusted-coordinate mean log probability and SHALL exclude wrapper and delimiter tokens from that score.

#### Scenario: Complete-row pair is scored
- **WHEN** both rows have trusted physical-owner and coordinate evidence and the local duplicate rejection profile is active
- **THEN** the system computes `softplus(margin - positive_row_score + duplicate_row_score)` in 32-bit floating-point loss math and logs each field score, row score, margin, and selected-token count

#### Scenario: Untrusted geometry cannot enter the primary pairwise loss
- **WHEN** either candidate lacks trusted coordinate evidence
- **THEN** the system rejects the pair from the primary complete-row duplicate preference instead of silently scoring uncertain coordinates

#### Scenario: Positive-only control omits the duplicate gradient
- **WHEN** the recovery-positive-only control uses the same replay prefix and recovery row
- **THEN** the system applies complete-row positive imitation without a duplicate-negative term

### Requirement: Duplicate-cleaned imitation is explicitly counterfactual
The system SHALL support complete-row imitation under prefixes produced only by deleting confirmed duplicate rows, SHALL label those prefixes `counterfactual_rewritten`, and SHALL preserve the original unedited trajectory separately.

#### Scenario: Valid suffix continues after a burst
- **WHEN** a confirmed duplicate burst is followed by trusted new-owner rows
- **THEN** the system retains the first accepted occurrence of every owner, removes only confirmed duplicate rows, validates each successive rewritten transition, and emits cleaned complete-row imitation events until the first untrusted row

#### Scenario: Untrusted suffix stops automatic admission
- **WHEN** the next suffix row is invalid, a category error, owner ambiguous, binding ambiguous, or otherwise untrusted
- **THEN** the system stops automatic cleaned-suffix construction before that row

#### Scenario: Later better box does not replace the first owner occurrence
- **WHEN** a later duplicate row has better geometry than an earlier usable accepted row for the same owner
- **THEN** the primary cleaned treatment retains the earlier row and records the later row as excluded rather than silently substituting it

### Requirement: Duplication-burst credit is normalized and effective
The system SHALL allocate one total unit of mechanism credit per duplication burst and SHALL then normalize by image. The final event weight SHALL affect both local pairwise and complete-row imitation losses.

#### Scenario: One-row burst receives one unit
- **WHEN** a burst contains one duplicate comparison
- **THEN** that comparison receives the burst's full mechanism credit before image normalization

#### Scenario: Long burst does not dominate
- **WHEN** a burst contains an entry comparison and multiple later duplicate-prefix comparisons
- **THEN** the entry receives one half of burst credit, the remaining comparisons share the other half, and all weights for the burst sum to one before image normalization

#### Scenario: Logged weight changes gradient magnitude
- **WHEN** two otherwise identical events have different admitted event weights
- **THEN** their weighted local or imitation loss contributions and gradients differ by the declared ratio

### Requirement: Training profiles and controls have matched dose
The system SHALL support local duplicate rejection and recovery, duplicate-cleaned imitation, and their combined profile while allowing Source-preservation-only and recovery-positive-only controls through existing complete-row primitives. Matched banks SHALL keep image allocation, optimizer updates, aggregate mechanism credit, seeds, and Source-preservation dose fixed.

#### Scenario: Combined treatment does not increase total mechanism dose
- **WHEN** the combined bank contains both local and cleaned event families
- **THEN** its total mechanism credit and optimizer update count equal those of each single-treatment bank

#### Scenario: Duplicate-negative contribution is identifiable
- **WHEN** local duplicate rejection is compared with recovery-positive-only
- **THEN** both arms use the same replay prefixes, recovery rows, Source-preservation dose, and optimization budget, differing only by the admitted duplicate-negative term

#### Scenario: Historical profiles remain isolated
- **WHEN** an existing rollout-calibration profile and historical StateBank are loaded
- **THEN** the new event flags, evidence object, and loss terms remain inactive and historical validation and loss behavior are unchanged

### Requirement: Multi-family planned steps contain every enabled family
The system SHALL schedule every multi-family profile so every planned optimizer step has a positive global denominator for each enabled loss family across all ranks.

#### Scenario: Family-stratified planned step is valid
- **WHEN** a combined bank is planned for replicated distributed training
- **THEN** every planned step contains local duplicate-rejection and cleaned-imitation eligibility across the global rank window and records both denominators

#### Scenario: Local duplicate-rejection planned step is valid
- **WHEN** a local duplicate-rejection and recovery bank is planned for replicated distributed training
- **THEN** every planned step contains both Source-preservation and local duplicate-rejection eligibility across the global rank window and records a positive denominator for every enabled term

#### Scenario: Incomplete multi-family window fails before forward
- **WHEN** a planned multi-family window lacks an enabled family globally
- **THEN** the system fails before model forward with the missing family and planned-step identity

### Requirement: Training artifacts expose physical-owner treatment semantics
The system SHALL record the source and replay provenance, owner-review exclusions, burst allocation, per-family dose, losses, margins, and terminal failures needed to audit the duplication treatment.

#### Scenario: Run receipt is complete
- **WHEN** a duplicate training run starts
- **THEN** its artifacts identify source checkpoint and diff, reviewed ledger, raw and cleaned trajectory hashes, image and burst counts, burst-length histogram, exclusions, profile, optimizer dose, per-family credit, and configured decode semantics

#### Scenario: Metrics do not rename unmatched predictions as hallucinations
- **WHEN** official unmatched predictions are counted in evaluation artifacts
- **THEN** the artifacts report them as unresolved unmatched predictions unless a separate review establishes semantic error or entity hallucination

#### Scenario: Checkpoint returns to ordinary inference
- **WHEN** a trained checkpoint is loaded for evaluation
- **THEN** ordinary greedy inference requires no StateBank, duplicate controller, or custom decoding component

