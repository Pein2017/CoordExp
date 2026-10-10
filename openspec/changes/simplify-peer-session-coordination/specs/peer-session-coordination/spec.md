## Purpose

Enable concise coordination among explicitly selected independent Pi Web sessions while preserving identity checks, delivery evidence, and the distinction between peer information and execution authority.

## ADDED Requirements

### Requirement: Collaboration guidance separates authority from transport
The maintained lead-worker guidance SHALL bind the worker's outcome, owned surfaces, relevant dependencies, report destination, permitted continuation and escalation boundary. It SHALL allow direct factual dependency coordination without granting peers scientific decision, execution expansion or acceptance authority. Shared research entrypoints SHALL remain at their existing owners rather than a new multiwriter board.

#### Scenario: Worker needs an interface fact
- **WHEN** an assigned worker needs a schema or input reference from a named peer inside its package
- **THEN** guidance permits direct factual communication and continuing unaffected work without a new lead approval stage

#### Scenario: Peer proposes a changed experiment
- **WHEN** a peer message would change the contrast, population, material cost or stop boundary
- **THEN** guidance requires escalation to the existing decision owner and does not treat message delivery as authorization

#### Scenario: Independent advisors compare conclusions
- **WHEN** multiple advisors are assigned independent scientific judgments
- **THEN** guidance requires recording their initial assessments before sharing peer conclusions

### Requirement: Selected peers resolve to exact session identities
The transport SHALL support one explicitly selected sender and one recipient for existing-session inspection or sending, including worker-to-worker communication. It SHALL support exact IDs with explicit working directories and task-local aliases from a caller-selected existing JSON state. Alias routing SHALL consume only a versioned transport binding section, resolve each alias once per invocation, and never infer targets from names, recency or scientific fields. Ambiguous routing combinations and invalid bindings SHALL fail before API mutation. The transport SHALL NOT rewrite participant state.

#### Scenario: Two workers in different worktrees communicate
- **WHEN** valid sender and recipient bindings identify distinct independent root sessions in different physical directories
- **THEN** inspection verifies both identities against their own directories and sending prompts only the selected recipient

#### Scenario: Selected alias is absent
- **WHEN** an alias is absent or bindings have duplicate keys, duplicate session identities or an unsupported version
- **THEN** the invocation fails without starting, loading or prompting any session

#### Scenario: Scientific state contains unrelated fields
- **WHEN** the selected state contains experiment metrics and a valid transport section
- **THEN** resolution does not interpret the scientific fields and leaves the state bytes unchanged

### Requirement: Recipient validation and legacy operation remain compatible
The transport SHALL retain the public script path and legacy pair/create/model-list commands, including separate lead/worker cwd support. Both selected peers SHALL remain distinct ordinary independent root sessions on the same declared Pi Web endpoint. It SHALL retain loopback-only endpoint validation, proxy bypass, redirect refusal, bounded responses and safe error output. Before prompting, it SHALL load the target only as needed and reject identity or model/effort drift. New alias/peer routing SHALL NOT create sessions or change their settings.

#### Scenario: Child or fork target is selected
- **WHEN** either selected session violates the independent-root relation or expected cwd
- **THEN** the operation refuses delivery rather than switching transport or choosing another session

#### Scenario: Target settings change on load
- **WHEN** loading an unloaded recipient reveals changed model or effort
- **THEN** no prompt is submitted and the receipt preserves the failed operation stage

#### Scenario: Existing worker creation command is used
- **WHEN** a legacy create command names provider, model and effort
- **THEN** catalog preflight and created-session verification remain required and the generated return command invokes the retained public entrypoint

### Requirement: Briefs can reference reports without expanding them
The transport SHALL preserve explicit brief input through inline text or a designated message file and SHALL support optional report-path, section, revision and supersession metadata. Report paths SHALL be literal absolute references, not implicit message inputs. The transport SHALL NOT open, hash or inject referenced report contents. Legacy message-file input SHALL continue to transmit the explicitly selected file body. Envelopes and receipts SHALL retain resolved sender/recipient identities, dispatch/reply correlation and report-reference metadata without presenting them as user authorization.

#### Scenario: Brief points to a detailed report
- **WHEN** a short brief references an absolute report path with revision and section
- **THEN** the transmitted content includes the brief and references but not the report body

#### Scenario: Referenced report is currently unavailable
- **WHEN** the path is syntactically valid but unavailable to the sender
- **THEN** the transport can carry the pointer without claiming the artifact exists or has been verified

#### Scenario: Correction supersedes only presentation
- **WHEN** a brief declares that a revision replaces a displayed table but not original keyed data
- **THEN** metadata preserves that distinction without deleting evidence or withdrawing earlier queued messages

### Requirement: Delivery remains explicit and is not passive notification
The transport SHALL default to queue delivery and retain explicit steering. Queue SHALL start an idle recipient or enqueue a follow-up for a busy recipient; steering SHALL use the existing processing-boundary behavior. Guidance SHALL distinguish routine updates, decision requests and urgent action corrections. The transport SHALL NOT derive execution permission or switch delivery mode from a brief label, and SHALL NOT claim read acknowledgment, exact-turn interruption or job cancellation.

#### Scenario: Routine result is queued
- **WHEN** a result brief uses the default mode
- **THEN** the receipt records queue delivery and guidance explains that an idle recipient can be activated

#### Scenario: Urgent correction is steered
- **WHEN** the caller explicitly selects steering to invalidate an ongoing action
- **THEN** the requested mode is recorded without claiming that external jobs stopped or the correction was consumed

### Requirement: Receipts preserve uncertainty and prevent implicit resubmission
A send SHALL create an exclusive durable receipt before the first mutating API call, record the invocation's resolved route and payload identities, and distinguish rejection, accepted-or-queued submission and unknown delivery. Existing receipt reuse SHALL block mutation. Creation and sending SHALL NOT automatically retry, refresh aliases to resend, or describe an HTTP acknowledgment as task completion or acceptance.

#### Scenario: Send response is lost
- **WHEN** the server may have received a prompt but the client times out
- **THEN** exactly one prompt attempt occurs and the receipt retains unknown delivery for reconciliation

#### Scenario: Receipt path already exists
- **WHEN** another invocation supplies an existing receipt path
- **THEN** the transport fails before loading or prompting the recipient

#### Scenario: Brief and report pointer are submitted
- **WHEN** a valid brief is sent with report metadata
- **THEN** the receipt binds the brief body and full transmitted envelope separately, including report-reference metadata in the latter

### Requirement: First-version communication remains single-recipient and task-local
The interface SHALL accept only one recipient per invocation and SHALL provide no whole-team selector, automatic fanout, shared editable board, passive inbox, scheduler or cross-harness identity bridge. Guidance SHALL direct ordinary progress into existing task/report owners and use notifications at useful checkpoints rather than demand acknowledgment for every phase.

#### Scenario: Caller requests broadcast
- **WHEN** a caller requests an all-team target or multiple recipients in the first-version interface
- **THEN** it is rejected rather than expanded into multiple session activations

#### Scenario: Routine progress is recorded
- **WHEN** a worker completes a routine phase within its package
- **THEN** guidance permits recording progress without a mandatory message, acknowledgment or pause
