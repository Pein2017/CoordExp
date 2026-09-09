## MODIFIED Requirements

### Requirement: Smallest sufficient capability profile

The infra-base SHALL let each producer select only the mechanics its execution needs. Ordinary exploratory execution MUST be possible without a journal, admission dossier, whole-worktree cleanliness gate, mandatory per-source hashes or chained receipts. Basic run context SHALL identify configuration, code revision with dirty status, input/model locations and output location, without claiming dirty status alone makes execution exactly replayable.

An immutable one-shot producer SHALL be able to select strict validation and exclusive publication alone. Durable continuation and mechanics-only launch admission SHALL remain explicit selections under their existing contracts. Selecting a capability MUST NOT silently enable another, assign retry policy or change scientific output meaning. Simplifying defaults MUST NOT weaken a selected strict contract or rewrite historical receipts.

#### Scenario: Ordinary experiment has unrelated local edits

- **WHEN** a new exploratory producer has valid declared inputs and unrelated modified or untracked files, without selecting strict admission
- **THEN** its CPU/offline execution is not rejected merely for whole-tree dirtiness, and the recorded context states the dirty status

#### Scenario: One-shot producer chooses the leaf profile

- **WHEN** a probe has one immutable strict-JSON result and no continuation or admission requirement
- **THEN** it can validate and publish exclusively without a journal, admission root, phase plan or generated runner

#### Scenario: Resumable producer adds only a journal

- **WHEN** one producer needs durable progress across attempts
- **THEN** it can select the existing journal while preserving work-item meaning, exact continuation checks, retry decisions, scheduling and terminal interpretation at their existing owners

#### Scenario: Custom differentiable runtime does not fit inference

- **WHEN** a probe needs gradient-preserving native model execution
- **THEN** it can use appropriate shared native operations without being wrapped as inference-only execution or inheriting an optimizer, packed-training structure or scientific objective

### Requirement: Canonical public mechanics are discoverable

Supported common operations SHALL have documented public import paths at their concept owner. The guide SHALL identify their inputs, caller-owned scientific choices, failure behavior and cheapest acceptance check. New retained probe code MUST NOT need sibling worktree imports, private backend fields or copied implementations to use a supported operation. Retained callers SHALL migrate to one implementation rather than a parallel execution framework.

The guide SHALL link the canonical lifecycle and selected artifact, inference and research-operation contracts rather than duplicate them. Existing deterministic scored inference, packed execution and snapshot export SHALL preserve their respective supported behavior when consuming lower-level operations.

#### Scenario: New one-shot probe finds the publication owner

- **WHEN** a direction needs to publish a strict immutable result
- **THEN** the guide identifies the public publication operation, occupied-path failure and caller-owned schema without requiring admission

#### Scenario: Producer finds assignment mechanics

- **WHEN** a direction needs global annotated-owner assignment
- **THEN** the guide identifies its public operation and matching policy rather than directing the caller to a visualization helper or another experiment script

#### Scenario: Existing deep owner remains authoritative

- **WHEN** shared operations are extracted beneath the deterministic HF adapter
- **THEN** strict HF history/evidence behavior still uses its existing adapter contract while native research no longer needs that adapter's private model or native-input fields

### Requirement: New shared layers require live cross-direction evidence

A proposed shared abstraction SHALL identify actual retained runnable consumers and duplicated caller knowledge it removes. Compatible observed consumers SHALL justify the scope of generalization; they need not be executing concurrently. Historical similarity alone MUST NOT justify a new framework, and a first exploratory consumer SHALL keep unproven variation local.

The design SHALL compare direct composition of existing owners, preserve explicit scientific controls and required topology, and name a bounded counterexample that would falsify the common behavior. Sharing model or tensor operations MUST NOT imply a scheduler, phase language, optimizer orchestrator, plugin registry or global run context.

#### Scenario: All experiments are stopped for refactoring

- **WHEN** retained producers demonstrate the same operation but no job is running
- **THEN** characterized caller behavior can justify a common owner without launching experiments to satisfy a live-consumer label

#### Scenario: Only archival similarity exists

- **WHEN** only retired implementations resemble one another and no retained caller needs the operation
- **THEN** their sources are preserved for recovery rather than repackaged into a new abstraction

#### Scenario: One live direction plus one retired lineage

- **WHEN** one retained direction needs an operation and a materially different retired lineage supplies the other example
- **THEN** historical resemblance alone does not justify generalizing their incompatible behavior

#### Scenario: A later direction proves an identical seam

- **WHEN** another retained runnable consumer demonstrates compatible behavior and direct composition is insufficient
- **THEN** the smallest shared operation can be proposed with consumer tests and explicit scientific controls

### Requirement: Integration never performs probe lifecycle actions

Infra-base operations and their acceptance checks SHALL NOT create, move, unlock, retire, merge, tag or delete worktrees or branches. Lifecycle actions SHALL remain explicit work under the canonical policy and accepted change scope, independent of runtime capability validation.

#### Scenario: Infra-base validation runs in the integration lane

- **WHEN** a capability is verified in the research base or an authorized temporary development worktree
- **THEN** its validation leaves worktrees, branches, tags, external artifacts and running processes unchanged

## ADDED Requirements

### Requirement: Native exact replay preserves conditioning and gradients

Native research execution SHALL accept exact token histories and declared multimodal inputs without requiring a packed-training sequence, state-bank event or HF evidence-session identity. It SHALL preserve literal conditioning/action IDs, image alignment and causal target ordering. Stale derived position/cache state SHALL be rejected or regenerated according to the operation's declared behavior.

Differentiable replay MUST NOT implicitly disable gradients, detach selected scores or transfer them to CPU. Model lifetime, device placement and train/eval mode SHALL remain explicit caller responsibilities. Sharing native operations MUST NOT change the strict inference adapter's private-history/evidence contract.

#### Scenario: Trainable action is replayed literally

- **WHEN** a caller requests scores for an exact action following a multimodal prefix
- **THEN** target rows align causally with those action IDs and gradients reach the caller-selected parameters without retokenizing the stored action or constructing packing metadata

#### Scenario: Identical positions cross gradient modes

- **WHEN** native position construction is used in differentiable replay
- **THEN** both values and tensor behavior support that forward/backward path, rather than passing only an integer-value comparison against inference-mode positions

### Requirement: Native continuation preserves request budgets and optional traces

Native research continuation SHALL preserve exact histories, stable request/result association and per-request remaining-token budgets in heterogeneous batches. Terminal actions and exhausted budgets SHALL produce terminal results without model work for those requests. Trace-free execution MUST NOT request vocabulary-score, hidden-state or attention traces that the caller did not request.

The operation SHALL preserve the resolved generation policy, including sampling controls, repetition processing and raw-model versus processed-policy likelihood meaning. Fixed-seed/fixed-batch behavior SHALL be preserved for migrated sampled callers; this contract does not promise invariance to changing batch size or order. Existing scored deterministic inference MUST NOT silently become trace-free or sampled.

#### Scenario: Mixed exact histories and remaining budgets

- **WHEN** a batch contains different history lengths and continuation budgets
- **THEN** padding does not become generated content, result identity remains attached to its request, and each returned suffix respects its own budget

#### Scenario: Terminal and trace-free requests

- **WHEN** one request ends with the terminal action and another requests continuation with traces disabled
- **THEN** the terminal request incurs no continuation forward and the other request returns its continuation without retaining unrequested score/hidden-state payloads

#### Scenario: DORA raw-softmax policy differs from a baseline

- **WHEN** migrated callers declare different sampling transformations, including top-k behavior
- **THEN** each retains its resolved policy and probability interpretation instead of inheriting one shared sampling default

### Requirement: Shared token scoring and parameter groups preserve scientific control

Shared token scoring SHALL return differentiable per-token values from explicitly aligned logits and target IDs without an implicit reduction, credit, mask, distributed factor or device transfer. Parameter selection/group construction SHALL preserve exact selected names, order and coverage without requiring unused optimizer settings for frozen parameter categories.

Objective reduction, freezing/enabling parameters, expected scientific counts, optimizer choice and distributed synchronization SHALL remain caller-controlled. Existing raw optimizer versus prepared execution-handle distinctions and persisted optimizer continuation checks MUST remain valid.

#### Scenario: Coordinate and full-action objectives share scores

- **WHEN** two objectives use the same token scores but different selected positions and denominators
- **THEN** sharing scoring leaves their losses and gradients equivalent to their separately specified formulas rather than replacing sum with mean or adding a hidden world-size factor

#### Scenario: Frozen token embeddings need no dummy optimizer settings

- **WHEN** a research caller selects only adapter parameters and token embeddings are frozen
- **THEN** it can construct exact optimizer groups without supplying irrelevant token-embedding optimizer configuration, while duplicate or unmatched trainable parameters remain errors

### Requirement: Assignment and layer capture retain distinct semantics

Public global annotated-owner assignment SHALL preserve the existing category constraint, cardinality-first objective, quantized-IoU secondary objective and deterministic tie behavior. Threshold, category normalization, dedup ordering and metric denominator SHALL remain explicit caller choices. It MUST NOT replace a retained greedy visualization or official COCO evaluation algorithm.

Named layer capture SHALL distinguish pre-injection output from post-injection input and preserve selected snapshots across in-place mutation. Hook cleanup SHALL occur after success or failure; intervention formulas SHALL remain direction-owned.

#### Scenario: Greedy and global assignment disagree

- **WHEN** the existing two-owner counterexample is evaluated
- **THEN** global assignment preserves the two feasible owners and greedy visualization retains its independently specified behavior

#### Scenario: DeepStack mutates a captured output

- **WHEN** a selected pre-injection layer output is subsequently modified in place
- **THEN** the captured pre-injection snapshot remains unchanged, and an exception does not leave its hook installed
