## MODIFIED Requirements

### Requirement: Configuration-first local training

The system SHALL expose training through `python -m src.train --config <path>`
and retain configuration support for single-node launches with one through four
GPU processes. It SHALL additionally provide an explicit eight-process smoke
configuration. General training configurations SHALL NOT require eight ranks.
Each launch SHALL validate the effective batch size against its world size.

#### Scenario: A local training launch is configured

- **WHEN** an operator loads a maintained one-through-four-process configuration
- **THEN** the launch does not acquire an eight-rank requirement

#### Scenario: An eight-process smoke is configured

- **WHEN** the eight-process production smoke is resolved
- **THEN** its batch arithmetic is valid for eight ranks and its finite schedule includes evaluation and checkpoint publication

### Requirement: Bounded exact resume

Exact resume SHALL be opt-in, admitted only at an optimizer boundary, require
the same world size, and fail closed when identities or runtime state do not
match. Checkpoints produced by the new implementation SHALL support this
contract. Migration of pre-change optimizer, scheduler, RNG, or cursor state
SHALL NOT be a compatibility obligation of this change.

#### Scenario: An exact continuation is requested

- **WHEN** a run requests exact resume from a nonterminal completed optimizer boundary produced by the new implementation
- **THEN** restoration proceeds only when checkpoint identity, runtime state, and world size match, and its continuation matches the uninterrupted execution under the declared comparison contract

#### Scenario: Incompatible state is presented for resume

- **WHEN** a requested resume state fails the current admission contract
- **THEN** the launch rejects it rather than silently migrating or partially restoring training state

#### Scenario: Strict continuation starts in a fresh distributed process

- **WHEN** a strict same-world-size continuation restores a new-implementation checkpoint
- **THEN** its admitted gradient reduction policy matches its parent and its optimizer updates and restored state remain exact despite starting in a fresh process

### Requirement: Separate checkpoint consumers

An inference payload SHALL be independently validated and its available
authentication evidence SHALL be checked. Exact-resume training state SHALL
remain separate and SHALL NOT be inferred from an inference payload. Supported
pre-change base-model, DoRA adapter, and special-token embedding-delta inputs
SHALL remain loadable for inference through their existing configuration
routes, including independently supplied components without a root publication
manifest. New publications SHALL retain their inference authentication contract.

#### Scenario: A checkpoint is consumed for inference

- **WHEN** inference loads a checkpoint payload
- **THEN** it validates the inference payload independently and does not consume optimizer, scheduler, RNG, or cursor state as an inference prerequisite

#### Scenario: Supported pre-change components have no root manifest

- **WHEN** an operator supplies supported pre-change DoRA or embedding-delta components through their existing configuration fields without a root publication manifest
- **THEN** inference validates the component format and identities and loads the declared composition without requiring a training-state migration or a newly generated root manifest

#### Scenario: An authenticated inference payload has been modified

- **WHEN** payload bytes disagree with an available authentication manifest
- **THEN** inference rejects the mismatch

### Requirement: Qualified inference backends

The system SHALL expose dynamic HF and vLLM inference. Dynamic HF SHALL retain
its adapter-plus-embedding-delta semantics. Composed BF16 execution SHALL be
identified as a derived inference model and SHALL qualify its fixed composition
fixture against dynamic HF with exact prompt IDs, sequence length and
non-coordinate tokens, permitting at most one grid unit of difference at each
coordinate-token position. Tied rows, merged target weights, stored dtype and
component identities SHALL retain exact authentication. Numeric/logprob drift
SHALL remain explicit diagnostics and SHALL NOT be labeled exact HF parity.
The complete current matching qualification set SHALL be required before
production vLLM use.

#### Scenario: A composed model is requested through vLLM

- **WHEN** a vLLM configuration resolves a composed execution model
- **THEN** production execution is denied unless the complete matching qualification set has been admitted

#### Scenario: BF16 composition changes a coordinate by one grid unit

- **WHEN** dynamic and merged HF fixture continuations have equal length and exact non-coordinate tokens, and every differing coordinate is within one grid unit under the authenticated canonical coordinate mapping
- **THEN** composition may qualify under the explicit bounded contract while preserving exact structural checks and reporting the original numeric and token differences

#### Scenario: Composition drift exceeds the accepted contract

- **WHEN** a fixture changes a non-coordinate token, EOS, sequence length, more than one grid unit at a coordinate, or supplies invalid/missing coordinate-policy evidence
- **THEN** qualification rejects the composition without broadening the allowance

#### Scenario: vLLM likelihoods are consumed or replayed

- **WHEN** the composed vLLM model produces policy likelihoods or forced raw-logprob replay
- **THEN** likelihoods are attributed to that merged execution model and replay retains exact request, prompt, continuation and stop alignment without claiming dynamic-HF probability parity

### Requirement: Evidence-scoped acceptance

Static interface evidence SHALL be distinguished from runtime evidence. A
runtime topology claim SHALL cite a matching executed witness. Acceptance of
this change SHALL include a current eight-GPU Qwen3-VL 2B plus COCO few-step
vertical witness covering successful optimizer updates, evaluation, checkpoint
publication, fresh exact resume, inference consumption, and direct evaluation.
Smoke success SHALL NOT imply model-quality gains, full-training stability,
untested topologies, or a measured performance improvement. Composed vLLM
consumption SHALL remain subject to the matching qualification gate, including
the explicitly bounded BF16 composition contract.

#### Scenario: A two-GPU runtime claim is reported

- **WHEN** the system is described as having executed on two GPUs
- **THEN** the claim cites a matching Qwen3-VL 2B plus COCO vertical witness and does not imply untested topologies or model quality

#### Scenario: An eight-GPU runtime claim is reported

- **WHEN** the change is reported as accepted on eight GPUs
- **THEN** the report links the completed matching vertical witness and names the executed model, workload, update counts, and consumer scope

#### Scenario: A runtime or consumer gate has not completed

- **WHEN** a required runtime or inference-consumer check fails or remains unexecuted
- **THEN** that gate remains incomplete and interface tests are not substituted for its runtime evidence

## ADDED Requirements

### Requirement: Bounded preprocessing submissions

Parallel preprocessing SHALL bound submitted, unfinished work by a finite
worker-proportional limit independent of dataset cardinality. It SHALL preserve
canonical example order, exactly-once result membership, encoded payloads,
and error propagation. This contract SHALL NOT imply that all dataset or cache
materialization is streaming.

#### Scenario: A dataset exceeds the submission window

- **WHEN** preprocessing a dataset larger than the worker-proportional window
- **THEN** unfinished submissions remain within that window and new work is admitted as earlier work completes

#### Scenario: Workers finish out of order

- **WHEN** parallel workers complete examples in a different order from the source
- **THEN** the returned examples and downstream packed contents retain the canonical source order

#### Scenario: A preprocessing worker fails

- **WHEN** a worker reports a preprocessing error
- **THEN** preparation fails without publishing a successful reusable cache or silently changing the execution route

### Requirement: Production semantic preservation

Execution-only optimizations SHALL preserve the frozen workload's dataset
membership and order, template and geometry, tokens and supervision positions,
pack and optimizer-step membership, objective and denominators, effective batch
size, optimizer and schedule, and declared precision. Cache admission SHALL
continue to bind semantic content determinants and reject stale or corrupt
state. A scheduling-only implementation change SHALL NOT invent a new semantic
cache determinant.

#### Scenario: The scheduling optimization processes the frozen workload

- **WHEN** baseline and candidate prepare the same workload
- **THEN** encoded and packed payload identities agree and the production optimization retains the declared training semantics

### Requirement: Comparable performance evidence

Performance claims SHALL use comparable fixed workloads and distinguish cold
preparation, warm cache reuse, training, and complete lifecycle observations.
A speedup claim SHALL require repeated paired observations exceeding observed
variation while satisfying semantic and resource constraints. A correctness or
maintainability improvement MAY be accepted without a speedup claim.

#### Scenario: Timings are insufficient to distinguish a speedup

- **WHEN** measurements are unpaired, incomparable, or within observed variation
- **THEN** the result is reported as inconclusive for speedup while any separately verified boundedness or correctness result retains its own evidence scope
