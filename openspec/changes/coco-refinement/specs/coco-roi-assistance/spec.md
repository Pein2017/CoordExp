## ADDED Requirements

### Requirement: Temporary single ROI interaction
The editor SHALL provide one temporary ROI rectangle independent of annotation
objects. Drawing a new ROI SHALL replace the prior temporary ROI and SHALL NOT
save it as a COCO bbox.

#### Scenario: ROI is drawn
- **WHEN** the operator enters AI Region mode and drags a valid region
- **THEN** one temporary ROI appears without changing the Draft object list or Undo history

#### Scenario: ROI is redrawn
- **WHEN** a temporary ROI already exists and the operator draws another
- **THEN** only the temporary ROI is replaced and all annotation objects remain unchanged

### Requirement: Profile-bound factor-valid resolution
ROI controls SHALL default to width `1024` and height `1024`, allow independent
width and height values, and require the selected profile's processor factor,
axis bounds, total-pixel bound, immutable artifact/config fingerprints, and
generation deadline.

#### Scenario: Valid rectangular resolution is selected
- **WHEN** width and height are factor-aligned and within the active profile bounds
- **THEN** the exact rectangular canvas and immutable profile fingerprint are bound into the request and receipt

#### Scenario: Invalid resolution is entered
- **WHEN** either axis violates factor/bounds or total pixels exceed the profile limit
- **THEN** inference remains disabled and no request is sent

### Requirement: Target-bound single-flight inference
Before inference the client SHALL durably flush the current Draft. The service
SHALL freeze the exact project/task/image/Draft revision, working generation,
profile, ROI, transform, and operator principal. At most one request SHALL run
at a time, and current-task navigation plus ROI controls SHALL remain locked
until a terminal response.

#### Scenario: Inference starts
- **WHEN** the operator has a valid ROI, resolution, profile, and durable current Draft and presses Infer
- **THEN** the service records one immutable target/request receipt and starts one resident-model execution

#### Scenario: Second request is attempted
- **WHEN** another ROI request is active
- **THEN** the service rejects the second request without changing either Draft or receipt authority

### Requirement: Reversible ROI transform and strict result validation
The service SHALL reuse the approved half-open crop, realized resize,
letterbox/padding, inverse mapping, clipping, norm1000 quantization, current
prompt/parser/runtime, and official COCO-80 validation contracts. It SHALL
retain replayable raw and per-result receipts.

#### Scenario: ROI aspect differs from canvas
- **WHEN** a nonsquare ROI is inferred on a differently shaped canvas
- **THEN** every accepted canvas bbox is mapped back through the recorded scale and padding to a strict clipped original-image norm1000 bbox

#### Scenario: Result is invalid
- **WHEN** parsing, class validation, geometry, target binding, or transform replay rejects a result
- **THEN** the receipt records the rejection and the invalid result is never inserted

### Requirement: Atomic server-side Draft insertion
After successful inference the service SHALL recheck the frozen Draft target
and append all valid results in one SQLite transaction. It SHALL return the new
Draft revision and authoritative objects; the browser SHALL treat the response
as one Undo action. Inferred objects SHALL have the same edit/delete/Commit
authority as human objects.

#### Scenario: Target remains current
- **WHEN** the saved Draft revision and task epoch still equal the frozen target and one or more results are valid
- **THEN** one transaction appends every valid native region with stable key and complete inference provenance, increments the revision once, and returns the authoritative Draft

#### Scenario: Target changed during inference
- **WHEN** the task or Draft revision no longer equals the frozen target
- **THEN** the service inserts nothing, preserves both current Draft and inference receipt, and reports a target-mismatch outcome

#### Scenario: Operator edits an inserted box
- **WHEN** a returned inference-origin bbox is moved, relabeled, or deleted
- **THEN** the editor persists the change through the ordinary native Draft path without an accept/copy operation

### Requirement: Explicit terminal outcomes without partial hidden mutation
The ROI service SHALL distinguish accepted, accepted-with-drops, empty,
all-rejected, malformed/unsupported, cancelled, timeout, runtime/profile, and
target-mismatch outcomes. No failure outcome SHALL partially mutate a Draft.

#### Scenario: Model returns no objects
- **WHEN** inference completes successfully with an empty valid result set
- **THEN** no Draft mutation occurs and the UI reports a completed-empty outcome

#### Scenario: Runtime fails
- **WHEN** loading, generation, cancellation, timeout, parsing envelope, or profile validation fails
- **THEN** no Draft mutation occurs, the temporary ROI follows the specified retain/clear outcome, and a credential-safe failure receipt remains inspectable

#### Scenario: Response is lost after insertion
- **WHEN** SQLite committed an insertion but the HTTP response is lost
- **THEN** request status/retry resolves the same receipt and Draft revision without duplicating regions

### Requirement: Inference review presentation
Uncommitted inference-origin objects SHALL use deterministic high-contrast
neighbor coloring, palette-exhaustion badges, and same-class overlap advisory
hints while retaining canonical class text. Presentation SHALL NOT suppress,
merge, replace, reorder, or change Commit semantics.

#### Scenario: Nearby inferred objects are inserted
- **WHEN** two or more uncommitted inference-origin objects occupy the same visual neighborhood
- **THEN** the editor assigns deterministic distinguishable colors and badges while every object remains independently selectable and deletable

#### Scenario: Potential duplicate is detected
- **WHEN** same-class boxes satisfy the approved overlap advisory threshold
- **THEN** the editor highlights the potential duplicate but Commit remains permitted and no box is removed automatically

### Requirement: Real-profile acceptance gate
The replacement SHALL NOT be declared complete until one accepted configured
profile executes through the current model/prompt/parser runtime and one
inserted ROI result is edited or retained, batch committed, materialized, and
loaded by the current CoordExp-Swift data path.

#### Scenario: End-to-end ROI acceptance succeeds
- **WHEN** the operator completes the accepted real-profile smoke
- **THEN** receipts bind the input canvas, profile, transform, raw result, mapped objects, Draft revision, batch member, terminal working row, materialized coord row, and unchanged source/image identities
