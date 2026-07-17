## ADDED Requirements

### Requirement: Lightweight editable task view
The browser SHALL render the active task image and every bbox in a lightweight
SVG editing view without a Label Studio frontend or Node build dependency. The
view SHALL preserve natural-image aspect ratio and derive displayed geometry
from canonical norm1000 objects.

#### Scenario: Task is opened
- **WHEN** the operator opens any indexed train or validation task
- **THEN** the image and authoritative committed-or-Draft objects render as editable axis-aligned rectangles with canonical labels and stable identities

#### Scenario: Task contains many objects
- **WHEN** a dense task is opened
- **THEN** every object remains individually selectable, visible, and addressable without changing its saved order or semantics

### Requirement: COCO-80 bbox CRUD
The editor SHALL create, select, move, resize, relabel, and delete axis-aligned
bboxes. Every saved object SHALL use strict non-degenerate norm1000 `xyxy`, one
canonical English COCO-80 name, its official sparse category ID, and zero
rotation.

#### Scenario: Bbox is drawn
- **WHEN** the operator drags a non-degenerate rectangle and selects a class
- **THEN** the client submits natural-image geometry and the server returns one canonical norm1000 object with a new stable region key

#### Scenario: Existing bbox is edited
- **WHEN** the operator moves, resizes, or relabels a selected object
- **THEN** the next durable Draft retains the same stable region key and any valid committed identity while updating only the requested semantics

#### Scenario: Bbox is deleted
- **WHEN** the operator deletes a selected object
- **THEN** the next durable Draft omits that object and later Commit records its deletion without renumbering unrelated objects

#### Scenario: Invalid geometry is submitted
- **WHEN** a bbox is rotated, degenerate, non-finite, out of range after clipping, or bound to an unsupported class
- **THEN** the server rejects the whole Draft save and preserves the prior revision

### Requirement: Fast canonical class selection
The editor SHALL accept only the official English COCO-80 names while offering
keyboard-operated exact, prefix, substring, and spelling-tolerant filtering
over that fixed registry. Aliases, translations, synonyms, and free-form names
SHALL NOT be persisted.

#### Scenario: Operator types a partial name
- **WHEN** the input matches one or more canonical names by the approved filter
- **THEN** the editor presents ranked canonical options and saves only the selected canonical name and sparse ID

#### Scenario: Operator types an unknown name
- **WHEN** no canonical option is selected
- **THEN** no object class mutation is persisted

### Requirement: Discrete autosave and navigation flush
The editor SHALL save after semantic gesture completion and SHALL await only
the active Draft save before in-app task navigation. It SHALL NOT wait for an
active batch worker. Pending Drafts SHALL remain visible and shall not force a
Commit on every task.

#### Scenario: Edit gesture completes
- **WHEN** create, move, resize, relabel, delete, Undo, or ROI insertion reaches a semantic boundary
- **THEN** the editor sends one idempotent full-Draft save against the current revision and displays Saving until the authoritative response arrives

#### Scenario: Operator selects Next
- **WHEN** the current task has an in-memory edit or save in flight
- **THEN** navigation awaits that Draft response, reminds the operator that pending Drafts exist, and then opens the next task without starting or waiting for Commit

#### Scenario: Save fails
- **WHEN** validation, transport, or CAS save fails
- **THEN** navigation remains on the task, the unsaved state stays visible, and retry/reload choices do not claim that the edit is durable

#### Scenario: Tab closes with no local edit
- **WHEN** every gesture has a durable Draft response even though pending Drafts are not committed
- **THEN** the browser does not warn solely because dataset Commit has not occurred

### Requirement: Undo preserves server authority
The editor SHALL group each discrete human gesture and each successful ROI
insertion into one local Undo step. Undo SHALL produce an ordinary new Draft
save and SHALL NOT mutate prior SQLite revisions or working generations.

#### Scenario: Operator undoes a deletion
- **WHEN** the last local action deleted one object
- **THEN** Undo restores that object in local state and persists the restored full object list as the next Draft revision

#### Scenario: Page reloads
- **WHEN** the browser reloads after a durable save
- **THEN** authoritative objects and revision recover even though prior in-memory Undo history is not required to survive reload

### Requirement: Dense-scene visibility controls
The editor SHALL provide show-all, dim-non-selected, hide-non-selected,
per-region visibility, restore, and focused deletion recovery. Visibility,
selection, zoom, pan, color, and badges SHALL remain presentation-only and
SHALL NOT alter Draft hashes or exported objects.

#### Scenario: Other objects are hidden
- **WHEN** the operator selects hide-non-selected
- **THEN** only the focused object remains fully visible while the saved Draft and pending state remain unchanged

#### Scenario: Hidden selected object is deleted
- **WHEN** the selected object is removed while other objects are hidden
- **THEN** the editor restores a usable visible state and permits selecting another object without semantic side effects

### Requirement: Clear task and batch status
The UI SHALL distinguish local save state, per-task committed-versus-Draft
semantics, pending-Draft count, and asynchronous batch state. Durable enqueue
SHALL NOT be displayed as committed dataset success.

#### Scenario: Batch is queued
- **WHEN** Commit returns a durable enqueue receipt
- **THEN** the UI shows Queued with batch identity/member count while included tasks remain distinguishable from terminally committed tasks

#### Scenario: Newer Draft follows a successful batch
- **WHEN** a captured task was edited after enqueue and its batch succeeds
- **THEN** the UI shows the committed generation plus that task's newer pending Draft without replacing it
