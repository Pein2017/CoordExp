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

#### Scenario: Selected resize handle overlaps another object
- **WHEN** Select mode is active and pointer down is nearest to a selected bbox corner within 14 CSS pixels or edge midpoint within 12 CSS pixels while another bbox body overlaps that location
- **THEN** the editor starts resize on the selected bbox through that handle, captures the pointer, and does not change selection

#### Scenario: Pointer is outside selected resize handles
- **WHEN** Select mode is active and the operator clicks another bbox outside every selected-handle screen-space zone
- **THEN** ordinary bbox hit order selects that object without the selected bbox body intercepting the click

#### Scenario: Selected handle zones overlap each other
- **WHEN** zoom or a small bbox places multiple selected handles within their accepted screen-space distance
- **THEN** the nearest handle wins deterministically, an exact tie prefers a corner, and hover displays the matching resize cursor before the gesture begins

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

### Requirement: Fast mode switching and visible class context
The editor SHALL expose Select as mode 1 and Draw bbox as mode 2 through
Command+1/Command+2 when delivered by the browser and plain 1/2 outside text
entry controls as a reliable fallback. The active canonical class SHALL remain
visible and sticky across image navigation, SHALL follow selection of an
existing object, and SHALL clear on train/validation split change.

#### Scenario: Operator switches modes by keyboard
- **WHEN** the editor is writable and the operator invokes mode 1 or mode 2 outside an input, textarea, or select control
- **THEN** the matching mode becomes visibly active without mutating the Draft

#### Scenario: Draw mode has no active class
- **WHEN** the operator enters Draw mode before any canonical class is active
- **THEN** Draw mode remains selected, class search receives focus, and no bbox is persisted until a canonical class is chosen

#### Scenario: Existing object supplies class context
- **WHEN** the operator selects an existing object and later enters Draw mode
- **THEN** that object's canonical class is shown as the active Draw class and is used for the next valid created bbox

#### Scenario: Dataset split changes
- **WHEN** the operator switches between train and validation
- **THEN** the active Draw class is cleared while ordinary image-to-image navigation preserves it

### Requirement: Draw-mode pointer guides
While Draw mode is active, the editor SHALL render one horizontal and one
vertical presentation-only guide through the current pointer position inside
the natural image. The guides SHALL remain visible during bbox drag and SHALL
respect zoom, pan, and image bounds.

#### Scenario: Pointer enters the drawable image
- **WHEN** Draw mode is active and the pointer is inside the natural image
- **THEN** bounded horizontal and vertical guides intersect at the pointer without changing Draft state

#### Scenario: Pointer leaves or mode changes
- **WHEN** the pointer leaves the natural image, the task is disabled, or the operator exits Draw mode
- **THEN** both guides disappear without emitting an edit gesture

### Requirement: Training-order object inventory
The right panel SHALL list every current object by canonical class and a
transient `#1..#N` ordinal in the exact training top-left order: ascending
`(y1, x1)`, with exact-anchor ties preserving prior rank before new-object
creation order. Inventory ordinals SHALL NOT be persisted as object identity.

#### Scenario: Current objects are shown
- **WHEN** a committed task or Draft is rendered
- **THEN** the right panel shows all objects once in training order and identifies each as `#<ordinal> <canonical class>`

#### Scenario: Geometry or membership changes
- **WHEN** create, move, resize, delete, Undo, reload, or ROI insertion completes authoritatively
- **THEN** the inventory and its transient ordinals are recomputed immediately from the resulting object list

#### Scenario: Object is actively dragged
- **WHEN** a move or resize pointer gesture is still in progress
- **THEN** the inventory keeps its prior order until the completed canonical result arrives

#### Scenario: Selection crosses canvas and inventory
- **WHEN** an object is selected in either the SVG or the right-panel inventory
- **THEN** the same stable region is highlighted in both surfaces and the inventory item is scrolled into view

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

### Requirement: Focus Queue navigation and status
When an active Focus Queue exists, the editor SHALL expose a Focus mode whose
task list and Next/Previous controls follow exact queue order. It SHALL show
queue position, pending queue Draft count, Focus Commit progress, training
publication progress/failure, and an explicit path back to full-dataset view.

#### Scenario: Focus task is opened
- **WHEN** the operator enters Focus mode with an active queue
- **THEN** the editor opens an ordered queue member and shows its `current/total` position without changing task or object identity

#### Scenario: Focus navigation reaches a boundary
- **WHEN** Previous is requested on the first member or Next on the last member
- **THEN** navigation remains within the queue and does not wrap or fall through to the full split

#### Scenario: Focus Commit is running
- **WHEN** a scoped batch or its automatic publisher is queued, running, reconciling, succeeded, or failed
- **THEN** the UI distinguishes Draft save, annotation Commit, and training-pair publication states while ordinary editing remains responsive

#### Scenario: Full view is selected
- **WHEN** the operator leaves Focus mode without releasing the queue
- **THEN** ordinary full-index navigation resumes and the durable queue remains available for later return
