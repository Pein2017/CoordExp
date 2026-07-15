## ADDED Requirements

### Requirement: Bbox-only COCO-80 annotation
The editor SHALL permit only axis-aligned rectangle annotations labeled with one
of the official 80 canonical English COCO class names and SHALL not expose
polygon, mask, rotated-box, crowd, alias, translation, or free-form class
creation in this project.

#### Scenario: Reviewer draws a new object
- **WHEN** the reviewer creates an annotation region
- **THEN** the editor creates an axis-aligned bbox and requires one canonical COCO-80 class before Commit

#### Scenario: Unknown label is entered
- **WHEN** the reviewer types a value outside the canonical registry
- **THEN** the editor offers canonical search results but cannot save that value as a class

#### Scenario: Rotation is attempted
- **WHEN** the fixed bbox project renders region controls or receives a nonzero-rotation Commit payload
- **THEN** no rotation handle/field is exposed and the parent validator rejects nonzero rotation

### Requirement: Editable active annotations
Every object in an opened task's one authoritative annotation SHALL remain
directly selectable, movable, resizable, relabelable, and deletable until
Commit, with no candidate/prediction-copy layer or alternate annotation target.

#### Scenario: Existing source object is corrected
- **WHEN** the reviewer selects a source bbox
- **THEN** its geometry and canonical class can be changed without copying it into a separate candidate layer

#### Scenario: Inferred object is corrected
- **WHEN** the reviewer selects a newly inserted ROI result
- **THEN** it supports the same edit and delete operations as a human-created bbox

### Requirement: Canonical spelling-tolerant class search
The editor SHALL rank canonical class names by exact match, prefix/substring
match, and spelling similarity after case/whitespace normalization, and every
selectable result SHALL be an exact member of the COCO-80 registry.

#### Scenario: Canonical name is entered
- **WHEN** the reviewer types `traffic light`
- **THEN** `traffic light` is ranked first and can be chosen by keyboard

#### Scenario: Canonical name is mistyped
- **WHEN** the reviewer enters a small spelling error such as `trafic light`
- **THEN** the canonical `traffic light` remains selectable without creating an alias

#### Scenario: Synonym or translated name is entered
- **WHEN** the reviewer enters a synonym, Chinese translation, or unrelated free text
- **THEN** the editor never stores that input and shows only canonical names that independently match the search algorithm

### Requirement: Fast keyboard class assignment
The class chooser SHALL support text focus, deterministic result ordering,
keyboard traversal, confirmation, and cancellation without requiring pointer
selection for each new or relabeled box.

#### Scenario: Reviewer labels several boxes
- **WHEN** the reviewer draws a box, types part of a class, and confirms the highlighted result
- **THEN** the canonical class is applied and the editor can return to the configured drawing flow

### Requirement: Overlay-only dense-scene focus
The editor SHALL provide show-all, dim-non-selected, hide-non-selected, and
restore controls that affect annotation overlays only and SHALL preserve image
pixels and annotation payloads.

#### Scenario: Selected object is obscured by overlays
- **WHEN** the reviewer chooses hide-non-selected for a selected bbox
- **THEN** all other overlays are hidden while the underlying image and every annotation remain unchanged

#### Scenario: Focus mode is cleared
- **WHEN** the reviewer restores all overlays
- **THEN** all non-deleted annotation regions return with no data mutation

### Requirement: Nearby inference instances are visually distinct
All uncommitted inference-origin regions in the active task SHALL use versioned
`visual_policy_v1`: two boxes are neighbors when their norm1000 rectangles
intersect after every edge is expanded by 12 bins and clipped to `0..999`;
stable region keys SHALL drive deterministic greedy assignment from an
accessible high-contrast palette while canonical class identity remains visible.

#### Scenario: Multiple inferred boxes overlap or lie nearby
- **WHEN** one or several sequential ROI responses leave neighboring uncommitted inference-origin instances
- **THEN** adjacent instances receive different presentation colors when available

#### Scenario: Same Draft is rendered again
- **WHEN** task state and region identities are unchanged
- **THEN** the same deterministic instance colors and groups are restored from persisted non-training request/region metadata

#### Scenario: Palette is exhausted
- **WHEN** a neighborhood clique is larger than the available palette
- **THEN** color reuse is deterministic and a visible numeric instance badge preserves local distinction

### Requirement: Potential duplicate cues are advisory
The editor SHALL mark same-class pairs with IoU at least `0.5` as potential
duplicates for review and SHALL never block Commit, suppress, replace, merge, or
delete a box because of that cue.

#### Scenario: Same-class boxes overlap strongly
- **WHEN** two active boxes of the same canonical class have IoU at least `0.5`
- **THEN** both remain independently editable and receive a nonblocking conflict cue

### Requirement: Instance color is non-semantic and temporary
Instance-local inference color SHALL NOT change class IDs, class names, geometry,
object ordering, exported fields, or Commit behavior, and SHALL revert to the
ordinary class-color presentation after successful sample Commit.

#### Scenario: Colored inference boxes are committed
- **WHEN** a sample containing locally colored inference regions commits successfully
- **THEN** the working JSONL contains no presentation fields, journal links remain, and the editor returns those regions to normal class colors

### Requirement: Dirty state is explicit
The editor SHALL compare the canonical semantic projection of the authoritative
annotation with that task's last committed row hash/time and SHALL visibly
distinguish `Committed`, `Draft`, `Committing`, and `Reconciling outcome`.
Validation/write errors SHALL be banners on the dirty Draft, while project
generation is displayed separately.

#### Scenario: Region is edited after Commit
- **WHEN** geometry, class, membership, or inference insertion changes the active annotation
- **THEN** the task immediately displays Draft/dirty state until a successful Commit restores parity

#### Scenario: Ordinary Draft save occurs
- **WHEN** Label Studio saves the current annotation but the working JSONL is not updated
- **THEN** the task remains visibly Draft rather than claiming dataset Commit

#### Scenario: Another task commits
- **WHEN** project generation advances because a different task commits
- **THEN** the current task remains Committed or Draft according to its own semantic row hash

#### Scenario: Commit outcome is unknown
- **WHEN** the browser loses a response after the server may have durably replaced the row
- **THEN** the task shows reconciliation state and resolves the idempotent commit ID before allowing another semantic Commit

### Requirement: Dirty navigation guard
The editor SHALL intercept in-app task navigation, Previous/Next, route changes,
and editor-close controls while dirty and SHALL offer `Commit and continue`,
`Continue with saved Draft`, and `Stay`, with persisted-Draft discard/reset only
as an explicit secondary choice. Hard reload/tab/window close SHALL use only the
browser-native unsaved-work warning and SHALL make no async save/Commit promise.

#### Scenario: Reviewer clicks Next with uncommitted edits
- **WHEN** the active task is dirty
- **THEN** navigation pauses and the guard explains that the working JSONL has not been updated

#### Scenario: Commit and continue succeeds
- **WHEN** the reviewer chooses Commit and the durable Commit succeeds
- **THEN** navigation resumes to the requested destination

#### Scenario: Commit and continue fails
- **WHEN** the reviewer chooses Commit and the Commit fails
- **THEN** the reviewer remains on the task with the Draft intact and an actionable error

#### Scenario: Continue with saved Draft is chosen
- **WHEN** the reviewer chooses Draft save
- **THEN** navigation continues only after confirmed Draft-save success and the working JSONL remains at its prior generation

#### Scenario: Draft save before navigation fails
- **WHEN** the selected Draft-save request fails
- **THEN** the reviewer remains on the task with the unsaved/dirty state visible

#### Scenario: Browser reload is requested while dirty
- **WHEN** the reviewer reloads or closes the tab/window
- **THEN** the browser-native Leave/Stay warning appears without offering or claiming asynchronous Commit

### Requirement: Deletion is explicit and undoable before Commit
The editor SHALL support deletion of any active bbox instance and SHALL retain
native Draft undo/revision behavior before a successful sample Commit. A
successful Commit SHALL rebase the undo history and tombstone deleted IDs.

#### Scenario: Existing object is deleted
- **WHEN** the reviewer deletes a selected object
- **THEN** it disappears from the active Draft and is omitted from the next valid committed object list

#### Scenario: Every object is deleted
- **WHEN** deletion leaves the Draft empty
- **THEN** the editor preserves the Draft but communicates that the approved V1 Commit contract requires at least one object

#### Scenario: Undo is requested after a committed deletion
- **WHEN** the reviewer invokes browser undo or reload after the deletion Commit succeeded
- **THEN** the tombstoned object does not reappear and a redrawn replacement receives a new identity

### Requirement: Visualization state never changes dataset state
Zoom, pan, selection, focus/hide mode, per-region visibility, and inference
presentation colors SHALL not mark object semantics changed or alter the working
JSONL.

#### Scenario: Reviewer only changes visibility
- **WHEN** overlays are hidden, dimmed, selected, zoomed, or restored without annotation edits
- **THEN** the sample's object payload and Commit generation remain unchanged

#### Scenario: Focused object is deleted
- **WHEN** the selected object disappears while hide-non-selected is active
- **THEN** the editor restores Show All so the canvas cannot appear falsely empty

#### Scenario: Inference inserts while others are hidden
- **WHEN** valid ROI results are inserted during hide-non-selected mode
- **THEN** the editor exits that mode and selects or flashes the inserted group without changing pre-existing objects
