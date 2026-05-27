## ADDED Requirements

### Requirement: Training surface descriptors declare whether they are active runtime authority

Training surface and pipeline descriptors SHALL be classified as active runtime
authority, migration scaffolding, shadow/audit-only descriptors, or historical
evidence.

Normative behavior:

- a descriptor-only pipeline module MUST NOT be documented as owning runtime
  behavior unless deleting it would remove real behavior;
- active docs and catalog entries MUST distinguish shadow validation surfaces
  from production launcher/runtime surfaces;
- retired specs or progress notes MUST NOT be routed as current authority when
  newer docs/specs govern the area;
- if a shadow descriptor is promoted to active runtime authority, tests MUST
  prove which behavior it owns.

#### Scenario: Descriptor-only pipeline is not mistaken for runtime owner

- **GIVEN** a training pipeline module only returns identity metadata
- **WHEN** docs, catalog entries, or architecture gates describe the training
  surface
- **THEN** the module is described as descriptor/shadow/audit-only unless it is
  deepened to own runtime validation or projection behavior
- **AND** future implementers are not directed to it as the runtime owner.
