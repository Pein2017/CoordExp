## MODIFIED Requirements

### Requirement: Qualified inference backends

The system SHALL expose dynamic HF and vLLM inference. Dynamic HF SHALL remain
the authoritative adapter-plus-embedding-delta execution semantics. A composed
vLLM production path SHALL execute qualification and fail closed when its
declared requirements are not met. Qualification identity SHALL bind the
project-local semantic source dependencies used by qualification and production
execution. Qualification children SHALL have a finite positive execution
deadline configurable by the caller. Deadline expiry SHALL trigger bounded
owned-process-group cleanup and an explicit failure, never successful admission.
External GPU census commands in this supervision path SHALL also be bounded;
unavailable cleanup evidence SHALL NOT be treated as successful cleanup.

#### Scenario: A composed model is requested through vLLM

- **WHEN** a vLLM configuration resolves a composed execution model
- **THEN** production execution is denied unless the complete matching qualification set has been admitted

#### Scenario: A project-local semantic source dependency changes

- **WHEN** a template renderer or another covered semantic dependency changes after qualification
- **THEN** the old qualification identity does not match and production admission rejects its receipts

#### Scenario: A qualification child exceeds its deadline

- **WHEN** a child or a descendant holding its output pipes prevents completion before the configured deadline
- **THEN** the supervisor performs bounded cleanup of its owned process group and reports failure without publishing an admitted receipt set

#### Scenario: A caller supplies an invalid deadline

- **WHEN** the caller supplies a non-finite or non-positive deadline
- **THEN** qualification rejects it before launching a child

#### Scenario: GPU census does not finish

- **WHEN** a GPU census command exceeds its bounded wait
- **THEN** its evidence is unavailable and qualification cannot claim successful cleanup from that command
