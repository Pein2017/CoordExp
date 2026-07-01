## ADDED Requirements

### Requirement: Stable External Runner and Plugin Contract During Internal Refactor
The system MUST preserve the existing external interfaces of:
- `public_data/run.sh` command grammar and behavior contract,
- dataset shell plugin contract under `public_data/datasets/*.sh`,
while routing execution through the unified shared internal pipeline/factory implementation.

This preservation requirement applies throughout migration and after cutover.

#### Scenario: Existing runner invocation remains valid
- **WHEN** a user runs an existing command such as `./public_data/run.sh coco all --preset rescale_32_768_bbox`
- **THEN** command semantics remain valid without requiring new CLI flags
- **AND** execution is handled by unified internal pipeline stages under the hood.

#### Scenario: Existing dataset shell plugins remain valid integration points
- **WHEN** a dataset plugin implements the current shell contract in `public_data/datasets/<dataset>.sh`
- **THEN** the runner continues to invoke it successfully during and after migration
- **AND** plugin authors are not required to rewrite to a new external interface for this change.

### Requirement: Core Orchestration is Dataset-Agnostic
Shared pipeline orchestration MUST NOT hard-code dataset-specific processing logic in core execution flow.

Dataset-specific behavior MUST be encapsulated in adapter implementations resolved by the registry/factory.

Boundary definition:
- `public_data/run.sh` and dataset shell plugins are compatibility wrappers/integration surfaces.
- "Core orchestration" refers to the internal pipeline planner/stage executor implementation.

#### Scenario: Adding a new dataset avoids core orchestrator edits
- **WHEN** a new dataset is introduced through a new adapter and registry registration
- **THEN** shared orchestrator code path does not require dataset-conditional branches for conversion behavior.

#### Scenario: Dataset-specific fast paths are encapsulated
- **WHEN** a dataset needs a specialized optimization path
- **THEN** that behavior is implemented in adapter/plugin-owned integration surfaces
- **AND** core stage orchestration remains dataset-agnostic.

### Requirement: Optional Max-Object Filtering and Suffix Naming
Max-object filtering MUST be optional and disabled by default in unified pipeline execution.

When max-object filtering is enabled with value `N`, the effective preset/output naming MUST include suffix token `max_{N}` (rendered in path naming as `_max_<N>`), so filtered artifacts are self-describing.
For example, token `max_60` corresponds to rendered path segment `_max_60`.

If the effective preset/output name already contains the same suffix token, the system MUST NOT append it again.

Legacy naming compatibility:
- Existing `max{N}` artifact naming (for example `max60`) MUST be treated as equivalent to `max_{N}` for resolver compatibility.
- When an equivalent legacy-named artifact directory already exists, implementation MUST reuse it rather than creating a second forked directory that differs only by underscore style.
- Fresh-run emission policy is explicit:
  - if no equivalent legacy `max{N}` artifact exists, writer emits canonical `_max_<N>` naming,
  - if equivalent legacy `max{N}` artifact exists, resolver reuses legacy path instead of creating a parallel canonical path.

#### Scenario: Default run has no max-object filter and no suffix
- **WHEN** pipeline execution runs without max-object filtering configured
- **THEN** no object-count filtering stage is applied
- **AND** output preset naming remains unchanged.

#### Scenario: Enabled max-object filter appends deterministic suffix
- **WHEN** max-object filtering is enabled with `N=60`
- **THEN** effective output preset naming includes suffix token `max_60`
- **AND** generated artifacts are written under the suffixed preset directory.

#### Scenario: Existing suffix is not duplicated
- **WHEN** effective preset naming already ends with token `max_60`
- **THEN** enabling max-object filtering with `N=60` does not append a second `max_60` token.

#### Scenario: Legacy max60 artifacts are recognized
- **WHEN** equivalent legacy artifact naming exists as `max60`
- **THEN** preset resolution treats it as equivalent to `max_60`
- **AND** does not create a forked parallel artifact path solely due to suffix-token formatting.

#### Scenario: Fresh run emits canonical suffix when no legacy artifact exists
- **WHEN** max-object filtering is enabled and no equivalent legacy `max<N>` artifact path exists
- **THEN** writer emits canonical `_max_<N>` naming for new artifacts.
