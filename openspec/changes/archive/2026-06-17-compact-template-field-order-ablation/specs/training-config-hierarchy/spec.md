## ADDED Requirements

### Requirement: Compact ablation configs remain YAML-first and auditable
Repo-owned compact field-order ablation configs SHALL express all experiment
identity through existing YAML config sections and SHALL NOT require new stable
CLI flags.

The production and smoke leaves for the paired ablation MUST make these values
directly auditable in the leaf or inherited facet chain:

- checkpoint path,
- detection template id,
- object field order,
- object ordering,
- static packing and global length,
- trainable module policy,
- run/output/logging identity.

#### Scenario: Bbox-first leaf exposes high-signal identity
- **GIVEN** the bbox-first compact ablation leaf config
- **WHEN** an operator opens the config and its declared inherited facets
- **THEN** the checkpoint, compact template id, `geometry_first` field order,
  sorted object ordering, `global_max_length: 12000`, and LLM-only policy are
  visible without relying on CLI flags.

#### Scenario: Smoke overlay changes only runtime scale
- **GIVEN** the bbox-first smoke config
- **WHEN** it is compared to the production bbox-first config
- **THEN** differences are limited to runtime scale, output/logging identity,
  and smoke-specific limits
- **AND** semantic template, field order, object ordering, checkpoint, and
  packing identity remain aligned.
