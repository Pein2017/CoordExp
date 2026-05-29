# teacher-forcing-objective-pipeline Delta

## MODIFIED Requirements

### Requirement: Live teacher-forcing pipelines use unified objective modules

The live teacher-forcing objective pipeline SHALL route active teacher-forcing
loss computation through the new unified objective modules.

Normative behavior:

- active pipelines MUST consume `teacher_forcing_target_ir`;
- active pipelines MUST NOT consume old recursive-detection sidecars;
- active pipelines MUST NOT expose duplicate unlikelihood, bbox geometry aux,
  bbox size aux, coordinate regression, W1, adjacent repulsion, or old coord
  gate modules as core teacher-forcing modules;
- stage-specific adapters MAY build target IR, but objective modules MUST remain
  shared.

#### Scenario: Removed auxiliary module is rejected

- **WHEN** a new teacher-forcing pipeline config declares
  `loss_duplicate_burst_unlikelihood`, `bbox_geo`, `bbox_size_aux`, or
  `coord_reg`
- **THEN** pipeline validation fails fast
- **AND** the error explains that these are not active teacher-forcing core
  modules.
