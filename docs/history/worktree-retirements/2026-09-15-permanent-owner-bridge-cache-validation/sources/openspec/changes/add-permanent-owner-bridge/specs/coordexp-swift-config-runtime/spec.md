## ADDED Requirements

### Requirement: Strict Stage 1 owner-bridge profile

Training configuration SHALL represent the permanent owner bridge as one strict, versioned model-family profile. For the initial Qwen3-VL-2B profile, configuration MUST resolve `K=4`, post-block-20 atom/read-write placement, post-final-RMSNorm router readout, a three-percent RMS admission cap, a ten-percent RMS row-write cap, and the four-presentation `geo_sorted, random-1, geo_sorted, random-2` schedule. Unsupported model families, independent layer overrides, alternate slot counts, or Stage 2 rollout settings MUST fail before cache preparation or model loading.

The production leaf MUST bind the exact source checkpoint identity, source resolved-config identity, existing COCO-80 train/eval paths, run seed, four complete presentations, global effective batch size 24, `global_max_length=12000`, the accepted no-resize processor limits including `max_raw_pixels=1048576` and `max_merged_visual_tokens=4096`, annotation trust class `ordinary_partial`, fresh AdamW state, cosine scheduling, ten-percent warmup, language DoRA learning rate `5e-5`, selected-embedding learning rate `2.5e-5`, bridge learning rate `2.5e-4`, and loss weights `L_AR=1`, `L_atom=1`, `L_route=0.5`, and `L_use=0.1`. It MUST set `adapter.seed_mode: warm_start_expand_dora`, bind `adapter.source_adapter_path` to the step-2444 `adapter/` payload and `adapter.repaired_embedding_payload_path` to the companion `special_token_embeddings/` payload, and record both payload fingerprints. Stage 1 MUST reject a mixed or trusted annotation source unless a later resolved production leaf explicitly identifies and fingerprints that source and its sampling ratio.

In-training evaluation MUST use the same full teacher-forced forward, matching, latch, and loss semantics as training with gradients disabled. The production cadence SHALL resolve one midpoint and one end event for each presentation on the predetermined planned-step clock: eight total landmarks at approximately 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, and 100%. Every event MUST reuse the same fingerprinted `geo_sorted` eval rendering. An unsafe optimizer update at a landmark MUST NOT cancel, delay, or rename the eval event; the artifact MUST retain the original planned-step id and record that the update was skipped. In-training rollout or natural decode evaluation MUST remain disabled.

#### Scenario: Independent bridge layer override
- **WHEN** a Stage 1 config attempts to override the atom layer, row-write layer, or router-read layer independently of the named architecture profile
- **THEN** strict config validation MUST fail before expensive work

#### Scenario: Resolved Stage 1 schedule
- **WHEN** a valid production Stage 1 config is resolved
- **THEN** its artifacts MUST identify all four presentations, their order policies and seeds, optimizer groups and learning rates, cap ramps, loss ramps, source checkpoint, and source resolved config
- **AND** they MUST identify the production annotation trust class as `ordinary_partial`

#### Scenario: Undeclared trusted source
- **WHEN** a Stage 1 production config supplies a trusted/exhaustive example or sampling ratio while the fixed production leaf declares only `ordinary_partial`
- **THEN** config or data validation MUST fail before cache preparation

#### Scenario: Undeclared in-training evaluation cadence requested
- **WHEN** the Stage 1 production leaf requests an evaluation cadence other than the declared midpoint-and-end landmarks or enables inference evaluation during training
- **THEN** strict config validation MUST reject that production cadence

#### Scenario: Presentation midpoint or end teacher-forced evaluation
- **WHEN** any declared midpoint or presentation-end planned-step landmark is reached
- **THEN** evaluation MUST execute the same forward and loss bundle without backward or optimizer mutation

#### Scenario: Optimizer update is unsafe at an eval landmark
- **WHEN** the optimizer update associated with a declared eval planned step is skipped as unsafe
- **THEN** that eval MUST still execute under its original scheduled event id
- **AND** its artifact MUST record both the planned-step identity and skipped-update outcome

#### Scenario: Stage 2 field appears in Stage 1
- **WHEN** a Stage 1 config declares rollout actors, replay buffers, snapshot cadence, self-prefix ratio, or atom-id replay
- **THEN** validation MUST reject the unsupported field rather than ignore it

### Requirement: Stage 1 owner-bridge ramps

The Stage 1 schedule SHALL ramp the row-write RMS cap from zero to ten percent during the first `geo_sorted` presentation, hold the boundary admission cap at no more than three percent, ramp `L_route` to its configured weight over the first five percent of planned optimizer steps, and ramp `L_use` to its configured weight over the first twenty percent. Ramp positions MUST use completed safe optimizer steps against the resolved planned-step schedule.

#### Scenario: Skipped optimizer update during a ramp
- **WHEN** an unsafe optimizer update is skipped
- **THEN** bridge-cap and auxiliary-loss ramps MUST NOT advance for that skipped update
