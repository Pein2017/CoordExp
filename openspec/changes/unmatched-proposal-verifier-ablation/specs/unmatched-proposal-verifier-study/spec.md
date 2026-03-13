# unmatched-proposal-verifier-study Specification (Delta)

## ADDED Requirements

### Requirement: The study is a reproducible offline ablation over existing checkpoints
The system SHALL provide a reproducible offline study workflow for evaluating
proposal-verification proxies on unmatched predicted objects without introducing
new detector heads or large-scale training.

Normative behavior:

- the study MUST operate on existing checkpoints,
- the study MUST accept an explicit checkpoint list via config rather than
  requiring hard-coded checkpoint paths,
- the study MUST accept a dataset JSONL path and deterministic sample count,
- the study MUST default to a small reproducible subset rather than a full-dataset
  run,
- the study MUST make deterministic claims only for:
  - subset sampling,
  - offline GT-positive / hard-negative construction,
  - teacher-forced scoring over frozen collected proposal artifacts,
- the study MUST record the resolved subset path, sample count, seed, and
  checkpoint list in its output artifacts,
- the study MUST default to proposal collection on:
  - `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
  - `N=200`
  unless the user selects another supported input,
- if the sampled subset is materialized outside the source dataset directory,
  the study config MUST set `run.root_image_dir` to the original dataset root,
- the study MUST NOT require any training step or model-architecture change.

#### Scenario: Default study uses a small reproducible subset
- **GIVEN** a study config that omits custom sampling overrides
- **WHEN** the study is started
- **THEN** it samples a deterministic subset from the default COCO 1024 input
- **AND** it records the exact subset metadata alongside the study outputs.

### Requirement: Proposal collection reuses the existing infer and evaluation stack with vLLM collection settings
The study SHALL collect rollout proposals through the existing inference and
detection-evaluation stack rather than a bespoke proposal collector.

Normative behavior:

- rollout proposal collection MUST reuse the existing infer pipeline contract,
- collection MUST support `backend.type: vllm`,
- the initial default collection settings MUST be:
  - `temperature: 0.1`
  - `repetition_penalty: 1.1`
  - `infer.generation.batch_size: 16`
  - `infer.backend.server_options.vllm_gpu_memory_utilization: 0.9`,
- the study MUST also support an explicit temperature sweep over multiple
  rollout temperatures while keeping the other collection and scoring controls
  fixed,
- when a temperature sweep is configured, the study MUST preserve `temperature`
  in per-run manifests, proposal tables, and aggregate reports,
- vLLM collection MUST be treated as best-effort repeatable rather than
  byte-identical deterministic,
- the study MUST freeze collected rollout artifacts before computing final
  scoring/reporting outputs,
- the study MUST reuse the detection evaluator to derive matched and unmatched
  proposal buckets,
- the evaluator configuration used by the study MUST set
  `f1ish_pred_scope: all` so unmatched open-vocabulary proposals are preserved
  instead of silently ignored,
- the proposal table MUST preserve proposal order and include at least:
  - checkpoint id,
  - record / image identifiers,
  - proposal index,
  - desc,
  - bbox,
  - matched / unmatched status.

#### Scenario: Study collection preserves unmatched open-vocabulary proposals
- **GIVEN** a rollout prediction whose description does not map to the image GT
  label space
- **WHEN** the study evaluator buckets proposals for analysis
- **THEN** that proposal remains visible in the proposal table
- **AND** it is not silently removed by an `annotated` prediction-scope policy.

#### Scenario: Temperature sweep keeps runs separable
- **GIVEN** a study config with multiple rollout temperatures
- **WHEN** the study collects and scores proposals
- **THEN** each run records its own `temperature`
- **AND** downstream summaries can compare verifier behavior by temperature
  without mixing artifacts across temperatures.

### Requirement: The study constructs GT positives and GT-derived hard negatives explicitly
The study SHALL build a labeled offline proxy-validation table from the sampled
dataset records using GT positives and GT-derived hard negatives.

Normative behavior:

- GT positives MUST reuse the dataset GT desc and bbox content as the positive
  realism examples,
- hard negatives MUST be derived from the same sampled images rather than random
  unrelated boxes,
- the initial hard-negative families MUST support:
  - same-desc wrong-location jitter,
  - desc / box cross-swap within the same image,
  - same-class wrong-location negatives when possible,
- the study MAY add oversized or group-box negatives when they can be generated
  deterministically,
- each negative example MUST carry enough metadata to explain how it was
  synthesized.

#### Scenario: Same-desc wrong-location negative remains auditable
- **GIVEN** a GT object and a synthesized same-desc wrong-location negative
- **WHEN** the study writes the labeled table
- **THEN** the negative row records its source GT object and synthesis family
- **AND** downstream reporting can isolate that negative family.

### Requirement: Teacher-forced proposal verifiers are desc-first, bbox-masked, and auditable
The study SHALL score proposal-verification proxies through the existing
teacher-forced forward path rather than a new scoring model.

Normative behavior:

- for rollout proposals, v1 scoring MUST use one fixed teacher-forced assistant
  sequence per sampled image / checkpoint rollout,
- the normative v1 fixed-sequence source mode MUST be
  `canonicalized_fixed_sequence_v1`, derived from the parsed rollout while
  preserving proposal order,
- the primary commitment proxy MUST be desc-only average log-probability under
  teacher forcing on the original image,
- one baseline teacher-forced forward on the original image MUST yield
  commitment scores for all proposal desc spans in that fixed sequence,
- the primary counterfactual proxy MUST be the drop in that commitment score
  after masking or occluding the proposal bbox region on the image,
- the initial counterfactual mask region MUST be bbox-based rather than
  attention-based,
- masked counterfactual scoring MUST reuse the same fixed teacher-forced
  assistant sequence and MUST NOT require an additional rollout,
- masked counterfactual scoring SHOULD batch masked images for the selected
  proposals whenever feasible,
- GT positives and GT-derived hard negatives MAY be scored under a canonical
  no-rollout prefix context,
- the study MUST record:
  - fixed-sequence source mode,
  - desc-span extraction mode,
  - mask policy,
  - prompt/control provenance used for scoring,
- the study MUST emit per-proposal score columns for:
  - commitment
  - masked commitment
  - counterfactual
  - a simple combined score,
- optional logistic calibration MAY be emitted as a secondary score, but the
  study MUST preserve a transparent non-learned combined score for auditability.

#### Scenario: Counterfactual score is defined as a commitment drop
- **GIVEN** a proposal with desc tokens and a candidate bbox
- **WHEN** the study scores it on the original and masked images
- **THEN** the stored counterfactual score equals
  `commitment_original - commitment_masked`
- **AND** both component scores remain available for audit.

#### Scenario: Fixed-sequence counterfactual does not rerollout
- **GIVEN** one sampled image / checkpoint rollout with a frozen fixed assistant
  sequence
- **WHEN** the study computes counterfactual scores for selected proposals
- **THEN** it reuses that same sequence for the original-image and masked-image
  teacher-forced forwards
- **AND** it does not rerollout after masking.

### Requirement: Scoring failures are explicit and metric denominators remain reproducible
The study SHALL preserve scoring failures explicitly rather than silently
dropping rows from the proposal table.

Normative behavior:

- each scored proposal row MUST carry:
  - `scoring_status`
  - `failure_reason` when scoring did not succeed,
- the study MUST define stable scoring failure reasons for at least:
  - `sequence_canonicalization_failed`
  - `assistant_span_build_failed`
  - `missing_desc_span`
  - `empty_desc_span`
  - `degenerate_bbox`
  - `mask_build_failed`
  - `nonfinite_logprob`,
- proposal rows with scoring failures MUST remain in the scored proposal table,
- rows with scoring failures MUST be excluded from primary AUROC/AUPRC
  denominators,
- the aggregate summary/report MUST count excluded rows by failure reason.

#### Scenario: Missing desc span remains visible but excluded from AUROC
- **GIVEN** a proposal whose desc span cannot be extracted from the fixed
  teacher-forced sequence
- **WHEN** the scored proposal table is written
- **THEN** the row remains present with
  `scoring_status=\"failed\"` and `failure_reason=\"missing_desc_span\"`
- **AND** the row is excluded from primary AUROC/AUPRC denominators
- **AND** the summary counts that failure reason explicitly.

### Requirement: Prompt and scoring provenance are recorded per checkpoint
The study SHALL record the prompt/control settings needed to reproduce proposal
collection and teacher-forced scoring for each checkpoint.

Normative behavior:

- for each checkpoint, the study MUST record:
  - `prompt_variant`
  - `object_field_order`
  - scorer template settings needed to rebuild the teacher-forced assistant
    span,
  - the evaluator semantic model path used for matched/unmatched bucketing,
- when checkpoint-native resolved config metadata is available, the study SHOULD
  default to those values rather than inventing new prompt controls.

#### Scenario: Prompt-control drift is visible in the run manifest
- **GIVEN** two checkpoints that use different prompt/control settings
- **WHEN** the study writes its per-checkpoint manifests
- **THEN** those prompt/control settings are recorded explicitly
- **AND** the report can explain score differences without guessing.

### Requirement: The study reports cross-checkpoint evidence for GT-vs-hard-negative and matched-vs-unmatched separation
The study SHALL produce a concise but decision-oriented report over the scored
tables.

Normative behavior:

- the report MUST state:
  - exact subset used,
  - checkpoint list,
  - resolved rollout settings,
  - proxy definitions,
  - scoring backend assumptions,
- the study MUST compute and persist at least:
  - AUROC,
  - AUPRC,
  - score distributions,
  - commitment / counterfactual correlation,
  - GT-positive vs hard-negative separation,
  - matched-vs-unmatched rollout separation,
  - top-k unmatched precision-style analysis,
- the report MUST compare:
  - commitment alone,
  - counterfactual alone,
  - commitment + counterfactual,
- when a temperature sweep is configured, the report MUST also compare the
  verifier metrics and proposal distributions across temperatures,
- the study SHOULD emit calibration / reliability bins when label support is
  sufficient,
- when calibration is skipped due to insufficient support, the report MUST say
  so explicitly,
- the report MUST include an explicit recommendation on whether the proxy signal
  is strong enough to justify later soft pseudo-label promotion,
- the study SHOULD prepare a small optional audit pack of high-scoring unmatched
  proposals for later manual review.

#### Scenario: Final report answers the promotion decision directly
- **GIVEN** a completed study run
- **WHEN** the markdown report is written
- **THEN** it explicitly states which single proxy is strongest
- **AND** whether the combined score materially outperforms either single proxy
- **AND** whether the signal appears stable enough to justify later soft
  pseudo-label promotion experiments.
