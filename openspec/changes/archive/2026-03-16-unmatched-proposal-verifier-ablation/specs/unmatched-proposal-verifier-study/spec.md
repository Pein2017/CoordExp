# unmatched-proposal-verifier-study Specification (Delta)

## ADDED Requirements

### Requirement: The study must separate clean verifier validation from rollout unmatched validation
The system SHALL treat clean GT-vs-hard-negative evaluation and rollout
proposal evaluation as different evidence layers with different claims.

Normative behavior:

- the study MUST preserve a clean verifier benchmark built from:
  - GT positives
  - GT-derived hard negatives,
- the study MUST preserve a rollout proposal benchmark built from:
  - collected rollout predictions
  - matched / unmatched / ignored proposal buckets,
- the study MUST NOT treat success on the clean GT slice alone as sufficient
  evidence for unmatched pseudo-label promotion,
- the final report MUST explicitly distinguish:
  - conclusions supported by the clean GT slice,
  - conclusions supported by rollout proposal evidence.

#### Scenario: Clean-slice success does not imply pseudo-label readiness
- **GIVEN** a study run where GT-vs-hard-negative metrics are strong
- **WHEN** rollout proposal evidence is weak, invalid, or missing
- **THEN** the report MUST preserve the GT conclusion
- **AND** it MUST explicitly withhold a strong pseudo-label-promotion
  recommendation.

### Requirement: Rollout proposal comparison requires a collection-validity gate
The system SHALL evaluate rollout collection health before using a run in the
main unmatched-proposal comparison.

Normative behavior:

- each checkpoint × temperature run MUST emit collection-health fields
  sufficient to judge whether the proposal population is usable,
- at minimum, collection-health outputs MUST include:
  - `temperature`
  - `checkpoint`
  - `pred_count_total`
  - `pred_count_per_image_mean`
  - `nonempty_pred_image_rate`
  - `matched_count`
  - `unmatched_count`
  - `ignored_count`
  - `invalid_rollout_count`
  - parser failure counts
  - duplicate-like rate,
- each run MUST record:
  - `collection_valid`
  - `collection_invalid_reason` when invalid,
- only collection-valid runs MUST enter the main rollout proxy comparison
  tables and temperature sweep conclusions,
- collection-invalid runs MUST remain visible in artifacts and appendices rather
  than being silently dropped.

#### Scenario: Collection-invalid runs are excluded from main temperature comparison
- **GIVEN** a checkpoint × temperature run with too few usable rollout
  proposals
- **WHEN** the study aggregates rollout proxy metrics across temperatures
- **THEN** that run is excluded from the main comparison tables
- **AND** its failure reason remains recorded in collection-health outputs.

### Requirement: The main temperature sweep is limited to four interpretable temperatures
The system SHALL restrict the authoritative temperature sweep to four explicit
values chosen to maximize interpretability rather than density.

Normative behavior:

- the main sweep MUST use exactly these temperatures:
  - `0.0`
  - `0.3`
  - `0.5`
  - `0.7`,
- the study MUST preserve `temperature` in per-run manifests, proposal tables,
  collection-health outputs, and aggregate reports,
- the report MUST interpret temperature primarily through rollout proposal
  evidence rather than the clean GT slice alone.

#### Scenario: Temperature is preserved across outputs
- **GIVEN** a multi-temperature study run
- **WHEN** the study writes collection, scoring, and report artifacts
- **THEN** each artifact can be attributed unambiguously to one temperature.

### Requirement: The study must be staged and resumable
The system SHALL support a staged workflow so collection, scoring, reporting,
and audit can be resumed independently.

Normative behavior:

- the study MUST support these stages:
  - subset / GT table preparation
  - rollout collection
  - collection-health summarization and gating
  - rollout scoring for collection-valid runs
  - report aggregation
  - manual audit artifact preparation / ingestion,
- each stage SHOULD emit its own manifest or equivalent provenance artifact,
- later stages MUST be able to reuse earlier frozen outputs rather than forcing
  a monolithic rerun.

#### Scenario: Scoring can be rerun without recollecting rollouts
- **GIVEN** frozen rollout artifacts and collection manifests already exist
- **WHEN** the user reruns only the scoring/report stages
- **THEN** the study reuses the existing collection outputs
- **AND** does not recollect rollouts unnecessarily.

### Requirement: The final recommendation requires a small manual unmatched audit
The system SHALL include a small manual audit layer before making a strong
pseudo-label promotion recommendation about unmatched proposals.

Normative behavior:

- the study MUST prepare a manually auditable unmatched subset,
- the audit subset SHOULD be stratified by:
  - checkpoint
  - temperature
  - score quantile
  - nearest-GT weak-overlap bucket,
- the audit schema SHOULD distinguish at least:
  - `real_visible_object`
  - `duplicate_like`
  - `wrong_location`
  - `dead_or_hallucinated`
  - `uncertain`,
- the final report MUST use this audit layer when deciding whether high-scoring
  unmatched proposals are trustworthy enough for pseudo-label promotion,
- absent this audit layer, the final report MUST downgrade the recommendation to
  a non-promotion-ready conclusion.

#### Scenario: No manual audit means no strong promotion claim
- **GIVEN** a study run without completed manual unmatched audit labels
- **WHEN** the final report is written
- **THEN** it may describe the verifier as promising
- **BUT** it MUST NOT present unmatched pseudo-label promotion as strongly
  validated.

### Requirement: Final conclusions must meet an authority-first decision standard
The system SHALL only present a strong unmatched-proposal recommendation when
multiple evidence layers agree.

Normative behavior:

- the final report MUST separately evaluate:
  - clean GT-vs-hard-negative evidence,
  - rollout collection validity,
  - rollout proposal scoring evidence,
  - manual unmatched audit evidence,
- the study SHOULD only claim strong pseudo-label-promotion readiness when:
  - `counterfactual` clearly outperforms `commitment` on the clean slice,
  - rollout collection is valid for most checkpoint × temperature conditions,
  - rollout proposal scoring remains informative on collection-valid runs,
  - manual audit confirms that high-scoring unmatched proposals are often real
    visible objects,
- otherwise, the report MUST explicitly downgrade the conclusion to a
  “promising but not yet promotion-ready” status.

#### Scenario: Mixed evidence produces a downgraded recommendation
- **GIVEN** a study run where clean-slice metrics are strong but rollout or
  audit evidence is incomplete or weak
- **WHEN** the report writes the final recommendation
- **THEN** it acknowledges the verifier signal
- **AND** it explicitly avoids a strong pseudo-label-promotion claim.
