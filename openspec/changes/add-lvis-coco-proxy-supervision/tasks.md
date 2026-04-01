## 1. OpenSpec Foundation

- [ ] 1.1 Keep the first implementation aligned with the agreed scope:
  - offline augmented COCO JSONL
  - determined proxy-mapping analysis before export
  - `same_extent_proxy | cue_only_proxy | reject`
  - `real | strict | plausible` object tiers
  - unchanged rendered CoordJSON object syntax
  - structure CE stays global
  - metadata-driven desc / coord weighting
  - Stage-2-first implementation
- [ ] 1.2 Keep the delta spec set in sync for:
  - `lvis-coco-proxy-supervision`
  - `teacher-forcing-unified-loss-registry`
  - `teacher-forcing-objective-pipeline`
  - `stage2-ab-training`
  - `rollout-matching-sft`
  - `trainer-metrics-components`
- [ ] 1.3 Re-validate the change after each artifact pass:
  - `openspec validate add-lvis-coco-proxy-supervision --type change --strict --no-interactive`

## 2. Proxy Determination Research

- [ ] 2.1 Implement a finer-grained mapping-analysis pass that classifies
  candidate LVIS->COCO mappings as:
  - `same_extent_proxy`
  - `cue_only_proxy`
  - `reject`
- [ ] 2.2 Add extent-compatibility metrics beyond IoU, including:
  - `intersection_over_lvis`
  - `intersection_over_coco`
  - `area_ratio`
  - one-way containment rates
  - normalized center offset
- [ ] 2.3 Add focused reporting for representative mappings such as:
  - `mug -> cup`
  - `tablecloth -> dining table`
- [ ] 2.4 Fail fast or reject mappings whose evidence is too weak to assign a
  proxy strategy confidently.
- [ ] 2.5 Export a determined semantic-proxy artifact set before dataset export:
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv`
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_summary.json`
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_report.md`
- [ ] 2.6 Make the determined mapping artifact auditable, including:
  - raw support / geometry evidence columns
  - the final `determination_tier`
  - the final `mapping_class`
  - rule-version provenance
  - rank within tier

## 3. Offline Augmented Dataset Export

- [ ] 3.1 Implement an exporter that merges base COCO objects with recovered
  LVIS `strict` and `plausible` objects into one canonical record.
- [ ] 3.1a Make the exporter consume the determined mapping artifact rather
  than deriving semantic proxy policy ad hoc from `mapping_evidence.csv`.
- [ ] 3.2 Preserve the existing raw JSONL / CoordJSON object schema:
  - object entries remain standard `desc` + `bbox_2d`
- [ ] 3.3 Add top-level proxy metadata under a stable namespace with one entry
  per final object.
- [ ] 3.4 Fail fast when:
  - metadata/object counts differ
  - object sorting and metadata sorting diverge
  - unsupported proxy tiers are encountered
  - a supervised exported object is missing a determined proxy strategy
- [ ] 3.5 Keep final object ordering deterministic under the repo's canonical
  `(minY, minX)` ordering rule.
- [ ] 3.6 Add focused coverage for:
  - merge ordering
  - metadata alignment
  - tier/weight serialization
  - strategy-to-tier assignment
  - no syntax change in rendered assistant payload

## 4. Teacher-Forcing Context And Span Carriers

- [ ] 4.1 Extend the Stage-2 target-building path to derive object-local proxy
  supervision carriers from metadata.
- [ ] 4.2 Provide derived object-local views for:
  - mapping class by object
  - desc spans by object
  - bbox groups by object
  - desc weights by object
  - coord weights by object
- [ ] 4.3 Fail fast when desc spans, bbox groups, and metadata counts cannot be
  aligned exactly.
- [ ] 4.4 Keep structure-token supervision separate from object-local desc
  spans.
- [ ] 4.5 Add targeted coverage for:
  - `"desc"` / `"bbox_2d"` field names remaining structure tokens
  - desc-value spans following object-local weights
  - bbox groups following object-local coord weights

## 5. Stage-2 Module Integration

- [ ] 5.1 Extend `token_ce.config` with `object_weight_mode`.
- [ ] 5.2 Extend `bbox_geo.config` with `object_weight_mode`.
- [ ] 5.3 Extend `coord_reg.config` with `object_weight_mode`.
- [ ] 5.4 Support:
  - `none`
  - `metadata`
  object-weight modes with strict validation.
- [ ] 5.5 In `metadata` mode:
  - keep `struct_ce` global
  - apply metadata `desc_ce_weight` only to desc-value supervision
  - apply metadata `coord_weight` only to bbox/coord supervision
  - fall back to weight `1.0` if the metadata block is absent
- [ ] 5.6 Add Stage-2 tests covering:
  - real/strict/plausible weighted desc supervision
  - real/strict/plausible weighted bbox supervision
  - structure CE invariance under plausible objects
  - strict config allowlist behavior

## 6. Metrics And Observability

- [ ] 6.1 Add canonical training metrics for proxy-supervision observability,
  including:
  - real object count
  - strict object count
  - plausible object count
  - effective desc-weight sum
  - effective coord-weight sum
- [ ] 6.2 Document the canonical metric names in:
  - `docs/training/METRICS.md`
  - `docs/training/STAGE2_RUNBOOK.md`
- [ ] 6.3 Ensure proxy observability remains aggregation-safe and does not get
  diluted across packed micro-steps.

## 7. Validation

- [ ] 7.1 Re-run change validation:
  - `openspec validate add-lvis-coco-proxy-supervision --type change --strict --no-interactive`
- [ ] 7.2 Re-run focused Stage-2 config coverage:
  - `conda run -n ms pytest tests/test_stage2_ab_config_contract.py`
  - `conda run -n ms pytest tests/test_training_config_strict_unknown_keys.py`
- [ ] 7.3 Re-run focused Stage-2 objective coverage:
  - token CE tests touching desc vs struct weighting
  - bbox/coord tests touching per-group weights
- [ ] 7.4 Add or update a smoke-style artifact check that confirms:
  - augmented JSONL renders unchanged CoordJSON syntax
  - proxy metadata aligns with the final object order
  - plausible objects lower desc/coord effective supervision without lowering
    structure supervision

## 8. Deferred Follow-Up

- [ ] 8.1 Evaluate Stage-1 support for the same proxy-metadata contract.
- [ ] 8.2 Evaluate finer-grained plausible subtypes beyond one shared tier.
- [ ] 8.3 Evaluate learned or tuned per-mapping proxy weights after the first
  strict vs plausible ablations.
