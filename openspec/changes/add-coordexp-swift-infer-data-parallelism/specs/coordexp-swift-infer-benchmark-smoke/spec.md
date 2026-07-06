## ADDED Requirements

### Requirement: Real multi-GPU HF smoke
Data-parallel inference SHALL be validated with a real multi-GPU HF smoke before any production data-parallel benchmark claim.
The smoke MUST use at least two visible CUDA devices, `backend.type: hf`, a real
Qwen model path, and a real input subset. It MUST verify worker binding,
rank-local artifacts, strict merge, merged scored artifacts, and evaluator
consumption. Evaluator smoke evidence MUST include an evaluation receipt that
binds the consumed raw, scored, provenance, and run-manifest artifacts by
SHA-256.

#### Scenario: Two-GPU smoke passes
- **WHEN** a two-GPU data-parallel HF smoke runs on a valid tiny or val subset
- **THEN** each active rank writes shard artifacts
- **AND** the merged top-level scored artifact has complete row coverage
- **AND** the evaluator consumes the merged scored artifact
- **AND** the evaluator writes a receipt binding the consumed artifact hashes

#### Scenario: Mock-only evidence
- **WHEN** only mocked worker-launch or merge tests pass
- **THEN** the implementation is not accepted as production data-parallel
  inference evidence

### Requirement: Evidence scope labeling
Data-parallel inference evidence SHALL label its scope.
Tiny and val-subset smokes are implementation evidence only. Production
benchmark claims require an explicitly launched production-scope run and must
name the config, dataset, checkpoint/adapter, visible GPU set, artifact root,
merged artifact path, and evaluator output.

#### Scenario: Tiny smoke evidence
- **WHEN** a tiny two-GPU smoke succeeds
- **THEN** the acceptance note labels it as smoke evidence, not full benchmark
  evidence
- **AND** evaluator metrics produced for that smoke are not marked as
  production benchmark metrics when the inference manifest is not benchmark
  eligible

#### Scenario: Production claim
- **WHEN** a production data-parallel benchmark result is reported
- **THEN** the report names the config, dataset, checkpoint or adapter, visible
  GPU set, merged inference artifact root, and evaluator metrics path
