## Why

Four committed HF research callers independently reach through
`HFBackendSession` private model, tokenizer, and native-input surfaces to replay
literal token histories and gather chosen-token evidence. That is enough
second-consumer evidence for one small HF-owned seam, but not for a generic
probe runner, artifact-verification framework, shared reducer, or research DSL.

The intended outcome is to remove only the repeated backend-private execution
knowledge while keeping every scientific choice—cohort, exact prefix,
intervention, candidates, controls, owner policy, estimand, aggregation,
uncertainty, claim boundary, and stop rule—visible in each research caller.

## What Changes

- Add an HF-only, single-request exact-history surface to `HFBackendSession`:
  backend-owned special-token IDs, opaque history preparation, literal token-ID
  extension, and teacher-forced chosen-token evidence.
- Keep exact histories immutable, session-bound, and auditable by request ID,
  conditioning token IDs, and SHA-256. Native tensors and the unforgeable
  session binding remain private implementation state rather than public
  attributes.
- Return only caller-requested chosen-token evidence: token ID, raw-model
  log-probability, and `candidate_vocab_rank`. A one-token terminal/opener
  comparison is a special case of the same sequence-scoring operation.
- Record observed HF model dtype and attention implementation separately from
  declared launch settings when the runtime can establish them.
- Validate the seam against two existing, semantically different consumers:
  `run_continuation_locality_boundary_scoring.py` for one-token boundary
  evidence and `run_exact_prefix_owner_compositionality.py` for multi-token
  candidate-row evidence.
- Preserve exact token-ID/hash parity and raw FP32 likelihood parity with each
  consumer's pre-migration path before removing only the superseded private
  accesses in those two scripts.
- Triage the explicit research/rollout-calibration suite before implementation.
  The current independent audit observed 38 mutable `unit.md` identity failures
  and 3 processed-data provenance failures. Record and route those known
  classes without changing `pytest.ini`, re-pinning data, or rewriting tests;
  stop this change if a third failure class appears.
- Use only two review gates: one after both real consumers pass parity, and one
  final audit before archive.

The change explicitly does **not** add a verified-run artifact handle, generic
model-session/probe abstraction, suffix-generation API, full-logit API,
scientific reducer, owner/matching policy, config-matrix DSL, new event family,
knowledge-workflow automation, or vLLM equivalent. It does not migrate the two
other existing private-HF consumers or historical artifacts.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-swift-infer-backend-trace`: Add a narrow HF-session contract for
  opaque exact-history construction and caller-selected teacher-forced token
  evidence, plus observed-runtime receipt fields, without changing ordinary
  decode or vLLM semantics.

## Impact

- Expected implementation owner: `src/inference/hf_backend.py`, with shared
  record definitions placed in `src/inference/backend.py` only if required by
  current inference ownership conventions.
- Pilot callers: exactly
  `scripts/research/run_continuation_locality_boundary_scoring.py` and
  `scripts/research/run_exact_prefix_owner_compositionality.py`.
- Tests: focused, default-collected tests under `tests/inference/` for history
  identity/lifecycle, literal append, chosen-token evidence, and observed receipt
  fields; no new artifact or research framework test tree.
- API impact is additive and HF-specific. Existing `DecodeRequest`,
  `DecodeResult`, `GenerationPolicy`, ordinary generation, parsing, scoring,
  evaluation, training, rollout-calibration, and artifact contracts remain
  unchanged.
- No new dependency, plugin registry, DAG engine, service, policy layer, or
  review bureaucracy is introduced.
