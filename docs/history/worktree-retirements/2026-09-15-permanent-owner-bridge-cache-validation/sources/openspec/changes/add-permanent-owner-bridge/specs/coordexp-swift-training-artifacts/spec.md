## MODIFIED Requirements

### Requirement: Minimal Inference Checkpoint Payloads

Scheduled checkpoint saving SHALL materialize only learned payloads required by
the inference engine: a standard PEFT/DoRA adapter directory when adapter
training is enabled, a compact selected-token embedding-delta directory when
that trainable surface is enabled, and a compact permanent-owner-bridge tensor
directory when the bridge is enabled. Checkpoints MUST NOT save base-model
weights or promise optimizer, scheduler, scaler, RNG, dataloader, iterator, or
sampler resume state. New checkpoints MUST NOT require `checkpoint.json`,
`checkpoint_handoff.json`, readiness manifests, setup receipts, or duplicated
identity graphs to be loadable. The bridge's own tensor manifest is part of
the learned payload contract and MUST NOT be treated as one of those historical
handoff/readiness artifacts.

All ranks SHALL enter checkpoint-save operations in the same order when
Accelerate collectives require it, but only rank zero SHALL materialize the
durable checkpoint directory. Supported distributed saving SHALL be limited to
replicated DDP: all ranks enter a pre-save barrier; rank zero unwraps the model
and writes only the configured adapter into staging using PEFT safe
serialization with embedding-layer saving disabled, then writes the optional
compact selected embedding delta and required compact bridge payload. The
adapter safetensor MUST contain required LoRA A/B and DoRA magnitude-vector
state and MUST NOT contain full embedding, LM-head, base-model, or bridge
tensors. Rank zero MUST atomically commit the step directory only after every
configured learned payload validates and MUST broadcast a bounded success/error
descriptor to every rank. Every rank MUST continue or raise the same named
checkpoint-save error from that collective. Failed staging state MUST be
removed, and aliases MUST update only after successful commit.

#### Scenario: Adapter-only checkpoint is saved

- **WHEN** adapter training reaches a scheduled checkpoint step without
  selected-token embedding or permanent-bridge training
- **THEN** the step directory MUST contain the standard adapter payload needed
  by the inference adapter loader
- **AND** MUST use `adapter_model.safetensors`
- **AND** MUST NOT contain a copy of base-model, full-embedding, LM-head, or
  bridge weights.

#### Scenario: Adapter and selected embeddings are saved

- **WHEN** both adapter and selected-token embeddings are trainable and the
  permanent bridge is disabled
- **THEN** the step directory MUST contain both inference-loadable payload
  directories
- **AND** inference MUST be able to compose them from explicit config paths.

#### Scenario: Permanent bridge is saved

- **WHEN** adapter, selected-token embeddings, and permanent bridge are
  trainable
- **THEN** the step directory MUST contain all three inference-loadable learned
  payloads and their owner-defined validation metadata
- **AND** dynamic HF inference MUST be able to compose them from explicit
  config paths.

#### Scenario: Rank-zero checkpoint save fails

- **WHEN** rank zero fails while saving or validating a distributed checkpoint
- **THEN** every rank MUST observe the same checkpoint-save failure without
  hanging
- **AND** no final/best alias MUST reference the incomplete step
- **AND** no committed partial checkpoint directory may remain.

#### Scenario: Existing checkpoint is used

- **WHEN** an older CoordExp-Swift checkpoint contains a standard adapter and
  optional compatible selected-token embedding payload plus extra historical
  metadata
- **THEN** inference MUST load the configured payload paths
- **AND** MUST NOT require the historical metadata to be regenerated
- **AND** MUST NOT infer that the older checkpoint requires a permanent bridge
  unless its explicit learned-payload identity says so.

## ADDED Requirements

### Requirement: Permanent bridge checkpoint payload

Every bridge-enabled checkpoint SHALL publish a permanent bridge tensor payload and a self-contained bridge manifest alongside the existing DoRA adapter and optional selected-token embedding delta. The bridge manifest MUST bind the base/model identity available to training, tokenizer identity, source adapter/checkpoint identity, bridge architecture profile and schema version, slot/key/value dimensions, layer seam, parameter names and shapes, tensor dtype, payload fingerprint, and the identities of companion adapter and embedding-delta payloads.

Checkpoint publication MUST be atomic at the checkpoint-directory level: a bridge-enabled checkpoint MUST NOT be considered inference-complete unless adapter, selected-embedding payload when configured, bridge payload, and their manifests all validate. `checkpoint-final` MAY remain a training convenience alias, but canonical inference leaves MUST continue to name concrete payload paths.

#### Scenario: Bridge save fails after adapter save
- **WHEN** rank zero cannot finish or validate the bridge payload
- **THEN** the checkpoint MUST NOT be published as inference-complete or selected as the final alias

#### Scenario: Stage 1 final checkpoint
- **WHEN** all four teacher-forced presentations complete safely
- **THEN** the final checkpoint MUST include all permanent bridge composition payloads regardless of natural greedy evaluation quality

### Requirement: Bounded owner-bridge diagnostics

Stage 1 artifacts SHALL record enough bounded evidence to distinguish atom inventory, selection, owner use, coverage transition, and grammar failures without storing unbounded hidden tensors. Required summaries MUST include atom assignment and slot utilization, matched atom box quality, trustworthy candidate counts, ordinary-unmatched masked counts, router uncovered/covered/null mass, aggregate availability by carrier/atom count, selected-atom matched/unknown status where labels permit, admission and row-write RMS ratios, description and geometry swap margins, loss numerators/denominators, and optimizer-group gradient norms. A bounded deterministic sample SHALL retain per-boundary top-k routes and per-row owner-use interventions.

Resolved artifacts MUST record the four presentation identities and seeds, source checkpoint and config lineage, bridge/loss ramps, and the first natural greedy evaluation configuration. Diagnostics are evidence about the saved checkpoint and MUST NOT gate its publication unless they reveal a mechanical contract violation or non-finite unsafe step.

Training-time eval artifacts MUST be emitted at the predetermined planned-step midpoint and end of each of the four presentations for the production run, for eight total landmarks, and MUST label themselves as teacher-forced forward/loss evidence. All eight events MUST use the same fingerprinted `geo_sorted` eval rendering and report the same loss-term numerators, denominators, and bridge summaries as applicable training forwards, without being confused with natural greedy inference evidence.

#### Scenario: Natural recall does not improve
- **WHEN** the Stage 1 checkpoint completes but its first natural greedy evaluation shows no recall gain
- **THEN** the checkpoint and bounded causal diagnostics MUST still be retained as a valid research artifact
- **AND** the run MUST NOT be relabeled as a mechanical failure solely from that model-quality outcome

#### Scenario: Scheduled training-time eval artifact is written
- **WHEN** any declared presentation-midpoint or presentation-end evaluation completes
- **THEN** its artifact MUST identify teacher-forced mode, original scheduled planned-step event id, applied/skipped optimizer outcome for that step, completed-safe-update count, fixed eval data/cache identity, and all applicable loss denominators
