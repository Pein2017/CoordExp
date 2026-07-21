## MODIFIED Requirements

### Requirement: Inference source ownership

The pipeline SHALL use the approved `src/inference/*` module ownership
boundaries. `runtime.py` SHALL own processor-only frontend setup and backend
session launch preparation. `backend.py` SHALL own semantic requests, results,
likelihood pairs, and the session protocol. `hf_backend.py` SHALL own dynamic
HF model composition and execution. `vllm_backend.py` SHALL own offline vLLM
engine execution and live operational diagnostics. `execution_model.py` SHALL
own immutable execution-model materialization and structural receipt
validation; explicit comparison probes MAY own optional composition-fidelity
diagnostics. `prompt.py`, `parsing.py`, `scoring.py`, and `artifacts.py` SHALL
own their corresponding semantic contracts. `pipeline.py` MUST remain
orchestration-only and MUST NOT import training runtime owners.

#### Scenario: Materialized vLLM runtime

- **WHEN** runtime setup launches vLLM
- **THEN** `execution_model.py` resolves a structurally validated immutable
  model before `vllm_backend.py` opens the rank-local engine
- **AND** the pipeline does not require or bind a historical behavioral proof

#### Scenario: No training pipeline import

- **WHEN** inference modules are imported
- **THEN** they do not import training pipeline or training schedule owners

#### Scenario: Runtime assembler

- **WHEN** runtime setup loads a base-plus-adapter checkpoint
- **THEN** it delegates Qwen, adapter, backend-session, and artifact identity
  mechanics to their owner modules rather than reimplementing them inline

#### Scenario: Dynamic HF runtime

- **WHEN** runtime setup loads a base-plus-adapter checkpoint for HF
- **THEN** `hf_backend.py` composes the base, DoRA adapter, and selected-token
  delta directly through their owner modules

### Requirement: Backend launch preparation

The pipeline SHALL resolve shared request evidence before opening a backend and
SHALL obtain backend identity only from the opened session receipt. vLLM
controller mode MUST resolve and publish a structurally validated
execution-model receipt before workers launch. The pipeline MUST NOT load an HF
model on the vLLM worker path or require a historical qualification receipt.

#### Scenario: vLLM launch

- **WHEN** backend type is vLLM
- **THEN** no HF inference model is loaded in the rank worker before the vLLM
  session opens
- **AND** actual engine construction and live decode evidence determine
  operational success
