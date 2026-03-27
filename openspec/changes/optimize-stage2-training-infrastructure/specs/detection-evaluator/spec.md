# detection-evaluator Specification (delta: rank-efficient train-time DDP integration)

## Purpose
Extend the detection evaluator integration contract so train-time DDP evaluation remains rank-efficient while preserving the current train-time summary-metric contract.

## Requirements

## ADDED Requirements

### Requirement: Train-time DDP detection evaluation MUST avoid all-rank heavy payload materialization when rank 0 owns final scoring
When detection evaluation is invoked from training under DDP and only rank 0 computes the final evaluation outputs, non-zero ranks MUST NOT be required to materialize the full combined evaluation payload solely to support rank-0 scoring.

Normative behavior:
- The train-time evaluation path MAY use shard-first or scalar-first coordination.
- Rank 0 MUST remain the owner of final heavy evaluation work and final train-time summary emission when the integration is rank-0-owned.
- Any rank-efficient transport optimization MUST preserve final metric meaning and the supported train-time summary outputs for that integration.
- This requirement does NOT broaden train-time evaluation into full offline evaluator artifact parity.

#### Scenario: DDP train-time eval keeps heavy final scoring on rank 0
- **GIVEN** training invokes detection evaluation under DDP
- **AND** rank 0 is the owner of final evaluation outputs
- **WHEN** predictions and GT data are coordinated across ranks
- **THEN** non-zero ranks are not required to materialize the full combined payload solely for rank-0 scoring
- **AND** rank 0 still produces the canonical train-time summary outputs for that integration.
