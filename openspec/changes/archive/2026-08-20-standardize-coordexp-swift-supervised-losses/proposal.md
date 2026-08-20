## Why

The current supervised-loss surface treats coordinate Gaussian/RPS as protected and allows supported configs to encode several token-gate variants, so “baseline SFT” does not yet identify one fail-closed objective contract. Standardizing the protected baseline and zero-weight behavior now makes objective correctness, ablation interpretation, distributed normalization, and telemetry comparable before further training-orchestration work builds on them.

## What Changes

- Sequence this change strictly after `reconcile-coordexp-swift-training-contracts`
  and `decompose-coordexp-swift-training-orchestration` are implementation-
  complete, synced, and archived at recorded commits. Consume the decomposed
  owner graph without moving cache-determinant ownership again.
- **BREAKING** Require every supported supervised config to declare protected base CE at weight exactly `1.0`; omission, zero, or any other value is invalid.
- **BREAKING** Bind the protected token-type gate to exactly `desc_text`, `schema`, `coordinate`, and `eos`. Its `enabled` mode requires weight exactly `0.1`; an explicit `zero_weight_ablation` mode requires weight `0` and still produces finite detached diagnostics without contributing autograd edges.
- **BREAKING** Move `coord_gaussian_rps` out of `losses.protected` into a typed `losses.auxiliary` surface. A zero-weight optional loss is fully omitted from objective computation, denominator construction, metrics, and finite checks rather than being treated as an enabled diagnostic term.
- Introduce one closed internal `TokenLossBinding` description for each implemented token loss, carrying its role, normalization policy, and zero-weight policy. Keep this an implementation-owned typed seam: public config will not accept import paths, arbitrary callables, or a dynamic registry of loss implementations.
- Preserve fp32 objective math, full planned-step `segment_balanced` normalization, and world-size-correct compensation for Accelerate/DDP mean-gradient reduction.
- **BREAKING** Replace ambiguous per-term loss metric names with explicit raw and weighted fields for every computed loss term, with matching finite/count diagnostics; keep the weighted sum as the optimized total and do not dual-write legacy aliases.
- Deliberately migrate current supported production, smoke, and measurement configs to the new strict schema. Historical configs and completed artifacts remain immutable historical evidence and are not rewritten to appear compatible.
- Limit this change to supervised training and forward eval. RL objective composition, rollout-derived losses, hidden-state losses, and a general loss-plugin API remain deferred.
- Treat the predecessor's single cache transition as final for this sequence:
  record the admitted train/eval determinant payloads and hashes before work,
  and stop if this change alters either hash. This change MUST NOT authorize or
  perform a second production cache materialization.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-swift-supervision-losses`: Strengthen the protected SFT baseline, define distinct protected-ablation and optional-disabled zero policies, formalize the closed typed binding seam, and require raw plus weighted per-term telemetry while preserving fp32 and distributed planned-step normalization semantics.
- `coordexp-swift-config-runtime`: Make the canonical supervised loss weights and token groups fail-closed, move coordinate Gaussian/RPS to a typed auxiliary config surface, and reject legacy protected-placement or arbitrary loss implementation hooks in supported configs.
- `coordexp-swift-training-artifacts`: Define the canonical train/eval logging representation for raw and weighted computed loss terms and omit disabled optional-term fields consistently.

## Impact

- Affected source owners include `src/config/models.py`, `src/losses/`, the supervised train/eval reduction paths, and the rank-zero logging projection.
- Supported files under `configs/coordexp_swift/prod/`, `configs/coordexp_swift/smoke/`, and current measurement/config routes require an intentional schema migration; configs with token-gate weights such as `0.2` or `0.25`, or with `coord_gaussian_rps` under `protected`, will fail strict validation until migrated.
- Loss, config, trainer/eval, distributed-reduction, and artifact-schema tests must distinguish the active baseline, explicit gate ablation, and omitted optional terms, including unequal-rank denominator cases.
- Existing historical config trees, archived OpenSpec changes, and completed run artifacts remain provenance only. This change does not promise compatibility aliases for their old loss layout.
- There is no intended change to data, template, token supervision, packing, pack-cache payload semantics, model forward precision, optimizer ordering, or the Accelerate-only replicated-DDP backend.
