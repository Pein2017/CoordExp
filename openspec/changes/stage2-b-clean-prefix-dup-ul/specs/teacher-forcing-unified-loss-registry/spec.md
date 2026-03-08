# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical contexts, token types, and loss component names are shared
The canonical registry-level loss component names SHALL include duplicate unlikelihood for the clean-prefix Channel-B contract.

Normative minimum canonical loss component names for this change:
- `struct_ce`
- `desc_ce`
- `duplicate_ul`
- `geo`
- `coord_reg`

#### Scenario: duplicate_ul is a canonical registry loss component name
- **WHEN** the clean-prefix Channel-B objective is reported through the unified registry
- **THEN** the duplicate-unlikelihood component is identified canonically as `duplicate_ul`
- **AND** it is not folded into `struct_ce` or `desc_ce`.

### Requirement: Channel-B rollout context is FP-neutral and EOS-enforced
For Stage-2 Channel-B (`context=rollout`), the rollout-context contract SHALL be defined over the clean accepted sequence rather than the raw rollout prefix.

Normative rollout object subsets:
- `matched_clean`: clean accepted objects matched to GT.
- `unmatched_clean`: clean accepted objects not matched to GT.
- `duplicate`: duplicate-certified objects removed from the positive clean prefix.
- `fn`: GT objects injected into the same top-level `objects[]` container for supervision.

Normative behavior:
- `duplicate` objects MUST NOT appear in the positive teacher-forced prefix.
- `matched_clean` objects receive matched-prefix structure supervision and positive geometry/coord supervision as defined by the Channel-B contract.
- `unmatched_clean` objects MAY remain in the clean prefix as context but MUST remain neutral.
- `fn` objects remain positively supervised.
- Closure / EOS remain supervised.

#### Scenario: Duplicate-certified objects are removed from the positive prefix
- **WHEN** a rollout object is classified as `duplicate`
- **THEN** it does not contribute to the positive teacher-forced prefix
- **AND** it is represented only through duplicate-ul supervision and diagnostics.

### Requirement: Rollout-context semantics are explicit, auditable, and coherent across trainers
The unified loss registry SHALL treat clean-prefix rollout semantics as the canonical Channel-B rollout contract.

Normative behavior:
- Channel-B positive masks are built from the clean teacher-forced target, not the raw rollout prefix.
- Neutral unmatched clean extras MUST stay outside matched-prefix struct masks, coord supervision groups, and duplicate-ul positives.
- Duplicate-ul supervision MUST be boundary-local and explicit rather than encoded through hidden token-ce behavior.

#### Scenario: Neutral unmatched clean extras remain context-only
- **WHEN** a clean accepted object is unmatched after Hungarian
- **THEN** it may remain in the clean prefix as context
- **AND** it contributes no positive CE/geo/coord or duplicate-ul target.

## ADDED Requirements

### Requirement: Duplicate UL is boundary-local and LCP-defined
The unified loss registry SHALL define duplicate unlikelihood as a boundary-local objective over canonical clean vs duplicate continuations.

Normative behavior:
- For each clean boundary `b`, define `clean_continuation(b)` from the canonical clean teacher-forced target.
- For each duplicate attached to boundary `b`, define `duplicate_continuation(b, dup)` as canonical serialization of that duplicate object at boundary `b`, followed by the same canonical clean suffix.
- The target token is the first true divergence token of `duplicate_continuation(b, dup)` relative to `clean_continuation(b)`.
- Duplicate-ul aggregation is one unit term per unique divergence token per boundary.
- If no safe divergence token exists for a continuation, that continuation is skipped and counted in diagnostics.
- This deduplicated-per-boundary aggregation is intentional: the canonical v1 contract does not sum one UL term per duplicate object when multiple duplicates encode the same divergence token at the same clean boundary.

#### Scenario: Same-class-next-object cases do not blindly suppress the first desc token
- **WHEN** a duplicate continuation shares a non-empty token prefix with the clean continuation
- **THEN** duplicate-ul targets the first true LCP-divergence token
- **AND** it does not blindly suppress the first desc token.

#### Scenario: Unsafe or unavailable divergence token is skipped and counted
- **WHEN** a duplicate continuation yields no safe divergence token relative to the clean continuation
- **THEN** duplicate-ul does not contribute a loss term for that continuation
- **AND** the skipped continuation is counted in the corresponding duplicate-ul diagnostics counter.
