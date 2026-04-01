# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical contexts, token types, and loss component names are shared
The unified loss registry SHALL preserve the existing token-type partition when
proxy-supervision weighting is enabled.

Normative behavior:

- structure tokens continue to include JSON syntax, punctuation, quotes, and
  field-name tokens such as `"desc"` and `"bbox_2d"`,
- free-text tokens inside an object's `desc` value span remain `type=desc`,
- coord-vocabulary tokens remain `type=coord`,
- object-local proxy weights MUST NOT reduce or disable `struct_ce`,
- object-local `desc_ce_weight` MAY scale only desc-value supervision for the
  aligned object,
- object-local `coord_weight` MAY scale only bbox-geometry and coord-side
  supervision for the aligned object.
- object-local `coord_weight` MAY be `0.0` for cue-only proxies whose
  objectness signal is useful but whose box extent is not trustworthy enough for
  geometry supervision.

#### Scenario: Proxy weighting does not leak onto structure tokens
- **WHEN** proxy-supervision weighting is enabled for a plausible object
- **THEN** the field-name tokens `"desc"` and `"bbox_2d"` remain fully
  supervised under `struct_ce`
- **AND** only the desc-value tokens and bbox/coord supervision for that object
  use lowered proxy weights.

### Requirement: Canonical loss scalars are mean-like and scale-invariant
The unified loss registry SHALL keep desc and coord families mean-like when
object-local proxy weights are applied.

Normative behavior:

- `desc_ce` remains a weighted mean over supervised desc-value tokens,
- `geo` remains a weighted mean over supervised bbox groups,
- `coord_reg` sub-terms remain weighted means over contributing coord slots,
- object-local proxy weights MUST change the contributing numerators and
  denominators consistently rather than introducing raw unnormalized sums.

#### Scenario: Proxy-weighted desc_ce remains mean-like
- **WHEN** a batch contains a mix of `real`, `strict`, and `plausible` objects
- **THEN** the reported `desc_ce` remains comparable as a weighted mean-like
  scalar
- **AND** it does not scale only because more plausible objects were present.
