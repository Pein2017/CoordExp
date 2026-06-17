## ADDED Requirements

### Requirement: Static packing cache identity includes compact row axes and length
Static packing fingerprints SHALL include compact row serialization identity and
the global packing length.

For compact Stage-1 SFT with static packing, the packing fingerprint MUST include
at minimum:

- resolved detection template id,
- resolved object field order,
- prompt hash,
- object instance ordering policy,
- tokenizer/chat-template identity,
- `global_max_length` or effective packing length.

#### Scenario: Static packing cache changes with bbox-first
- **GIVEN** two compact Stage-1 SFT configs that differ only by
  `custom.object_field_order`
- **WHEN** static packing cache fingerprints are computed
- **THEN** the fingerprints differ.

#### Scenario: Static packing cache records 12000 global length
- **GIVEN** a final ablation Stage-1 SFT config
- **WHEN** static packing cache identity is materialized
- **THEN** the effective packing length is recorded as `12000`.
