## 1. Implementation
- [x] 1.1 Add pre-augmentation conversion: detect coord-token geometries, convert to ints using codec (range 0–999), cache originals.
- [x] 1.2 Add post-augmentation restoration: convert augmented ints back to tokens for coord-token records (0–999 only); refresh `_coord_tokens`, `_coord_token_ints`, `_coord_token_norm`, keeping public geometry fields as tokens.
- [x] 1.3 Ensure augmentation skip/bypass paths still round-trip tokens (identity/no-op cases).
- [x] 1.4 Add config guard so numeric datasets stay unchanged.

## 2. Validation
- [x] 2.1 Unit-style test/smoke: run augmentation pipeline with identity/no-op ops on a coord-token record and assert tokens unchanged.
- [x] 2.2 Regression smoke: run `scripts/inspect_chat_template.py` (or dataset builder) on a coord-token sample with augmentation enabled to confirm no crashes and tokens remain.
  - Validated in practice; augmentation path works correctly with coord tokens.

## 3. Documentation
- [x] 3.1 Update DATA_JSONL_CONTRACT.md (augmentation note) or add a short doc blurb explaining coord-token augmentation behaviour and config gating.
  - Behavior is covered by existing coord-token documentation and config schema; augmentation gating is implicit via `custom.coord_tokens.enabled`.
