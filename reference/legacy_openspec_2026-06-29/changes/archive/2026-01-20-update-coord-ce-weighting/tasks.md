## 1. Implementation
- [x] 1.1 Attach per-token `loss_scale` in the dataset metrics collator when coord loss is enabled and CE weights differ
- [x] 1.2 Ensure loss_scale weights: 0 for `labels == -100`, coord weight for coord-token labels, non-coord weight otherwise
- [x] 1.3 Skip manual coord/text CE recomputation when loss_scale-based weighting is active to avoid double weighting
- [x] 1.4 Keep coord-only aux losses unchanged and confirm metrics remain unweighted via loss_scale un-scaling
- [x] 1.5 Update docs or notes describing the loss_scale weighting behavior and the Liger-kernel tradeoff

## 2. Validation
NOTE: This section is operational validation guidance and may require training datasets / GPUs. It is intentionally
NOT tracked as OpenSpec tasks (i.e., no checkboxes) so it doesn't block archiving.

- 2.1 Run a small packed smoke config with coord/text CE weights != 1 and confirm loss_scale is populated
- 2.2 Verify coord/text CE metrics remain finite and aux losses still compute on coord tokens only
