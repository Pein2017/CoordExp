## 1. Modify _ForceEosOnRepeatGuard to support per-sample max_object_keys
- [ ] 1.1 Change guard constructor to accept optional per-sample max_objects parameter
- [ ] 1.2 Update _should_force_eos_for_seq method to use per-sample limits
- [ ] 1.3 Maintain backward compatibility with global max_object_keys

## 2. Update rollout generation logic
- [ ] 2.1 Extract ground truth object count from sample in _rollout_one method
- [ ] 2.2 Calculate dynamic cap: max_pred_objects = round(1.5 * gt_objects)
- [ ] 2.3 Pass dynamic cap to _ForceEosOnRepeatGuard initialization

## 3. Add configuration support
- [ ] 3.1 Add YAML parameter for dynamic cap multiplier (default 1.5)
- [ ] 3.2 Update config documentation and comments

## 4. Testing and validation
- [ ] 4.1 Test that dynamic cap prevents excessive object generation
- [ ] 4.2 Verify backward compatibility with existing configs
- [ ] 4.3 Monitor training metrics for improved packing efficiency