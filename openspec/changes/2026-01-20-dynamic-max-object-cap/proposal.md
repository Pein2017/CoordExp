# Change: Dynamic Max Object Cap for Rollout Generation

## Why
The current rollout-matching training suffers from severe hallucination issues where the model generates vastly more objects than ground truth (e.g., 71 predicted objects vs 5 ground truth objects). This causes 98% gating rejection rates, wastes computational resources, and prevents efficient packing. The static `max_object_keys: 128` limit is too high and doesn't adapt to the actual complexity of each sample.

## What Changes
- Add dynamic object limit: `max_predicted_objects = round(1.5 * num_ground_truth_objects)`
- Modify `_ForceEosOnRepeatGuard` to accept per-sample max object limits
- Update rollout generation to calculate dynamic caps using ground truth object counts
- Prevent excessive object generation that leads to hallucination and packing inefficiency

## Impact
- Affected specs: `rollout-matching-sft`
- Affected code: `_ForceEosOnRepeatGuard` class, rollout generation logic
- Benefits: Reduced hallucination, better packing efficiency, faster training convergence
- Breaking changes: None (adds optional configuration, maintains backward compatibility)