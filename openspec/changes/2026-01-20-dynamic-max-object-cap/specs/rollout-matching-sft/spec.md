## ADDED Requirements
### Requirement: Dynamic object generation cap prevents hallucination
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL dynamically limit the maximum number of predicted objects based on ground truth complexity to prevent excessive hallucination.

The trainer SHALL:
- Count ground truth objects from the `assistant_payload` using the `object_\d+` key pattern
- Calculate `max_predicted_objects = round(multiplier * num_ground_truth_objects)` where `multiplier` defaults to 1.5
- Count predicted objects during generation by detecting `object_\d+` keys in the JSON response text
- Apply this limit during rollout generation using the `_ForceEosOnRepeatGuard`
- Stop generation early when the predicted object count exceeds this dynamic cap
- Maintain backward compatibility by allowing static `max_object_keys` configuration

Configuration MUST be YAML-driven under `custom.extra.rollout_matching.generation`:
- `max_predicted_objects_multiplier` MUST accept a float and MUST default to `1.5`
- When set, this overrides the static `max_object_keys` limit for dynamic calculation

#### Scenario: Dynamic cap prevents excessive object generation
- **GIVEN** a sample with 5 ground truth objects and `max_predicted_objects_multiplier: 1.5`
- **WHEN** rollout generation begins
- **THEN** the maximum allowed predicted objects is set to `round(1.5 * 5) = 8`
- **AND** generation stops early if the model attempts to generate more than 8 `object_\d+` keys
- **AND** this prevents the model from generating 71 objects when only 5 exist in ground truth

#### Scenario: Backward compatibility with static limits
- **GIVEN** rollout-matching training with `max_object_keys: 10` configured
- **AND** `max_predicted_objects_multiplier` is not set
- **WHEN** rollout generation runs
- **THEN** the static limit of 10 objects is used
- **AND** behavior matches the previous implementation

#### Scenario: Dynamic cap improves training efficiency
- **GIVEN** samples that previously generated excessive objects causing 98% gating rejections
- **WHEN** dynamic capping is enabled
- **THEN** object generation is limited to reasonable multiples of ground truth count
- **AND** gating rejection rates decrease
- **AND** packing efficiency improves due to shorter effective sequences