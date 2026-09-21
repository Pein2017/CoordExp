## Why

Frozen-C K4 sampling shows current-policy variation in annotated-owner coverage on 109 of 248 training images, but it does not show that a policy-gradient update changes natural greedy behavior. The shortest useful test is one static, one-step RLOO replay into the existing shared language DoRA, judged first by train-side natural owner coverage and only then by image-disjoint evaluation.

## What Changes

- Add an experiment-local K4 replay materializer that binds existing C-policy trajectories, appends the verified terminal `im_end` action, and computes detached image-normalized leave-one-out advantages.
- Add one production-shaped eight-rank update path that scores the complete sampled actions and applies exactly one globally normalized update to the existing shared language-DoRA surface.
- Persist only an unmerged DoRA adapter plus compact identity, gradient, update, and cold-read receipts.
- Evaluate natural greedy behavior first on the frozen eight-image training panel; a train-positive result may advance to train-248 and image-disjoint monitoring under a separate recorded stage.
- Keep output-QP, PPO/critic, reference-model/KL machinery, hard owner-debt projection, online resampling, and merged checkpoints out of this vertical.

## Capabilities

### New Capabilities

- `static-rloo-trajectory-update`: Experiment-local construction, distributed normalization, one-step update, persistence, and replay checks for static current-policy K4 owner-coverage RLOO.

### Modified Capabilities

None.

## Impact

The change is confined to probe-local research scripts, one focused test, one cold-inference config, and the owning research unit. It reuses the current Hugging Face composition, native replicated DDP, language-DoRA trainable surface, and adapter-only checkpoint writer. It adds no dependency and changes no production default or stable public contract.
