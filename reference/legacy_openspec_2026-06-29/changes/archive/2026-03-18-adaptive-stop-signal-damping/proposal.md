## Why

Current dense object-list SFT in Stage-1 and Stage-2 Channel-A can become overly conservative when training targets omit real objects: standard CE keeps pushing the model toward the serialized stop boundary even when continuation is still plausible. Recent inspection of the real JSONL -> builder -> chat-template -> tokenizer path shows that the decisive stop branch is the first terminal `']}'` token contrasted against the non-final `']},'` token, so treating the problem as generic EOS pressure is too coarse for a reproducible experiment.

## What Changes

- Introduce an optional research capability that dampens stop-token CE at the branch-local "continue another object vs stop after this object" decision using a continuous weight derived from the model's own `']},'` vs `']}'` preference.
- Define the canonical dense-serialization stop locator from the real tokenizer/chat-template path: the first terminal `']}'` after the final object is the semantic stop branch, while the later closing `']}'` and `<|im_end|>` remain secondary closure signals.
- Add config-first training surfaces for enabling the adaptive stop-signal experiment in teacher-forcing code paths without introducing new CLI flags or changing default training behavior.
- Add auditable branch-local diagnostics for margin, damping weight, and stop-vs-continue preference, while reusing the existing rollout/eval metric families to judge downstream parse validity, recall, and false-positive drift.
- Preserve existing geometry, object ordering, and Qwen3-VL chat-template compatibility; this change only adjusts how ending supervision is weighted when the experiment is enabled.

## Capabilities

### New Capabilities
- `adaptive-stop-signal-damping`: Optional branch-local ending-signal damping for dense CoordJSON teacher forcing, including canonical stop-branch identification, continuous weighting, and reproducible experiment controls.

### Modified Capabilities
- `teacher-forcing-objective-pipeline`: Extend the objective pipeline contract so teacher-forcing can enable a config-validated stop-signal damping objective or sub-objective with deterministic branch-token targeting and fail-fast prerequisite checks.
- `teacher-forcing-unified-loss-registry`: Clarify the canonical stop/continue supervision contract so the first terminal `']}'` vs `']},'` branch is modeled explicitly instead of being treated as undifferentiated structure/EOS pressure.
- `trainer-metrics-components`: Add canonical diagnostics for stop-branch behavior and document how branch-local damping metrics are emitted and interpreted alongside existing rollout/eval monitors.

## Impact

- Affected code is expected in the teacher-forcing loss path and branch-location utilities, especially [rollout_meta.py](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/src/trainers/teacher_forcing/rollout_meta.py), [token_ce.py](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/src/trainers/teacher_forcing/modules/token_ce.py), and adjacent config/dataclass surfaces that define objective-module behavior.
- Affected configs are expected under [configs/stage1](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/configs/stage1) and [configs/stage2_two_channel](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/configs/stage2_two_channel) so the experiment remains YAML-first and opt-in.
- Affected docs/specs include new OpenSpec artifacts for `adaptive-stop-signal-damping` plus delta specs for the modified teacher-forcing and metrics capabilities; training docs such as [METRICS.md](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/docs/training/METRICS.md) and [STAGE2_RUNBOOK.md](/data/home/xiaoyan/AIteam/data/CoordExp/.worktrees/adaptive-stop-signal-damping/docs/training/STAGE2_RUNBOOK.md) will need to reflect the new diagnostics and interpretation.
- No upstream HF model files, tokenizer vocab files, or dataset geometry contracts should change as part of this proposal.
