# sol-xhigh result: HOLD

The initial invocation failed before model open because the old-worktree config
was admitted from the self-rollout worktree cwd. Root authorized one bounded
correction that temporarily changed cwd only during config/session admission.

The correction loaded the frozen model and reached the final post-close guard,
which is after the four independent singleton forwards, two native mixed-B2
forwards, direct-logit checks, and one cell0 language-DoRA autograd forward. It
then exited before artifact persistence because the runner incorrectly required
`torch.cuda.memory_allocated() == 0` inside the still-live Python process. GPU2
was 0 MiB with no process immediately after exit. Since the raw replay evidence
was not persisted, this is **HOLD**, not `candidate` or `lead-accepted`.

No optimizer, parameter update, new generation, second model-open retry, or
scope/tolerance/source change was made.

## Exact commands

Attempted corrected real command (exit 1):

```bash
cd /data/CoordExp/.worktrees/self-rollout-behavior
CUDA_VISIBLE_DEVICES=2 python /data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-a-owner-visual-followup/benchmark/candidates/sol-xhigh/run_real.py --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a-owner-followup-v1/benchmark/sol-xhigh > /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a-owner-followup-v1/benchmark/sol-xhigh/real-run-correction-1.log 2>&1
```

Cold consumer command (documented but not run because `artifact-v1.json` was
not produced):

```bash
cd /data/CoordExp/.worktrees/self-rollout-behavior
CUDA_VISIBLE_DEVICES= python /data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-a-owner-visual-followup/benchmark/candidates/sol-xhigh/verify_artifact.py --artifact /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a-owner-followup-v1/benchmark/sol-xhigh/artifact-v1.json
```
