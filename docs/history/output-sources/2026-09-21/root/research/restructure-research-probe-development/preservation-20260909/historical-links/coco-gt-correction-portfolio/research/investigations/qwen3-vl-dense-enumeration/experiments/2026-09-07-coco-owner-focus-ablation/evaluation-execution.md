# Evaluation execution candidate

Status: Source launched; candidates and reduction **not launched**. No scientific result.
Authority: unit.md and evaluation-qualification.md; lead's execution allocation and holdout-first ruling supersede the illustrative train-first loop.

## Durable Source invocation

Started 2026-09-07T14:25:52Z, tmux `coco-owner-focus-source-eval-v1`.
Driver PID 2061168; timeout PID 2061178; Python controller 2061197;
native workers 2061400 / 2061401, rank 0 / 1, parent-visible devices 2 / 3.
GPU2/3 had no compute processes before launch; training remained on 0/1,4/5,6/7.
Source output roots and launch log root were absent before launch.

`focus-v1/launch-source-evaluation/` owns driver log/PID/start/finish/exit,
per-panel log/timeout PID/start/finish/exit, launcher SHA256 and checked fixed
code/input SHA256 manifest. Final markers are
`FOCUS_SOURCE_EVALUATION_COMPLETED` / `FOCUS_SOURCE_EVALUATION_FAILED`.
Holdout512 runs first, train256 only after holdout exits zero. Per-panel 12h
timeout is a safety ceiling, not a forecast. No retries.

Observed generated config: FP32 SDPA, batch4, cap3084, RP1; actual shard plan
has two active ranks and 128 decode batches for holdout512. Both worker
processes were observed; checkpoint-loading output exists. This proves startup
only, not completion or model quality.

## Fixed launchers

| File | SHA256 |
|---|---|
| run-source-evaluation.sh | c7dd0ad840276e9d8f7234f27808d817e2c9dc2434a2e9422723153a4e9ff963 |
| run-candidate-evaluation.sh | a3de036cfc02131bfa8c4767162931ab8b63c42a730a1968027e8ecf4f9bff03 |
| run-reduction.sh | 67b12826d9bf96bae73f8170aaeff84d56844aa07c3355feb256072fbf199eb3 |

Candidate launch is a separate lead action after accepted terminal update-64
checkpoints and fresh GPU reconciliation. Its explicit acknowledgement flag
does not perform or replace scientific/checkpoint acceptance:

```bash
cd /data/CoordExp/.worktrees/coco-gt-correction-portfolio
UNIT=research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation
tmux new-session -d -s coco-owner-focus-candidate-eval-v1 "bash $PWD/$UNIT/run-candidate-evaluation.sh --lead-accepted-update64"
# After all eight panels finish; does not wait or launch decode:
bash "$UNIT/run-reduction.sh"
```

Candidates use R=0,1 M=4,5 Rweak=6,7 concurrently; each holds holdout-before-train,
records panel exits and stops its own failed arm. The controller waits for all
three arm exits. Atomic new log-directory creation prevents controller reuse;
existing arm evaluation roots are rejected. The separate reducer prevalidates
all eight panels through existing frozen-row and topology/completeness guards
before any detection output, then calls the qualified fixed reducer.

## Compact verification and falsification

```bash
cd /data/CoordExp/.worktrees/coco-gt-correction-portfolio
UNIT=research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation
ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1
for f in run-source-evaluation.sh run-candidate-evaluation.sh run-reduction.sh; do bash -n "$UNIT/$f" || exit; done
sha256sum -c "$ROOT/launch-source-evaluation/fixed-sha256.txt"
sha256sum -c "$ROOT/launch-source-evaluation/launcher-sha256.txt"
tmux list-panes -t coco-owner-focus-source-eval-v1 -F '#{pane_pid} #{pane_current_command}'
cat "$ROOT/launch-source-evaluation/startup-processes.txt"
```

Fresh checks passed: all three bash parses; six frozen code/input SHA256
comparisons; resolved configuration and two-rank plan; git diff --check.
Sensitivity checks: repeat Source launcher rejected by existing log root;
candidate launcher without explicit acceptance rejected; premature reducer
rejected missing raw Source holdout artifact before creating detection outputs.
No shared code, inputs, checkpoints, training processes, or prior artifacts edited.

First explicit clock observation 14:23:50 UTC; candidate receipt at 14:27:38 UTC:
observed interval 3m48s, a lower bound, **not exact spawn-to-candidate timing**.
No substantive correction or lead intervention required. One tool-hook read
rejection was retried; one initial brace-expansion read was corrected. Lead
confirmed exact spawn timestamp unavailable; no history scan was added.
This is a real bounded execution sample, not an Astra/Sol ranking or cost claim.
