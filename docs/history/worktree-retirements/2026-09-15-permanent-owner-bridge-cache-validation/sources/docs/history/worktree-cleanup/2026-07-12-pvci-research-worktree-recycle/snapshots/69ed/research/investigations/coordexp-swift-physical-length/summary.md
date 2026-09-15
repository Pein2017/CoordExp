# CoordExp-Swift Physical-Length Isolation Summary

Evidence scope: `tiered smoke`.

## Runs

| run | length | steps | status | boundary | finite | step1 loss | step1 top1 | step1 examples | qwen mean pack | qwen mean max segment |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| physical_length_12k_ebs64_1step_limit1024 | 12000 | 1 | completed | True | True | 5.05219 | 0.317666 | 64 | 11081.9 | 1468.62 |
| physical_length_6k_ebs64_1step_limit1024 | 6000 | 1 | completed | True | True | 5.01333 | 0.325691 | 32 | 5673.12 | 1521.62 |

## Comparisons

### 1-step pair

Verdict: `unresolved_count_confounded`.

- Boundary pass: `True`
- Runtime pass: `True`
- Physical length materially changed: `True`
- Counts match: `False`
- Metric pass: `False`

| field | 12k - 6k |
| --- | ---: |
| loss/total | 0.038861 |
| loss/base_ce | 0.038861 |
| acc_top1 | -0.00802571 |
| acc_top5 | -0.01197 |
| count/examples | 32 |
| count/packs | 0 |
| count/supervised_atoms | 1039 |
| count/eligible_segments | 32 |
| count/skipped_segments | 0 |

## Handles

- `physical_length_12k_ebs64_1step_limit1024`
  - config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/length_isolation/physical_length_12k_ebs64_1step.yaml`
  - run: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/research/coordexp_swift/length_isolation/physical_length_12k_ebs64_1step_limit1024`
- `physical_length_6k_ebs64_1step_limit1024`
  - config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/length_isolation/physical_length_6k_ebs64_1step.yaml`
  - run: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/research/coordexp_swift/length_isolation/physical_length_6k_ebs64_1step_limit1024`

## Interpretation

- `pure_physical_length_eliminated_smoke_scope` means the pair changed physical packed-row length, preserved boundary/runtime/count gates, and stayed inside the metric tolerances.
- `unresolved_count_confounded` means physical row length changed, but examples or supervised atoms per compared step also changed, so the run cannot isolate pure length by itself.
- This report is smoke evidence only; it is not a full validation or benchmark claim.

## Recommended Next Action

- Do not run the planned 2-step confirmation for this pair, because the 1-step gate did not isolate pure physical length.
- If a cleaner isolation is needed, add a separate count-matched follow-up arm, for example by increasing the 6k pack presentations so examples and supervised atoms match before comparing precision-sensitive metrics.
