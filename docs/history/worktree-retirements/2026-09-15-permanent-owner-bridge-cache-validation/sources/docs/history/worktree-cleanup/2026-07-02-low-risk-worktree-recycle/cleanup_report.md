# Low-Risk Worktree Recycle Record

Date: 2026-07-02

This record preserves text evidence before recycling local experimental worktrees. It is historical provenance, not current behavior authority. Current behavior remains governed by `docs/` and stable contracts in `openspec/specs/`.

Cleanup principle: propose in a worktree, train/eval/infer when useful, record failures or inconclusive results locally, then recycle the stale worktree and branch instead of letting unmerged experiment code accumulate.

## Scope

The cleanup scope is the user-approved low-risk set plus `segment-aware-packing-infra`. Active/dirty lanes such as CoordExp-Swift, coverage-ledger mechanistic probing, ledger auxiliary loss, prefix denoising, and Gaussian-RPS mechanistic round are intentionally excluded.

## Summary Table

| Worktree | Branch | Head | main...branch | git cherry + | Size | Text rows | Snapshots | Local artifacts | Outcome note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `autoregressive-binding-template-study` | `codex/autoregressive-binding-template-study` | `e026b290e1` | `28/388` | `388` | `171M` | 98 | 98 | 2 | Retired after research synthesis moved to research/investigations/autoregressive-binding-template-study on main. Local smoke artifact was a contrast-panel/case-selection probe, not a promoted training result. |
| `coord-repel-conservative-design` | `codex/coord-repel-conservative-design` | `fc73fbfdf1` | `46/18` | `18` | `140M` | 33 | 32 | 0 | Retired as a conservative Stage-1 design/prototype lane. Branch-local decision note is preserved as historical design evidence; no current runtime contract is promoted. |
| `fully-compact-2x2-ablation` | `codex/fully-compact-2x2-ablation` | `45945fd15c` | `47/9` | `8` | `132M` | 29 | 28 | 0 | Retired packed Gaussian SFT retrain lane. The preserved handoff says production training was not launched after the scaling fix and grammar-constrained eval should be treated as legacy/unreliable. |
| `loss-only-instance-enumeration` | `codex/loss-only-instance-enumeration` | `eb6871b949` | `54/68` | `68` | `432M` | 63 | 62 | 0 | Retired experiment lane for loss-only enumeration plus row coverage/packing directions. Text is preserved as historical idea evidence, not current guidance. |
| `mechanistic-diagnosis-experiments` | `codex/mechanistic-diagnosis-experiments` | `9d66860749` | `54/251` | `251` | `213M` | 75 | 70 | 0 | Retired old diagnostics branch after the useful June diagnostics were already consolidated into main-side progress/research surfaces. |
| `row-conditioned-visual-coverage` | `codex/row-conditioned-visual-coverage` | `1629d7fd8e` | `54/45` | `45` | `5.4G` | 47 | 46 | 10 | Retired row-conditioned coverage prototype. Local temp reports are smoke-only/no-claim, with zero mechanism rows and no production rollout launched. |
| `segment-aware-packing-infra` | `codex/segment-aware-packing-infra` | `e8f9bb8e1a` | `54/51` | `51` | `136M` | 42 | 40 | 0 | Retired segment-aware packing prototype per cleanup decision. Design text is preserved as historical architecture evidence; code is not promoted from this stale branch. |
| `compact-template-field-order-ablation` | `codex/compact-template-field-order-ablation` | `4db33dbbe4` | `33/0` | `0` | `0` | 0 | 0 | 0 | Loose branch only; merged into main with no unique text/code delta. Safe local branch cleanup. |

## Artifact Notes

- `row-conditioned-visual-coverage` local smoke reports record `claim status: smoke/no-claim`, `mechanism rows: 0`, and `runtime_note: smoke-only placeholder; no production rollout launched`. This is treated as inconclusive/unsatisfactory evidence for carrying the branch forward.
- `autoregressive-binding-template-study` local contrast-panel smoke summary records `model_perturbation_ran: false`. The durable value has already been synthesized into `research/investigations/autoregressive-binding-template-study/` on main, so the worktree is recycled.
- `fully-compact-2x2-ablation` preserved handoff records that production training was not launched after the packed Gaussian SFT scaling fix; later current Gaussian/RPS work on main supersedes this branch as an implementation base.
- Branches without local temp artifact snapshots are preserved through branch-local Markdown/config text snapshots and cleanup rationale only; no current performance claim is promoted.

## Preserved Files

- Branch text manifest: `manifest.tsv`
- Local artifact manifest: `local_artifacts.tsv`
- Branch text snapshots: `snapshots/<worktree-slug>/...`
- Local temp summaries/reports: `local-artifacts/<worktree-slug>/...`

## Cleanup Decision

After this preservation bundle exists and validates, the listed worktrees may be removed and their local branches deleted. Remote branches, if any, are not deleted by this local cleanup record.
