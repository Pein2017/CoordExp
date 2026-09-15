# B: same-covered-set history stress — candidate complete

Status: implementation and bounded fixed-panel execution complete; root owns
acceptance. No training or extension was launched.

## Observation

All 16 frozen images remain in the artifact. Twelve prefixes had at least two
valid complete rows; four were explicitly ineligible (287484, 291918, 313924,
551023). Only the final two complete row token spans were swapped; intervening
tokens, row token contents/multiset and total prefix length were preserved.

| Eligible12 metric | Native reference | Swapped history |
|---|---:|---:|
| Annotated owners recovered / denominator | 83 / 110 | 83 / 110 |
| Remaining owners realized / fixed remaining denominator | 44 / 71 | 44 / 71 |
| Predictions / annotation-relative unmatched | 123 / 40 | 121 / 38 |
| Later geometry repeats, pixel IoU strictly >0.95 | 0 | 0 |
| EOS / caps | 12 / 0 | 12 / 0 |
| Parser drops | 0 | 0 |
| Whole generated token count, including prefix | 1202 | 1183 |

The prefix covered 39 owners; no prefix-covered owner was lost in either final
matching. No local equal-IoU midpoint matching ambiguity was detected. The
full16 native reference is 90/124 owners; no treatment outcome is imputed for
the four ineligible images.

Three owners were gained and three lost, across four images:

| Image | Delta TP | Gained owner IDs | Lost owner IDs |
|---|---:|---|---|
| 323322 | 0 | 1608021 | 1956103 |
| 345388 | +1 | 1187893 | — |
| 486123 | +1 | 1783860 | — |
| 532132 | -2 | — | 2099224, 2118217 |

Nine of twelve exact suffix token sequences changed; three were identical.
Thus aggregate zero does not mean every continuation or owner set was stable.

## Interpretation and stop

This particular legal off-path order stress changes access to some annotated
owners, in both directions, without aggregate recovery improvement in this
fixed panel. It neither establishes an order-invariance requirement nor
identifies missing memory: learned ordering/position priors and ordinary
continuation sensitivity remain plausible explanations. The panel is small
and previously selected training-side data, not representative confirmation.

Do not prioritize a new state module or claim a deployed-model gain from this
result. A bounded architecture-preserving recovery-training option is recorded
in `recovery-training-proposal.md`, but is not justified as the next automatic
launch. Close this stress test at the authorized fixed panel.

## Mechanics and reproducible acceptance

- Sole native baseline consumed from A, SHA256
  `0117a01b7d2b4b28489d18b6d71120af5f77053221cd732eaff217d1c177c077`.
- Input manifest SHA256
  `b680f78e6fb8ded1ebc3b5c089aef62d8a42a1e3c839df853655f3c7be838261`.
  The absolute-path source16 JSONL is audit serialization, not a supported
  native-loader input. Runtime uses its hash-bound canonical raw_train256,
  selects the same16 IDs, and verifies each original row and media hash.
- Existing HF session/native exact-prefix helper, Source step-2444 DoRA and
  selected embeddings, FP32/SDPA, native greedy RP1.10, whole cap3084. No
  source/config/checkpoint changes; 72 imported source-module hashes checked.
- First eligible image81589 qualified exact unperturbed prefix replay against
  the sealed native trajectory and parser output. The qualification and all
  final rows were persisted/reloaded before acceptance. All12 swaps pass exact
  prefix tokens, semantic multiset and prompt/image-grid identity checks.
- One technical attempt, PID3626629, shell exit0; GPU3 runtime including load,
  one reference replay, and12 swapped continuations: **70.514 seconds**.
  GPU3 observed released afterward. No second attempt or training.
- Ten CPU tests pass, including mutation rejection, span/gap preservation,
  one-to-one owner accounting, strict0.95 repeat boundary and GT identity.

From the new worktree, replay:

```bash
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-self-rollout-behavior-portfolio/b/verify.py
python -m pytest -q research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-self-rollout-behavior-portfolio/b/test_run.py research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-self-rollout-behavior-portfolio/b/test_reduce.py
```

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/b`.
Decision artifact `results.json` SHA256
`51893fed4e648ac3c5666fdf20c7697791000acc1be2f1b8642ee3b18e0ad3bb`;
raw `attempt-1/outcomes.jsonl` SHA256
`27b195170a227d9cf7b1a6f3526de4c3d6b4d813bd2ac1c45fea73f593876931`.
`verification.json` holds the fresh reducer/pin/identity replay; attempt1 also
contains launch, qualification, runtime-dependency and terminal receipts.
