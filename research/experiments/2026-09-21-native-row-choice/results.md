# Native row-choice witness: worker candidate

Status: `candidate`; lead scientific acceptance pending.

## Decision-bearing result

No frozen credible unvisited-owner N row beats the actual greedy repeated row in any of the three admitted mature-model native states. This is a negative finite-candidate diagnostic, unresolved outside the tested rows. It does not prove that the actual row is global row MAP, that another repeated row would not score higher, or that owner-level probability favors recurrence. Because there was no positive witness, the optional supplied-N continuation was not run.

| Native state | Model | Common description | Actual repeated log p | Best tested N owner | Best N log p | N−actual |
|---|---|---|---:|---|---:|---:|
| tied-14038-first-revisit | tied | book | -12.613591 | annotation:-127 | -15.259282 | -2.645691 |
| tied-885-first-revisit | tied | person | -9.737416 | annotation:439117 | -15.395899 | -5.658483 |
| untied-885-first-revisit | untied | person | -8.630569 | annotation:439117 | -14.818982 | -6.188413 |

Finite tested-set log masses are:

| State | A mass | C mass | N mass |
|---|---:|---:|---:|
| tied-14038 | -12.030411 | -13.134339 | -15.024295 |
| tied-885 | -9.011177 | absent | -15.341442 |
| untied-885 | -7.936922 | absent | -14.706502 |

## Denominators and accounting

- Frozen source trajectories: 11; admitted states: 3; identity `HOLD`: 8.
- Completed/analyzable states: 3/3; positive local witnesses: 0/3.
- Each row score uses original full-vocabulary log softmax and includes the row opener, description, four coordinate tokens and terminator. There is no candidate-coordinate renormalization or beam search.
- Actual-row full-trace top-two maximum errors are `5.340576171875e-05`, `3.4332275390625e-05`, and `4.1961669921875e-05`, each below `2e-4`.
- The saved validator rejects both a dropped opener and a dropped terminator.
- The first tied-885 attempt completed seven forwards but its reducer incorrectly required optional C; it is retained as `technical_invalid`. The repaired state and the other two states are distinct immutable outputs.

The tied-14038 qualification producer SHA is `25378ca2545effde282eea62d920e0870090731f6e04f1d89a2310b135c5fcb0`. Producer `0d140ff5eefe9ec603e6309e28ac4b5c7c4c97bcf41df595b2eb3d70d23fce39` changes only empty optional-C representation and validation from rejected `-inf` to required `null`; original full-vocabulary replay, token positions/log probabilities, row sums, native boundary logits and trace parity are unchanged. A later current-source edit adds the model field to receipts only. Dedicated source snapshots for the two production hashes were not retained; launch receipts preserve their path, hash and size.

## Scope

The tied and untied states are separate model strata, not an untie-only causal contrast. The book result is conditioned on book, and both image-885 results are conditioned on person. Candidate masses are finite tested sets, not complete owner probabilities. No result here identifies repair, training origin, population recall, abstract owner memory or global sequence optimality.

## Stable evidence and replay

- Deterministic reduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-native-row-choice/reduction-integrated.json`
- Integrated candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/candidate-manifest-v3.json`

```sh
PYTHONPATH=. python probes/training_set_completion/native_row_choice/selfcheck.py
PYTHONPATH=. python probes/training_set_completion/native_row_choice/reduce.py --selfcheck
```
