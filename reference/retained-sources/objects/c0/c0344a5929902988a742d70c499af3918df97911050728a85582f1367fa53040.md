# Native coordinate-branch completion — candidate

No retained higher-likelihood nonrepeated owner row. Width4 improves repeated A realization; forced52 yields credible B but lower full-row probability;30 partial-owner identity HOLD.

At genuine native donut417044 row6/action49 (x1/action54), original R16 FP32/SDPA bs4 runtime, original readout, three fixed branches and width4 row-only search. All prior history including rows3/4 HOLD is unchanged. Root acceptance is pending.

| x1 | Greedy row / logp | Best retained row / logp | Best minus original greedy | Physical status |
|---|---|---|---|---|
| 0 | [0, 256, 66, 315] / -12.577036 | [0, 259, 69, 317] / -12.104793 | +0.472242 | repeated_A |
| 52 | [52, 265, 106, 329] / -12.738894 | [52, 269, 106, 329] / -12.736211 | -0.159176 | unvisited_B |
| 30 | [30, 197, 61, 234] / -12.892689 | [30, 209, 61, 234] / -12.695042 | -0.118006 | physical_identity_HOLD |

All scores include raw common prelude and forced x1 probability. Branch52 has a much easier conditional suffix, but its x1 is rank50, margin−2.856792 to the original winner; omitting that cost would mis-rank branches. Branch30 x1 is rank2, margin−0.228556. No normalization was applied.

The best x1=0 row departs at y1/action55:259 rather than greedy256, rank4 and chosen-minus-best margin−0.192986. It improves full-row logp by0.472242 but still denotes repeated A. The actual greedy suffix has positive all-competitor margins (minimum0.011030), certifying deterministic row replay from this exact start. The best52 and30 paths first differ from native at x1/action54; every selected token rank/margin is in result.json. Their subsequent beam deviations do not certify greedy selection.

Four52 rows denote admitted B with extent variations; all five0 rows denote A. Five30 boxes are small partial regions at adjacent upper-left donuts/reflection: dominant physical identity and coveredness remain HOLD, not FP. Frozen IoU50 matches agree on A/B and leave30 unmatched; this is separate from visual evidence. Fourteen unique final rows inspected via full-image family context and all local extents; no additional census.

All14 finals are complete and geometry-valid; none EOS, malformed or capped. There are12 beam and3 greedy provenances (one duplicate). Four terminal hypotheses were pruned in30 and remain recorded. Conditional beam captured mass is0.000360656/0.003289912/0.000240585 for0/52/30; omitted/pruned mass remains explicit. These are finite row events, not owner probability or scene probability.

Technical checks: fresh full-prefix/no-KV branching prevents parent-cache sharing; immutable CPU observations only. Root requery is bit-exact. Native0 matches saved onset through row completion; largest2×logit-error/argmax-margin is0.01393. Independent complete-prefix rescoring differs by at most4.27e-6 nats. Saved-output reducer reconstructs every selected probability, parent choice, pruning/tie decision and mass balance. Forced-token omission falsification passes. Input/checkpoint/annotation bindings remain unchanged; no parameter mutation.

Strongest surviving alternatives: finite search missed distinct high-likelihood completions; model prefers repeated A among these realizations; coordinate30 may denote partial geometry rather than a coherent new owner. No global search optimality, coverage repair or loss/training prescription follows. This package stops here.

Cost: 61 batch model forwards, 166.598 allocated GPU-seconds, 151.16MiB tensors;3 successful producer exits,0 model failures/retries, no owned jobs. One CPU scoring import needed PYTHONPATH=.; no model re-execution.

Artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-native-coordinate-branch-completion/result.json`, `ARTIFACTS.md`, `reduction.json`, `physical-score.json`, `runtime/`, `terminal.json`. Reconstruct with `python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-native-coordinate-branch-completion/reduce.py` and `PYTHONPATH=. python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-native-coordinate-branch-completion/physical_score.py`. Frozen protocol: `unit.md`.

Knowledge check: one expected root-owned frontier omission for this new state; all other checks pass. Root integrates the frontier at acceptance. Scoped diff whitespace check passes. No cleanup performed.
