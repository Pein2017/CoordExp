# Repetition history mechanism — artifact reconstruction

Execution complete; scientific candidate, not lead-accepted. No training, parameter edits, label admission or successor launch.

## Reading entry
- Owning frozen protocol: `/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-17-repetition-history-mechanism/unit.md`; sampling freeze: `stage-c.md` there.
- Candidate narrative/state: `results.md` and `state.json` in that unit.
- Authoritative numerical result: `result.json`; complete per-cell ledgers: `reduction.json`; sampling rates/individual trajectories: `sampling-summary.json`.
- Bounded physical identity review: `physical-review.json`, with full plotted contexts and selected local views under `review/`. UNKNOWN is not FP. Supplied, symmetric-union-excluded and full-final sets remain separate.
- `terminal.json` owns final completion/live-job status; `cost.json` binds all49 receipt executions and allocation evidence. Earlier runtime-handoff.json and partial/interim files are historical transport, not live state.

## Inputs and exact execution
- `runtime-manifest.json`, `panels/{A,B,C}/`, `sampling-manifest.json`: every frozen condition, exact prefix, config/model/adapter/paired embeddings, processed images, owner banks, original heterogeneous batch and runtime source identities.
- `stage-a-panel.json`, `stage-a-alt-panel.json`, `anchors.json`, `alternate-expression-freeze.json`: literal A/B/C and alternate expressions. `anchor-summary-correction.json` corrects an advisory book hash typo without modifying the preserved source; execution tokens were exact.
- `source-deltas.json`, `native-before.py`, `native-additive-helper.diff`: additive source helper difference versus predecessor; both native baselines reproduced every saved batch member exactly.
- `source-snapshots.json` and `source-snapshots/`: maintained executed runtime, scorer and reducer snapshots. `sampling-producer.py` is the executed immutable sampling producer; `sampling-producer.diff` is its actual diff against the native runtime.
- Effective readout rows/factors remain at immutable predecessor paths bound by each panel's coefficients/source identity. They are not recomputed or tuned here.

## Raw and intermediate artifacts
- A/B: `runtime/{A,B}/<condition>/<image>/<mode>/{raw.json,receipt.json,pulse-captures.pt}`. Raw includes full token IDs/text/stops, exact native companions, prefix hashes and interventions; captures contain actual final head input plus raw/intervened full-vocabulary logits at declared seams.
- C: `runtime/C/T<temperature>-seed<seed>/{raw.json,receipt.json,sampled-logits.pt,rng-states.pt}`. Every one of216 sampled steps retains full raw logits, forced-selection scores, final head inputs, per-draw CUDA RNG states, chosen token and temperature log probability. All24 outputs and failures are retained; all24 completed successfully. Post-withdrawal full outputs are retained, but no C post-withdrawal hidden-state trace is claimed.
- `scores/309264-smoke/` and `scores/309264-alt/`: complete-row likelihoods, common-prefix fork scores, raw tensors and CPU checks; 35 full-prefix score forwards. `qualification-parity.json` and predecessor-parity.json there compare saved incremental/full-prefix seams relative to margins.
- `fork-summary.json` is an earlier compact A/B capture projection, not the exhaustive archive. `runtime-verification.json` covers23 completed A/B runs/236 captures. `sampling-verification.json` covers all24 C runs/216 draws and includes local margins, field types, raw/EOS probabilities and entropy.
- No complete KV/attention archive, layer sweep, fresh labels or exhaustive owner census.

## Recompute without model forwards
From `/data/CoordExp/.worktrees/research-probes`, set `R=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-repetition-history-mechanism` and `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`.

```
python -m probes.training_set_completion.repetition_history_reduce "$R" --output /tmp/repetition-reduction.json
python "$R/summarize_sampling.py" --input /tmp/repetition-reduction.json --output /tmp/repetition-sampling-summary.json
python "$R/verify_sampling.py" --cuda --output /tmp/repetition-sampling-verification.json
```

The last command uses CUDA only to replay saved categorical draws, not a model forward. `independent_check.py` deterministically replays the consumer and verifies raw row/invalid/EOS counts and supplied-owner exclusion; final evidence is `independent-reduction-check.json`. It writes that receipt if invoked, so independent auditors should copy the script/change only its output destination rather than overwrite sealed evidence. `summarize_sampling.py --selfcheck` falsifies A-template, malformed-gap and Wilson endpoint errors.

## Commands, failures and costs
- `run-*.sh`, `logs/*.log`, `logs/*.pid`, `logs/*.exit`, settlement markers: exact finite commands/process and success/failure witnesses. All47 native batch runs and2 score batches succeeded. Six replay controls are included, not extra scientific arms.
- `sampling-first-qualification*.json`: first counted cell admission; it was reused, never relaunched.
- `independent-check-invocation-failure.json`: a CPU PYTHONPATH invocation failure, fixed by the correct worktree import path; zero model reruns.
- `cost_receipt.py`: per-receipt measured runtime plus conservative log-birth-to-exit allocation upper bounds and union of execution intervals; idle task gaps are excluded from actual execution time.
- `knowledge-check.json`/log: final repository checker. Cross-study lead owns global frontier integration and final acceptance.
