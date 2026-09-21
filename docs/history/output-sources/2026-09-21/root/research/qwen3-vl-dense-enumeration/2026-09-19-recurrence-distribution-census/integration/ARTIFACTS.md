# Integrated artifact index — candidate, not lead-accepted

Owning checkout: `/data/CoordExp/.worktrees/research-probes`.
Output base: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration`.
This integration directory is under `2026-09-19-recurrence-distribution-census/`.

## Start here

- `results.md`: integrated scientific interpretation, conditioning and exclusions.
- `result.json`, `integrated-terminal.json`: package denominators/status and stable terminal.
- `manifest.json`: exact selected evidence, executed probes and research-record bindings. Nested lane artifacts retain their raw bindings; this manifest does not claim archival gaps were recovered.
- `cost.json`, `account_cost.py`: disjoint attempt counters and conservative device-reservation accounting, including invalid work.
- `job-closure.json`: fresh process reconciliation and child completion.
- `changed-paths.json`: scoped repository/output surfaces; unrelated dirty work is preserved.
- `direct-completion.json`: actual App Server delivery receipt, created after candidate sealing. Delivery is not root acceptance.

## Lane evidence

| Lane | Output root | Reconstruction entries |
|---|---|---|
| A | `2026-09-19-recurrence-distribution-census` | `eligible-manifest.json`, `exclusions.json`, `identity-audit.json`, `new128.runtime.jsonl`, `runtime/<condition>/<group>/{raw,trace,receipt}.json`, `mature-census.json`, `new-census.json`, `scientific-synthesis.json`, `shared-panel.json`, `shared-sources.json`, `archival/archival-binding-reconciliation.json` |
| B | `2026-09-19-recurrence-spatial-source` | Corrected `final/corrected-pilot-v2/` and `final/broad-v1/{runtime,reduced,result.json,artifact-map.json,job-closure.json,cost-receipt.json}`; `final/technical-gate-v3/{source.pt,target.pt,trace.json,compare.json}`; all prior invalid attempts remain separate in `final/` and candidate-v1 archive. `final/attempt-ledger.json` distinguishes disjoint counters. |
| C | `2026-09-19-recurrence-conditional-mass` | `draws.jsonl`, `state-bindings.json`, `sampler-entry.json`, `reduction.json`, `run-manifest.json`, `qualification-attempts.json`, `launch-failures.json`, `qualification/qualification.json` |
| D | `2026-09-19-coordinate-input-continuity` | `cpu-geometry.json`, `selection.json`, `runtime/` exact paired captures/source checks and closure, `reduction.json`, `verification.json`, `result.json`; old reducer snapshot preserved under `runtime/archive/candidate-reducer-v1-before-full-vocab/` |

The four owning `research/experiments/2026-09-19-<lane>/` directories contain frozen units, candidate results/state and local artifact maps. Original and normalized policy data, old/mature/prospective/runtime strata, and raw versus intervened observations stay separate. Raw tokens, stops, prefixes, model/media identities, compact logits/head inputs, trained coordinate tensors and short-draw RNG information are preserved where the lane calls for them. No complete attention/KV archive or training update is claimed.

## Independent CPU checks

Run from the owning checkout. Let `I` denote this absolute integration directory. These commands do not launch models. Verification scripts preserve lane candidate bytes and put recomputed results under integration.

```bash
PYTHONPATH=. python "$I/verify_census.py"
PYTHONPATH=. python "$I/verify_spatial_inputs.py"
PYTHONPATH=. python "$I/verify_spatial_pilot.py"
PYTHONPATH=. python "$I/verify_spatial_broad.py"
PYTHONPATH=. python "$I/verify_spatial_aggregate.py"
python "$I/verify_spatial_bindings.py"
python "$I/summarize_spatial.py"
PYTHONPATH=. python "$I/verify_mass_events.py"
PYTHONPATH=. python "$I/verify_mass_conditioning.py"
PYTHONPATH=. python "$I/verify_continuity.py"
PYTHONPATH=. python "$I/verify_continuity_reduction.py"
python "$I/account_cost.py"
python scripts/research/check_research_knowledge.py check
```

Saved complete-reducer equality receipts are `mature-check.json`, `new-census-check.json`, `mass-reduction-check.json`, `spatial-broad-check.json`, `spatial-aggregate-check.json` and `continuity-reduction-check.json`. D narrative `scientific_record` is explicitly outside the generated-field equality claim. B corrected nested binding check passes 386 references/138 unique files. This is not an all-package archival-completeness claim.

The position-coded falsification and native readback consumer are `probes/training_set_completion/recurrence_spatial/parity_readback.py`; pass its `--source`, `--target`, `--trace`, `--target-index 3` and a disjoint `--out` using the v3 files above. `parity-position-red-green.json` and `spatial-parity-v3-check.json` record the parent check. The two v3 calls are the final two of ten gate-lineage calls; the earlier wrong-position readbacks remain measurement-invalid.

## Limits that must travel with the artifacts

- Shared mechanism panel: 21 failure + 24 proxy states, 23 image identities, unbalanced source/model representation; not a population estimate.
- B: 267/315 possible corrected cells executed; 48 signed cells unexecuted after local HOLD; 12 pilot signed cells are diagnostic-only. Target-only native qualification is bounded to the actual checked seams, not a blanket parity claim for every transformed state.
- C: seven empty same-description unions make q=0 structural; full-softmax short draws condition on supplied description, not free scene enumeration.
- D: all 16 selected roles are y2; fixed-suffix scores are not free trajectories.
- A's overwritten original prelaunch/source JSON and D's overwritten shard receipt remain explicit archival gaps. Exact current raw/reducer/source evidence does not recover unavailable original bytes.
- UNKNOWN/unmatched is not FP, numerical recurrence is not verified physical identity, and current-positive matching is not exhaustive scene truth.
