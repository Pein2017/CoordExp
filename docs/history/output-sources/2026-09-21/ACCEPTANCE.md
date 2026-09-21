# Independent acceptance: source/artifact boundary

Status: **lead-accepted**, 2026-09-21. Scope: the current `research-probes`
checkout's maintained-source migration, historical source recovery and output
placement. This is CPU/file-identity acceptance, not GPU numerical parity or
authorization to resume any historical experiment.

## Accepted changes

- `c395a7a2e`: maintained scorer, native-input identity, composition and retained
  runtime owners; explicit historical source recovery; external source captures;
  generated model-card and visualization metadata.
- `f2b310e40`: legacy completed-gate runner recovery by the receipt's original
  SHA, with strict checks for newly declared source captures; three remaining
  source-capture writers; DoRA model-card packaging; generated analysis,
  gallery and coordinate-margin reconstruction text preserved in JSON.
- The documentation/archive commit containing this record preserves original
  source bytes, storage rules and the recovery locator. Historical whitespace
  is deliberately preserved through `.gitattributes`, not reformatted.

No sealed receipt, expected historical hash, model tensor or scientific result
was rewritten. New generated reports retain their text in JSON `summary` fields;
the coordinate-margin reconstruction map is `artifact-map.json`. Generated
research-unit `results.md` remains in its research unit.

## Fresh validation

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python -B -m pytest -q -p no:cacheprovider \
  probes tests/artifacts tests/adapters tests/test_gt_vs_pred_visualization.py \
  tests/research/test_research_knowledge.py tests/analysis/test_output_writer_boundary.py
```

Result: **930 passed, 5 failed, 15 skipped**, 950 collected. The five failure
names and messages exactly match the preserved pre-migration baseline; no new
failure is accepted. All report `ValueError: sources.producer source bytes changed`:

- `test_v3_bank_and_dev_rows_materialize_as_complete_known_owner_ledgers`
- `test_main_packet_is_held_and_binds_fresh_source_training_plus_batch4_readback`
- `test_control_reuse_replays_all_prior_raw_shard_bindings`
- `test_plan_freezes_high_prefill_probe_and_balanced_full_cohort_shards`
- `test_v3_lean_cases_hydrate_for_the_actual_bound_request_builder`

The completed-gate test reproduced the missing `runner.py` before repair and
passes through the actual preparation caller after repair. It also rejects a
wrong historical hash and changed current capture. Writer regressions failed
on loose output Markdown before repair and pass afterward. Runtime capture
tests stop before model loading; they do not qualify GPU execution.

The full source archive verified **10,877 entries**. Both physical outputs roots
passed with zero placement findings (82,322 root files and 27 worktree files;
124 directory/file links reported without traversal). Staged whitespace checks
passed. High-confidence token/private-key patterns and literal credential
assignments were scanned without exposing values; no candidates were found.
The configured local credential source is ignored and untracked.

Local logs, JUnit, RED evidence, source hashes and the machine-readable receipt:
`/data/CoordExp/.worktrees/research-probes/.local/source-artifact-boundary/independent-acceptance/`.
The original baseline remains in its parent directory. Commit metadata binds
the exact maintained code; Git commits do not replace these local evidence files.

## Cleanup and preserved ownership

Removed only the unused
`.local/source-artifact-boundary/test-source-captures/`: **216 files, 4,087,057
bytes**. Its exact file hashes were recorded; no process holder or archive
manifest dependency was present. The main archive, checkout backups, tar,
retired environments and baseline/final validation records remain intact.

The archive mapping spans this worktree, the root checkout and the maintenance
backup; [the locator](../README.md) describes that boundary. No independent or
portable worktree-only recovery claim is made.

The following concurrent/pre-existing research surfaces are intentionally not
included in the maintenance commits:

- `probes/training_set_completion/spatial_progress_gate/`
- `probes/training_set_completion/visual_instance_binding/`
- `research/experiments/2026-09-21-spatial-progress-gate/`
- `research/experiments/2026-09-21-spatial-progress-recovery/`
- `research/experiments/2026-09-21-visual-instance-binding/`
- Existing edits to research index, catalog and the history-repetition / visual
  designation question pages.

The first two untracked research packages already contained migration edits to
their runtime source captures. Those tested working-tree edits are preserved
with their owning packages, not committed as incomplete standalone runtimes.
The root checkout and shared skill changes made by the original migration are
also outside these `research-probes` commits. No fetch, merge or push occurred.
