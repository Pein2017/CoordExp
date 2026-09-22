# Selected-score role evidence recovery

The native magnitude-lr0.003 step256/dev128 run failed on image460339/span1:
`scoring.selected_count_mismatch`, expected8, selected11. Its failed rank's raw
decode was not retained by native failure sanitation; the exact description
spelling is therefore unknown. Preserve the failed v1 artifact root.

The shared parser accepted exactly four geometry coordinates but collected score
evidence by lexical scanning across the entire object, including description.
An accepted description with three coordinate tokens, or two coordinates and a
commit token, independently reproduces selected11. Description wrapper spellings
also reproduced a silently wrong score0.115307 instead of0.5 with selected8.

Fix only accepted-object `schema_spans` and `coord_token_spans` in
`src/inference/parsing.py`: derive structural positions from regex role bounds.
Keep malformed/drop diagnostics lexical. No acceptance, text, geometry, matching,
score policy/fingerprint/formula, reward, or sample-denominator change.

Fresh RED: all three cases in
`tests/inference/test_scoring.py::test_special_tokens_in_description_are_not_geometry_score_evidence`
failed before the fix (two selected11 failures and one wrong-score assertion).
Fresh GREEN: `conda run -n ms python -m pytest tests/inference/test_scoring.py tests/inference/test_parsing.py tests/inference/test_artifacts.py -q`
passed74 tests. The regression also checks unchanged description and geometry.

Lead reparsed all16 existing completed native baseline/control runs referenced
by ce-evaluation-plan-v1.json:2816 rows,32737 accepted predictions. Every existing
prediction field, including score-evidence ranges, and every dropped prediction
is identical. Thus the already-sealed baseline/control comparisons do not need
regeneration. This count is parser-valid objects, not category-qualified owners.

Recovery uses a fresh step256/dev128 `natural-v2` run via the canonical
`qwen3_vl_2b_ce_controls_magnitude_prior_lr_step256_dev128_parserfix_v2.yaml`.
The never-executed step256/train256 uses its original v1 config. The completed
step64 read is reused. `ce-evaluation-plan-parserfix-v2.json` changes only the
failed point's run/config identity; adapter, cohort, decode and comparison stay
fixed. Launcher: `launch_magnitude_prior_recovery_v2.sh`; eventual reduction:
`native-ce/analysis-magnitude_prior_lr-v2`. Technical recovery is not a scientific
result until those two reads and the complete three-point aggregate pass.

Recovery completed14:18:24UTC. All three prior-lr native artifacts and hashes
passed lead verification. The recovered image460339/span1 has three coordinate
tokens and repeated object-end/box-start wrappers inside the parser-accepted
description. Its raw-span SHA-256 is
78f27828ac744b2cde85898b5d12dde33437048d1ac179442c502e08877accfe.
Replaying this actual new-run token trace with unchanged `_span_evidence`
lexical ranges reproduces selected11; current role-bound ranges select8 and
score0.33112907584011214. The old failed raw stream remains unavailable, so this
is real recovery-trace evidence, not a claim of byte identity with lost output.
One separate all-spans-dropped dev row remains behavioral evidence and is not
excluded from the fixed cohort. Scientific results are closed in results.md.
