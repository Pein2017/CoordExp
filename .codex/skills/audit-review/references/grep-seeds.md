## Grep Seeds (High-Signal `rg` Starting Points)

Use these as a breadth-pass index. Prefer scoping with `--glob` or `relative_path` when possible.

### OpenSpec Change Audits
- `rg -n \"openspec/changes/\" openspec/changes -S`
- `rg -n \"\\- \\[x\\]|\\- \\[ \\]\" openspec/changes/<change>/tasks.md -S`
- `rg -n \"Requirement:|Scenario:\" openspec/changes/<change>/specs -S`

### Pipeline / Data Flow
- `rg -n \"pipeline|planner|stage|artifact|manifest|resolved_config\\.path|effective_runtime|experiment_manifest\" src docs openspec tests public_data -S`
- `rg -n \"raw\\.jsonl|norm\\.jsonl|coord\\.jsonl|gt_vs_pred\\.jsonl|gt_vs_pred_scored\\.jsonl|metrics_guarded|duplicate_guard_report\" -S`
- `rg -n \"do_resize|geometry|bbox_2d|poly|norm1000|coord\" src public_data -S`

### Config / Contracts / Deprecated Keys
- `rg -n \"from_mapping\\(|_validate_section_keys_strict|Unknown top-level\" src/config -S`
- `rg -n \"deprecated|unsupported|fail fast\" src -S`
- `rg -n \"unknown_policy|semantic_fallback|use_pred_score|packing_length|duplicate_control|pred_coord_mode|bbox_format\" src tests docs configs -S`

### Current Architecture Seams
- `rg -n \"run_pipeline|ResolvedArtifacts|resolved_config\\.path|_maybe_run_confidence_postop|_run_eval_stage\" src/infer tests docs -S`
- `rg -n \"evaluate_and_save|EvalOptions|with_constant_scores|gt_vs_pred_scored_guarded|metrics_guarded\" src/eval tests docs -S`
- `rg -n \"Stage2ABTrainingTrainer|stage2_coordination|rollout_runtime|RolloutMatchingSFTTrainer|stage2_ab\\.pipeline|rollout_matching\\.pipeline\" src tests docs openspec configs -S`
- `rg -n \"geometry_from_dict|transform_geometry|compute_coverage|bbox_2d|poly\" src/datasets tests docs -S`

### Progress / Benchmark Scope
- `rg -n \"val200|limit=200|first 200|full-val|full val|raw-text|coord-token|coco_real|strict_plausible|throughput|GPU\" progress docs -S`

### Silent Failures / Exception Swallowing
- `rg -n \"except Exception:\\\\s*(pass|continue|return|\\\"\\\")\" src -S`
- `rg -n \"except:\\\\s*pass|except BaseException:\\\\s*pass\" src -S`
- `rg -n \"logger\\\\.(warning|exception)|raise RuntimeError|raise ValueError\" src -S`

### Determinism / RNG
- `rg -n \"random\\\\.|np\\\\.random|torch\\\\.manual_seed|seed\" src -S`
- `rg -n \"sorted\\(|sort\\(\" src public_data -S`

### Tests That Gate Behavior
- `rg -n \"test_.*policy|fail.*fast|deprecated\" tests -S`
- `rg -n \"parity|matches_legacy\" tests public_data/tests -S`
