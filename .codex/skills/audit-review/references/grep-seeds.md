## Grep Seeds (High-Signal `rg` Starting Points)

Use these as a breadth-pass index. Prefer scoping with `--glob` or `relative_path` when possible.

### OpenSpec Change Audits
- `rg -n \"openspec/changes/\" openspec/changes -S`
- `rg -n \"\\- \\[x\\]|\\- \\[ \\]\" openspec/changes/<change>/tasks.md -S`
- `rg -n \"Requirement:|Scenario:\" openspec/changes/<change>/specs -S`

### Pipeline / Data Flow
- `rg -n \"pipeline|planner|stage|artifact|manifest\" src public_data -S`
- `rg -n \"raw\\.jsonl|norm\\.jsonl|coord\\.jsonl|gt_vs_pred\\.jsonl\" -S`
- `rg -n \"do_resize|geometry|bbox_2d|poly|norm1000|coord\" src public_data -S`

### Config / Contracts / Deprecated Keys
- `rg -n \"from_mapping\\(|_validate_section_keys_strict|Unknown top-level\" src/config -S`
- `rg -n \"deprecated|unsupported|fail fast\" src -S`
- `rg -n \"unknown_policy|semantic_fallback|packing_length\" src tests -S`

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

