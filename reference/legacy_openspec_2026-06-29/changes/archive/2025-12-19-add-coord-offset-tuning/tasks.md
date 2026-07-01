## Tasks

- [x] Document coord-offset requirements and scenarios (spec delta) covering embedding/head offsets, ID range, PEFT save via modules_to_save, and interaction with dlora.
- [x] Design coord-offset adapter and optimizer grouping (design.md) with chosen defaults and failure modes.
- [x] Implement adapter module and hooks; wire enable flag and ID config into `sft.py` without disturbing default runs.
- [x] Extend optimizer plugin with `multimodal_coord_offset` LR buckets and ensure ZeRO/flash-attn compatibility.
- [x] Add YAML overlay for dlora production config enabling coord-offset with tunable LRs.
- [x] Add tests: forward/backward correctness (offset only hits coord IDs), optimizer grouping, config load on/off.
- [x] Run `openspec validate add-coord-offset-tuning --strict` and targeted test command(s); report results.
  - retested optimizer grouping after duplicate param fix.
