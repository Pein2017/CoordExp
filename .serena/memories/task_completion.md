# Task Completion Checklist (Memory)

Role separation:
- Memory role: minimal done-definition before wrapping a coding task.
- Canonical docs: domain docs in `docs/` and runbooks for full procedures.
- Owner memories for contract details: `.serena/memories/config_yaml_guide.md`, `.serena/memories/packing_and_stage2_rollout_matching.md`.
- Update trigger: when acceptance criteria, trainer contracts, or validation flow changes.

Pre-close sanity:
- Confirm configuration passes owner-memory contract checks (config + packing/stage2 semantics).
- Ensure selected variant-specific knobs are coherent for the active run (for example stage2 rollout namespace and effective batch semantics).
- For coord-token experiments, explicitly choose whether coord distributional loss is enabled.

Quick verification loop:
- Validate JSONL: `public_data/scripts/validate_jsonl.py`
- Render one sample: `scripts/tools/inspect_chat_template.py`
- Smoke run config: `python -m src.sft --config <yaml> --debug`

Closure:
- Run targeted tests when behavior changed.
- If contract semantics changed, update canonical docs and follow OpenSpec workflow.
