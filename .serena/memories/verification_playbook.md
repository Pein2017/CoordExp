# Verification Playbook

Role: minimal closeout posture plus the small command spine we reach for most often.

Canonical pointers:
- `docs/IMPLEMENTATION_MAP.md`
- `docs/ARTIFACTS.md`
- `docs/data/PREPARATION.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/eval/WORKFLOW.md`

Defaults:
- Run from repo root with `conda run -n ms`.
- Start with the narrowest targeted test or smoke path from `docs/IMPLEMENTATION_MAP.md`.
- For data, prompt, or template changes, validate one JSONL sample and inspect one rendered example.
- For changes that affect artifacts, provenance, or logging surfaces, verify against `docs/ARTIFACTS.md` instead of relying on remembered filenames.
- If entrypoints, defaults, artifacts, metrics, or recommended workflows changed, update the relevant docs/specs/router pages in the same pass.
- Final handoff should say what was verified and what remains unverified.

Useful commands:
- `PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> --debug`
- `conda run -n ms python public_data/scripts/validate_jsonl.py <jsonl>`
- `conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path> --index 0`
- `conda run -n ms python -m pytest -q <narrow-targeted-tests>`
- For Stage-2 or infer/eval changes, prefer the specific tests named in `docs/IMPLEMENTATION_MAP.md` before any broad suite.