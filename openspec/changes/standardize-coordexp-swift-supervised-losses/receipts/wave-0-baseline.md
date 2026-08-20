# Wave-0 Baseline (tasks 0.1 + 0.2)

Recorded 2026-08-20 by the Claude Fable lead (session c9895ff9).

## 0.1 Predecessor pins

| Fact | Value |
| --- | --- |
| reconcile change | ARCHIVED `openspec/changes/archive/2026-08-19-reconcile-coordexp-swift-training-contracts/` (spec-synced at `eb2dc97ab`) |
| decompose change | ARCHIVED `openspec/changes/archive/2026-08-20-decompose-coordexp-swift-training-orchestration/` (final commits `ebfa78ff1` close + `68191f7ea` archive; skip_specs; completion audit 0 P0/P1) |
| this change's base commit | `0cb0ae729` (tracked tree clean at recording) |
| `openspec validate --all` | 21 passed / 0 failed post-archive |

No disagreement found between code, stable specs, docs, or archive
dispositions: the decompose completion audit (0 P0/P1) verified
docs/IMPLEMENTATION_MAP.md, docs/SYSTEM_OVERVIEW.md, and
docs/COORDEXP_SWIFT.md against the accepted owner graph at the same source
bytes this change starts from.

## 0.2 Owner graph and cache invariant

`PACKING_CACHE_DETERMINANT_OWNERS`: 31 determinants across 24 owner files
(full list in `wave-0-determinant-baseline.json`; registry code
`src/training/pack_cache.py`).

**Overlap analysis with this change's affected surfaces**
(`src/config/models.py`, `src/losses/`, train/eval reduction, rank-zero
logging projection, supported `configs/`):

- The ONLY determinant owned by a loss-package file is
  `realized_vocab_groups` → `src/losses/vocab.py`. Its content derives from
  token identity + tokenizer (not from `losses.*` config), and the canonical
  gate group tuple already exists there as `V1_TOKEN_TYPES = ("desc_text",
  "schema", "coordinate", "eos")` — Wave 1 imports it, never edits the file.
- `micro_step_runtime_config_identity` (owner `src/training/cache_contract.py`)
  serializes exactly `training.precision`, `model.fa2_branch_proof`-derived
  booleans — no `losses.*` field.
- The full semantic payload reads dataset/template/packing/processor/
  ordering/augmentation/qwen/vocab/micro-step fields only. **No determinant
  content reads the `losses` config section**, so the Wave-1.3 supported-config
  migration cannot move either fingerprint.
- **DO-NOT-EDIT list for this change** (aggregate fingerprint hashes owner
  source bytes — decompose Wave-7 projections proved owner-source changes move
  the fingerprint): the 24 owner files, notably `src/losses/vocab.py`,
  `src/supervision/tokens.py`, `src/training/micro_steps.py`,
  `src/training/cache_contract.py`, `src/training/pack_cache.py`. Any edit
  there = blocking contract review (proposal stop rule), NOT a rebuild.

**Predecessor cache facts (admitted evidence, immutable):**

- root `.cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/`
- train target `8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f`
- eval target `3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662`
- hit receipts: decompose archive `receipts/wave-7-cache-preparation.json`
  (both `built`), `receipts/wave-7-cache-verification.json` (both `hit`,
  `--require-all-hit`), plus the Wave-8 smoke run.json admission.
- Recomputed at this change's base commit `0cb0ae729`: both aggregate
  fingerprints equal the published targets exactly (live cache-hit invariant).
  Full payloads frozen in `wave-0-determinant-baseline.json`
  (`baseline_sha256` inside; 405,123 bytes) — the byte-for-byte equality
  reference for task 5.3.

**Old-telemetry consumer inventory (Wave-4 migration scope, recorded now):**
`rg 'loss/(base_ce|token_type_gate|coord_gaussian_rps|total)'` over current
roots hits ~20 files: tests/losses, tests/runtime, tests/eval,
tests/artifacts, tests/training (incl. orchestration_compatibility and the
wave7 exact-resume comparators), tests/test_teacher_forcing_*,
scripts/probes + scripts/analysis consumers. The frozen fixtures
`tests/fixtures/training_orchestration/{completed_step_rows.json,
run_writer/logging.jsonl}` carry the OLD field family and are commit-bound
historical evidence — Wave 4 must declare row-schema flips in this change's
command manifest instead of regenerating fixtures.
