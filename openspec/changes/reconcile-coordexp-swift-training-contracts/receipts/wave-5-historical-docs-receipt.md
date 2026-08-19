# Wave 5 Historical Reading And Canonical Documentation Gate Receipt

Date: 2026-08-19. Lead: Claude Fable session. Base: `9ad1d66ec` (Wave-4
close). This wave changed two test files and seven documentation surfaces;
no `src/`, stable-spec, or history edit.

## 5.1 Fixture matrix

The five fixture classes are covered by executable tests:
(a) current committed exact checkpoint — real `publish_training_state`
publications in `tests/artifacts/test_training_state.py` plus the new
end-to-end pair node below; (b) current inference-only checkpoint —
`test_checkpoint_payload_identity.py::_write_checkpoint_payload` and
`test_checkpoint_writer.py::test_disabled_exact_state_publishes_only_the_inference_payload_and_aliases`;
(c) older adapter-plus-delta payload with extra metadata —
`test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata`
and `test_historical_payload_without_a_current_manifest_stays_inference_loadable`;
(d) incomplete current publication — the incomplete/staging/abort family in
`test_training_state.py` plus probe boundary receipts; (e) unknown
resume-like historical files — `optimizer.pt`/`legacy_resume.pt` nodes,
undeclared-file rejection, and schema guards.

The one load-bearing gap — no fixture paired a REAL committed inference
payload with a REAL committed `training_state/` in one checkpoint — is closed
by `tests/artifacts/test_checkpoint_payload_identity.py::test_inference_reader_ignores_real_committed_training_state_without_opening_it`.

## 5.2 Reader-boundary proofs

- Inference readers consume explicit payloads without reading
  `training_state/`: proven at equality level (manifest/identity/admission
  byte-identical to a sibling-free control) AND now at file-access level —
  `pathlib.Path.open` interception scoped around the three inference-reader
  calls captured 45 real opens (manifest, adapter config/weights, embedding
  delta json/safetensors) and zero paths containing `training_state`. The
  interception matches the reader's actual mechanism
  (`checkpoint_payload.py` reads via `Path.read_text`/`path.open("rb")`,
  including `_sha256_file` hashing of every inventoried file).
- Exact admission accepts only the current committed typed schema: existing
  strict-manifest family plus the new
  `tests/artifacts/test_training_state.py::test_unknown_schema_value_is_rejected`
  (unknown schema family string and unknown future `schema_version` both
  rejected with `training_state.unsupported_schema` before digest
  validation).
- Historical artifacts are never upgraded from names, extra files, or
  archived claims: existing undeclared-file/path-escape/symlink/legacy-file
  nodes; the claim boundary "payload-reader level; explicit-path consumption
  only" recorded in evidence-matrix row 85 stands, with the supporting fact
  that `training_state` has zero references under `src/inference/`,
  `src/adapters/`, and `src/qwen/`.

## 5.3 / 5.4 Canonical documentation

Seven surfaces reconciled with the accepted bounded contract (always-published
minimal inference payload vs opt-in disabled-by-default `training_state/`
sibling; optimizer-step boundary at same world size/rank map; fail-closed
admission; commit-then-alias ordering; non-goals):
`docs/COORDEXP_SWIFT.md` (denial replaced; owners row and `resume.mode`
config route added, including the verified `strict_cuda_replay_v1`
co-requirement), `docs/SYSTEM_OVERVIEW.md` (denial replaced; flow annotated),
`docs/IMPLEMENTATION_MAP.md` (three routing rows, callback seam, two stable
spec routes added), `docs/ARTIFACTS.md` (two-surface split, inventory names
`inference_payload_manifest.json` + `training_state/`, failure semantics,
non-goals), `docs/PROJECT_CONTEXT.md` (denial replaced with one bounded
sentence), `docs/catalog.yaml` (six inventory keys + notes), and
`docs/AGENT_INDEX.md` (route sentence extended). Schema detail stays in
specs; receipts/history stay out of evergreen docs. No page links a stable
`coordexp-swift-training-resume` spec path because that spec materializes at
archive sync.

## 5.5 Conflict scan resolution

- Canonical-doc categorical denials: all removed (scan below).
- `openspec/specs/coordexp-swift-training-artifacts/spec.md:361-363` still
  carries the pre-change denial by design: that block is MODIFIED by this
  change's delta and resolves at spec sync/archive (6.6), not by editing the
  stable spec mid-change.
- `openspec/specs/coordexp-swift-vertical-smoke/spec.md:130-135` is a
  smoke-scoped non-goal, not a repository-wide denial; left as-is.
- Provider-mode residue (`legacy_fused`,
  `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE`) is absent from docs and
  stable specs; live source residue remains hand-off to the decomposition
  change per task 1.3.
- Cache-mutability language: no live contradiction (docs and pack-cache spec
  agree; `docs/data/PACKING.md` already carves out exact-resume-referenced
  payloads).

## 5.6 Gate

- Executable doc-authority scan across 14 canonical surfaces: CLEAN — no
  categorical denial, no archived-change-as-authority citation, no
  unfinished packing/efficiency/logging/loss/RL/architecture promotion, and
  the required exact-state mentions present
  (scan script preserved at the session job directory; findings zero).
- `docs/catalog.yaml` parses (`yaml.safe_load` OK).
- Focused suites at the close commit: `test_checkpoint_payload_identity.py`
  + `test_training_state.py` + `test_checkpoint_writer.py` +
  `test_exact_resume.py` + `test_pipeline_exact_resume.py` → **179 passed**.
- No new independent audit layer was added; the scan is a change-local gate
  vehicle, not a permanent test.
