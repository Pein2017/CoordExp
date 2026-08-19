# Wave 4 Cache And Provenance Gate Receipt

Date: 2026-08-19. Lead: Claude Fable session (takeover of thread
`019fef89-...`). Implementation commits in scope: tests-only
`4e4c5f5d7bb46d21f7a5a647aa63bae6120e334a` atop `582a19ff3` (Wave-3 close).
No `src/` change was made in Wave 4.

## 4.1 Trace outcome

The live cached-payload determinant registry is
`src/training/pack_cache.py:61-129` (31 named determinant owners with
per-determinant reasons, registry schema v1), constructed by
`build_packing_cache_determinants` (`:254-338`) with aggregate fingerprint and
code identity. Its independent completeness test declares its own literal
owner inventory (`tests/training/test_pack_cache_determinant_registry.py:64-96`)
and proves every unique owner's source bytes reach the fingerprint. Post-build
determinant revalidation runs inside `_publish_micro_step_cache`
(`src/training/pack_cache.py:570-586`) before the no-replace install; immutable
absent-target publication holds a per-root flock with byte-preserving hits and
fail-closed collisions (`:467-505`, `:588-595`); pre-model train/eval admission
is `_resolve_model_free_training_preflight`
(`src/training/pipeline.py:2812-2998`) ordered before `_build_accelerator`
(`:3344`) and model assembly (`:3647`).

**No unsupported cache delta claim was found**: every normative sentence and
scenario in `specs/coordexp-swift-pack-cache-semantic-identity/spec.md` maps to
a live source owner and at least one executable test (trace table preserved in
the Wave-4 scout map, reproduced in the change record). Nothing was removed.

## 4.2 Demonstrated gaps closed (three focused additions, commit `4e4c5f5d7`)

1. `tests/training/test_pipeline_assembly.py::test_resolve_eval_pack_cache_hardcodes_payloads_verification_level`
   — the real `_resolve_eval_pack_cache` body must pass
   `verification_level="payloads"`; previously all three call sites
   monkeypatched the resolver wholesale so the eval level was asserted nowhere.
   Sensitivity was proven by an empirical mutation kill.
2. `tests/training/test_pack_cache.py::test_publication_fails_closed_on_hash_matching_forbidden_global_payload`
   — a staged chunk whose bytes decode to a forbidden global while the declared
   SHA-256 matches those exact bytes fails closed during publication: canonical
   target absent, stage removed, no partial install.
3. `tests/artifacts/test_provenance.py::test_environment_secrets_never_enter_provenance`
   — four secret-shaped environment variables with sentinel values never appear
   in the serialized receipt; top-level key set unchanged; no
   `environment`/`env` key anywhere in the nested structure.

Rulings logged, no code change made:
- Optimizer/scheduler non-construction on cache failure is accepted
  transitively: `build_optimizer_and_scheduler` runs only after
  `assemble_model_surface` inside `_run_initialized_training`, and the focused
  suite proves cache failure precedes model assembly and accelerator
  construction (`model_load_calls == [False]`, `accelerator_calls == []`).
- P3 disposition for 6.5: the pre-publish `payloads` validation failure
  surfaces as a plain `ValueError("packing cache chunk payload is unreadable")`
  rather than `PackingCacheInvalidError` with a `validation_category`, unlike
  sibling branches (`determinant_drift`, `target_already_exists`). Fail-closed
  semantics are intact; only the error taxonomy is inconsistent.

## 4.3 Model-free cache probe

`receipts/wave-4-cache-probe-receipt.json` (file SHA-256
`0c0cfbd61f6943cfda3b16c3a3d03a12ce71cf72af8fe551705167e0d891a35e`), three arms
through the real `src.prepare_train_cache` entrypoint against a private
`COORDEXP_SWIFT_PACK_CACHE_ROOT`:
- absent target → `built`/`built`, admission level `payloads` both splits,
  `model_loaded: false`, train fingerprint `ec5baadb...` and eval `76f369a7...`
  (matching the Attempt-8 production-path setup identities);
- valid existing target → `hit`/`hit` byte-preserving reuse after `payloads`
  admission, preparation/publication `not_run_cache_hit`;
- invalid target (one corrupted chunk byte, digest mismatch) → exit 1,
  `training.pack_cache_immutable_collision`,
  `validation_category: required_payload_digest_mismatch`,
  `automatic_recovery: "unavailable"`, and the corrupted bytes untouched by the
  failed run (no repair/rewrite/GC).
No production cache was published; no efficiency claim is made.

## 4.4 / 4.5 Provenance

Focused suite `tests/artifacts/test_provenance.py` (41 nodes) covers
clean/dirty/untracked repository states, non-git explicit unavailability,
per-component dependency failure isolation, deterministic strict JSON, and the
new secrets-cannot-enter node. The bounded receipt from the current
worktree/runtime is `receipts/wave-4-provenance-receipt.json` (file SHA-256
`b7c24c144cc3a6fc459dff6378b850fd9a45dbffaeeba94ba3ba2f0c7ed40e1e`): tracked
tree clean at `4e4c5f5d7`, `untracked_changes_present: true` solely from the
not-yet-committed Wave-4 receipt files themselves, no environment section
anywhere. Exact-resume compatibility consumes only the declared identity
projection (`build_resume_compatibility_projection`: resolved-config
`schema/schema_version/semantic_config`, never `resume`/`run`/environment/
provenance), verified by the three projection tests named in the receipt.

## 4.6 Gate

Executed on clean bytecode at `4e4c5f5d7` (all green):

- cache contract + provenance + projections (10 files):
  `tests/training/test_pack_cache.py`,
  `test_pack_cache_determinant_registry.py`,
  `test_pipeline_cache_preflight.py`, `test_pipeline_pack_cache_rebuild.py`,
  `test_prepare_train_cache_cli.py`, `test_pack_cache_runtime_constructor.py`,
  `tests/artifacts/test_provenance.py`, `test_run_artifacts.py`,
  `test_training_state.py`, `tests/training/test_exact_resume.py`
  → **489 passed**.
- `tests/training/test_pipeline_assembly.py -k "prepare_training_pack_caches
  or hardcodes_payloads"` → **3 passed**.
- `tests/training/test_reconcile_exact_resume_probe.py -k "prepare and not
  real"` → **15 passed**.

Incident recorded: the first gate run executed under a poisoned
`src/training/__pycache__/pipeline.cpython-312.pyc` left by a same-length,
same-second mutation-kill probe (CPython pyc validation is second-granular
mtime + size, so the byte-identical restore did not invalidate it). The stale
bytecode was proven by comparing `__code__.co_consts` against an in-process
compile of the disk bytes, the pyc was deleted, and the entire gate re-ran
clean; only the counts above are load-bearing. No archive scope was inherited
and no new audit layer was added.
