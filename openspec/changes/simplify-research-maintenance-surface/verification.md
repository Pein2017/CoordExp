# Verification: first maintenance tranche

Date: 2026-09-25. Workflow Session: `wc_sess_HFinegRbtRv2B086`.

## Scope and outcome

Implemented the twelve tasks in this change's bounded first tranche. The full
`training_set_completion` family cutover is **not** complete and is outside this
tranche. No generic experiment framework or historical source library was added.

| Dimension | Assessment |
|---|---|
| Completeness | Navigation, exact owner imports, token/text extraction, same-byte artifact deduplication, selected launcher retirement and current test ownership implemented. |
| Correctness | 142 focused CPU tests passed; real frozen exposure read/selection tests passed; extracted function ASTs match their old implementations except approved renaming. |
| Coherence | Current mechanics stay under their concept owner; scientific recipes, identity/collision distinctions and independent evidence remain separate. |
| Remaining limitations | No full runtime-suite execution or real-model/GPU parity; no canonical integration; full family split remains pending. Global OpenSpec strict and metadata-only hygiene have the separately explained findings below. |

Target Project and branch are `research-probes-web-codex`, with canonical path
`/data/CoordExp/.worktrees/research-probes-web-codex` and unchanged HEAD
`d4763fd048f1e651ca6067045e5f6d56798cdc07`. Edits remain uncommitted; the index
was not staged. No commit, push, merge, rebase, stash, experiment or model call
was performed.

Canonical `research-probes` advanced independently to
`48ba07071993c5b953a37e537b38ef46163cc88f`, with target/canonical commit difference
1/3. It was clean at one intake observation and had fresh uncommitted research
again at closeout. This change made no writes to that checkout and did not import
its newer recurrence files. Recheck it before any subsequent integration.

## What was removed or relocated

The following six files were removed after exact current-byte/HEAD checks and
source/reference inspection:

| Removed file | Disposition |
|---|---|
| `scripts/run_infer.py` | Unsupported old inference entry importing the removed `src.infer.pipeline`; no compatible-forwarding alias. |
| `scripts/run_infer_eval.sh` | Direct caller of the retired entry. |
| `scripts/analysis/run_ckpt_pair_confidence_eval.sh` | Legacy inference/confidence launch chain, not a saved-result reader. |
| `scripts/pipelines/run_rollout_stability_probe.sh` | Legacy launch wrapper; offline stability analysis remains. |
| `tests/test_run_infer_legacy_shared_runtime.py` | Test for the removed CLI, which injected the missing old pipeline as a fake module. |
| `memories/bootstrap-main-thread-prompt.md` | Consumed one-time reconstruction instruction, with no current consumer found. |

Historical commands in dated evidence remain historical rather than being
rewritten to pretend they used current entries. `compare_detection_runs.py` and
`report_rollout_stability.py` still read saved outputs. The old command schema is
not advertised as compatible with `python -m src.infer`.

Seven test files were relocated, not discarded: two current knowledge tests to
`tests/knowledge`, the exposure contract to `probes/parallel_owner_research/tests`,
and four COCO22 contract tests to `probes/training_set_completion/tests`.
The new locations are included by default pytest discovery.

Sixteen files now import native row/request/composition operations directly from
existing owners. Three exact token/text operations were moved to the pure
`src.inference.token_text` module and their consumers migrated. Four route-bank
artifact implementations were removed in favor of the existing completion-family
owner; Unicode/newline/hash and publication distinctions remain unchanged.
The geometric-dedup training producer's new execution-code list includes the
extracted source file. Its list was inspected without invoking the GPU run path.

## Navigation and evidence preservation

The frontier now links the complete catalog rather than repeating every state.
`research/index.md` changed from 60,987 to 5,691 UTF-8 bytes. It explicitly marks
its checkout-scoped research boundary instead of posing as the later canonical
frontier. All 307 catalog entries and 91 current-schema state owners remain.

`memories/current.md` is a 675-byte stable route, not a second live state ledger.
Unique September 2 recollections were attributed in
`memories/notes/2026-09-02-closed-research-context.md`, including an explicitly
unverified historical audit number. Automatic memory-write policy was removed;
the current cleanup was explicitly authorized by the user.

No `research/experiments`, `research/questions`, `manifests` or `reference`
tracked content changed. Within `docs/history`, only README's current validation
command was updated. The frozen exposure corpus remains 106 exact files and 135
excluded image IDs. Model tensors, annotations, external output payloads and
sealed receipts were not rewritten. The retired `SourceArchive` was not restored.

`human13/output_qp.py`, `training_set_completion/readout_norm_fresh.py` and
`training_set_completion/untied_natural.py` remain byte-identical to target HEAD.
Their reuse priority is not a claim of novelty or arbitrary-input support.
Required teacher/DoRA dependencies were not deleted merely for being ordinary.

## Executed validation

All pytest commands used the installed `ms` Python, CUDA hidden,
`PYTHONDONTWRITEBYTECODE=1`, one-thread CPU library settings and
`-B -m pytest -q -p no:cacheprovider`. Final checks also set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Test subprocesses were tiny local CPU
children, not research runs.

| Check | Observed result |
|---|---|
| Initial knowledge/documentation/artifact baseline | 34 passed. |
| New catalog-navigation tests against old checker | Three expected failures, then repaired; missing/fake links still fail. |
| Old exposure test in this noncanonical checkout | Expected path-root failure; fixed by explicit test-local root binding, not a producer/data change. |
| Relocated knowledge/exposure/COCO22 first run | 61 passed, one stale test called removed `_wait_child`. |
| Relocated tests after using actual completion waiter | 62 passed; exact child exit 7 recorded and reaped. No old alias or skip added. |
| Combined final focused acceptance | **142 passed, one existing tensor-to-scalar warning**, 7.95 seconds reported by pytest. |
| Default collection, no explicit test paths | **2,351 tests collected**, exit 0. Collection is not execution of those tests. |
| Exact extraction comparison | Three normalized ASTs identical except renamed identifiers; pure import does not load torch, transformers or probes. |
| Knowledge checker on real tree | `ok: true`; 307 catalog entries, 323 protocols, 91 states, 893 local links and 106 frozen records; 553 external handles explicitly unverified. |
| New navigation links / changed Python syntax / `git diff --check` | Passed. |
| This OpenSpec change and changed main spec, strict | Both passed under installed OpenSpec 1.13.0. |

The combined pytest command selects:

```text
tests/knowledge
tests/inference/test_token_text.py
probes/dora_owner_learning/tests/test_native_import_owners.py
probes/dora_owner_learning/tests/test_geometric_dedup.py
probes/dora_owner_learning/tests/test_geometric_dedup_eval.py
probes/dora_owner_learning/tests/test_branch_bridge.py
probes/training_set_completion/tests/test_artifact_primitives.py
probes/training_set_completion/tests/test_route_bank.py
probes/training_set_completion/tests/test_repair_bank.py
probes/training_set_completion/tests/test_repair.py
probes/training_set_completion/tests/test_coco227_readback.py
probes/parallel_owner_research/tests/test_transfer_exposure.py
probes/training_set_completion/tests/test_coco22_acquisition.py
probes/training_set_completion/tests/test_coco22_annotations.py
probes/training_set_completion/tests/test_coco22_evaluation.py
probes/training_set_completion/tests/test_coco22_review_packets.py
probes/human13/tests/test_output_qp.py
```

## Warnings and remaining work

**Global OpenSpec strict is not green:** 8/26 specs passed; 18 unchanged specs
failed strict mode for placeholder Purpose text. Every failing spec was compared
byte-for-byte with HEAD and was unchanged. This change's delta and
`research-probe-development` main spec both pass. Do not misreport the global
command as successful or rewrite unrelated normative behavior to satisfy it.

**Metadata-only workspace hygiene returned a raw blocking result:** it classified
`src/inference/token_text.py` and `tests/inference/test_token_text.py` as secret-like
paths because of their names. Both are the newly authored/inspected literal-token
source and its tests, with no credentials. Other findings were intended new probe
files and expected uncommitted edits. These are manually resolved name-based
false positives, not a clean-tool verdict. Normal source was neither ignored nor
deleted to silence the heuristic.

The new tests check the modified navigation scenarios, pure token mapping and
actual frozen-data consumers. Existing multi-objective/profile tests and
output-QP CPU tests protect their bounded contracts; no general transfer,
new-profile/model parity or historical whole-run replay was verified. Saved
receipt source hashes remain historical and were not weakened to fit edited code.

Before the next family-cutover slice, inspect canonical's newly committed and
uncommitted recurrence consumers, choose a safe local editing handoff and migrate
the retained dependency closure together. Keep the target's edited index separate
from canonical's newer scientific synthesis; never overwrite the latter wholesale.
Remaining legacy data/production wrappers, Source256 profile placement, the
historical migration-checker coupling and independent CPU reader ownership remain
explicit follow-on work, not silently declared complete by this tranche.
