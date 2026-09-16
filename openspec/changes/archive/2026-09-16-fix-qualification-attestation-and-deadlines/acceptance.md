# Acceptance: qualification attestation, deadlines and renderer fixture

## Scope and ownership

User authorized drafting and executing the three bounded repairs on 2026-09-14.
Baseline: `6cf1a53b23ab25107c2e1a407e7dd0171e2cbd2f`, clean canonical
`/data/CoordExp/.worktrees/coordexp-infras` checkout. Proposal/design/spec/tasks
in this change own scope; no scientific or numerical admission criteria change.

Flat native team: Sol high (`qualification`, fork none) owns source identity and
supervision together because code/tests are shared; Luna high (`renderer_fixture`,
fork none) owns the two-value fixture repair. Lead owns planning, operator docs,
diff inspection and integration acceptance. No delegated review or nested team.
Costs are unknown; no model ranking or cost-efficiency claim is inferred.

## Renderer fixture: lead-accepted

- RED: the existing snapshot test failed on baseline. Worker retained
  `/data/CoordExp/.local/share/rtk/tee/1789357015_pytest.log`; lead had also
  independently reproduced the failure before implementation authorization.
- Actual/expected comparison found only `examples[0].template_fingerprint` and
  `examples[1].template_fingerprint`. Approved rename commit
  `412877ab58e6441deaa3943540ad7c9283df2708` changes the renderer identity from
  `coordexp-swift-template-v1` to `coordexp-infras-template-v1`.
- The exact fixture diff changes two values from
  `8feee87f23928d7d934ff9a76f7f464bddc15266a1d6b80f8e4b17e87799c104` to
  `5929cf3fa990b8c04cc4b4c95b40f67b368522ff5e77508705be7f3b535e6e0b`.
  Lead independently compared parsed baseline/current JSON and asserted those
  exact two changed paths with every other field identical.
- GREEN: isolated-pycache renderer suite: `29 passed in 13.49s`, exit 0;
  `/tmp/coordexp-renderer-fixture-green.log`. Post-fix actual/expected comparison:
  zero differences, `/tmp/coordexp-renderer-fixture-compare.log`.

## Qualification: lead-accepted

- Existing source selection missed renderer drift. The baseline supervisor
  required an external alarm after 1.021 seconds to interrupt a child whose
  descendant held its pipes; baseline harness cleanup left no live process-group
  members. Evidence: `/tmp/coordexp-qualification-repair-red-baseline.log`.
- Initial test RED log (`/tmp/coordexp-qualification-repair-red-tests.log`, exit 1)
  contains 14 failures. Its first renderer regression failed because the copied
  baseline source tree omitted the renderer itself; that failure is NOT evidence
  of admission behavior. Lead spotted the verifier gap before submission; owner
  corrected the copied-tree fixture, relative-import coverage and test cleanup.
- Lead independently restored only baseline `_SOURCE_PATHS` from the baseline
  Git AST in an isolated pytest process and ran the final
  `test_renderer_source_drift_rejects_stale_receipts_through_admission`.
  Result: `DID NOT RAISE RuntimeContractError`, 1 failed, exit 1. This proves
  old source selection still admitted renderer drift with the final verifier.
  Evidence: `/tmp/coordexp-qualification-repair-red-admission.log`.
- Current identity selects 74 files. The regression guard checks static absolute
  and relative local imports plus parent package initializers. Concrete additions
  include templates, losses, packing, supervision, coordinate targets and
  augmentation geometry. There is no generic dynamic-import coverage claim;
  inspection found no current local dynamic imports in the selected closure.
- `produce(child_timeout_seconds=1800.0)` and run CLI
  `--child-timeout-seconds` validate finite positive input before work. Real
  CPU-process acceptance exercises production `produce` and its supervisor with
  a 1-second deadline, real child/pipe-holding descendant, stubbed GPU census,
  and existing process-group TERM/KILL cleanup. It asserts explicit
  `vllm_qualification.child_timeout`, reaped worker, empty live group and no
  passed receipt publication. Test finally cleanup is retained.
- Both GPU census calls use a 5-second wait. Timeout preserves unavailable
  evidence (`None` or `-1`); this is not successful cleanup evidence. Timeout
  always fails, regardless of cleanup observations. Receipt schema and normal
  numerical/cleanup admission rules remain unchanged.
- Worker GREEN: `112 passed, 2 warnings in 11.40s`, exit 0;
  `/tmp/coordexp-qualification-repair-green-final.log`.

## Integration: lead-accepted

- Lead ran `python -m pytest -q tests/inference tests/templates` on the integrated
  candidate: **417 passed, 5 warnings in 26.77s**, exit 0;
  `/tmp/coordexp-qualification-repair-integration.log`. Warnings concern SWIG
  deprecations and the existing fork-based cache concurrency test.
- Lead inspected all implementation/fixture/documentation diffs, confirmed
  CLI run help exposes the 1800-second default/override, and checked whitespace.
- `openspec validate fix-qualification-attestation-and-deadlines --strict`
  and task routing receipt validation pass. Four tasks are complete.
- No GPU/model forwards, training, qualification artifact regeneration, receipt
  admission, commit or publication occurred. This is CPU source-admission and
  process-supervision acceptance, not current model/GPU qualification. Expanded
  source identity invalidates old receipts intentionally; fresh qualification
  remains necessary before production admission. Logs above are local ephemeral
  evidence; commands, outcomes and exact counterexamples are preserved here.
