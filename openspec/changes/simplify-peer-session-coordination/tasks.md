## 1. Characterize the retained transport boundary

- [x] 1.1 Run the existing Pi CLI fixture tests and retain a compact baseline in change verification; confirm legacy create, inspect, cross-cwd sends, queue/steering, redirects, receipt reuse and unknown-delivery cases pass without contacting the live Web service.
- [x] 1.2 Add characterization coverage for receipt durability before target loading, generated public return-command execution, other-cwd CLI invocation and `runpy` argument loading; demonstrate the checks reject a wrong ordering/entrypoint without weakening existing tests.

## 2. Extract a minimal Python package

- [x] 2.1 Extract HTTP/errors and exact session/model validation into `scripts/peer_transport/api.py`; retain the public script CLI and verify the unchanged fixture suite and help/argument behavior.
- [x] 2.2 Extract receipt/envelope and create/send orchestration into `scripts/peer_transport/dispatch.py`, passing the public return-command builder or entrypoint explicitly; verify generated returns use `pi_worker_turn.py` and receipts precede every mutating API call.
- [x] 2.3 Keep dependency direction acyclic and remove extracted duplicate implementations rather than adding broad compatibility shims; verify package imports via subprocess and `runpy` from another cwd, and verify Codex `worker_turn.py` and its imports remain unchanged.

## 3. Add explicit peer and task-local alias routing

- [x] 3.1 Add `--sender`/`--recipient` with explicit cwd values for existing independent peers; extend the local fixture with a third root and verify worker-to-worker routing prompts exactly one intended recipient without requiring a lead role.
- [x] 3.2 Implement read-only version-1 `pi_peer_transport` bindings in `scripts/peer_transport/bindings.py` and `--participants-state`; verify selected aliases resolve once, unrelated scientific fields are ignored, and state bytes remain unchanged after inspect, success and failure.
- [x] 3.3 Add rejection tests for absent/duplicate aliases, duplicate keys/session IDs, unsupported versions, unknown transport fields, bad cwd/identity, same-session routes, mixed legacy/peer modes and multi-recipient requests; verify zero API mutation for invalid routing.
- [x] 3.4 Normalize both routing modes into the same guarded send path and preserve legacy outputs/commands; verify unloaded-target inspection is read-only, setting drift blocks prompts, exclusive receipts block reuse, and a lost response yields one attempt plus unknown-delivery evidence.

## 4. Add concise report-reference metadata

- [x] 4.1 Add optional report path, section, revision and supersession flags to envelope/receipt formatting without reading referenced files; verify a secret-bearing report is never injected and an unavailable absolute pointer remains a reference rather than an existence claim.
- [x] 4.2 Preserve `--text`/`--message` exclusivity, explicit message-file body delivery, reply correlation and separate body/full-envelope hashes; verify metadata survives round-trip escaping and changes the envelope hash while leaving the body hash unchanged.
- [x] 4.3 Cover reference argument validation and default/explicit delivery in both new and legacy send modes; verify brief labels never change queue/steering or grant new task authority.

## 5. Refine collaboration guidance at its existing owners

- [x] 5.1 Use the installed skill-creator guidance to revise `lead-worker/SKILL.md` and add compact `references/coordination-examples.md`; verify examples cover outcome-sized assignments, relevant peer awareness, direct factual coordination, independent judgment, unaffected continuation and escalation without duplicating native-team rules.
- [x] 5.2 Update `references/pi-transport.md` with legacy, exact-peer and alias commands, participant section, short-body/report-pointer distinction and accurate idle/busy semantics; verify every example parses against the CLI and Markdown links/anchors resolve.
- [x] 5.3 Perform a bounded static walkthrough of three representative research cases: schema dependency coordination, superseding a CE presentation report, and an urgent invalidated target-binding action; verify each identifies correct audience, source/revision, required action and authority boundary. Record that this is not a live skill/token benchmark.

## 6. Qualify and close the implementation evidence

- [x] 6.1 Run the full focused Pi helper tests and unchanged Codex helper tests through the repository's selected Python interpreter, using test-owned loopback fixtures only; retain actual exits and caller-to-prompt-to-receipt evidence plus relevant RED/GREEN counterexamples.
- [x] 6.2 Check the final diff for preserved unrelated ownership, removed duplicate code, no new dependency/runtime/board/broadcast, and no edits to active research state or historical receipts; run `git diff --check` and OpenSpec strict no-interactive validation.
- [x] 6.3 Write change-local verification mapping each requirement to code/docs/tests and qualification limits; keep live Web session delivery, behavioral efficiency and research outcomes unclaimed, and leave archive/publication outside this implementation checklist.
