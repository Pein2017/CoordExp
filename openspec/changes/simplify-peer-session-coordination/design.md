## Context

See `proposal.md` for motivation and `specs/peer-session-coordination/spec.md` for observable behavior. Discovery used main HEAD `c63fb22f4ca8601be5227b0b4a3f503ead4eefc5`; the pre-existing dirty `.codex/AGENTS.md` is not owned by this change.

The current roughly 760-line `scripts/pi_worker_turn.py` combines argument parsing, a standard-library HTTP client, exact-session validation, exclusive receipts, create/name/send workflows, envelopes and return commands. `make_envelope` already carries explicit sender/recipient IDs; the missing abstraction is selecting peers without assuming opposite lead/worker roles. `inspect_or_send_pair` durably writes a receipt before `POST get_state`, then verifies loaded settings before prompting. Queue maps to `followUp`, steering to `steer`. Creation checks the model catalog and retains a partially created worker for reconciliation. These are useful contracts, not accidental complexity to remove.

`test_pi_worker_turn.py` exercises the public CLI with a local HTTP fixture and uses `runpy.run_path` for argument checks. There are no current monkeypatch targets to preserve. `test_worker_turn.py` separately imports the async Codex implementation; that implementation and its signatures are outside this refactor.

The research index/state already own shared decisions and assignments. Recent source-closure supplements and corrected CE tables demonstrate supersession needs; a target-binding file read before its queued message demonstrates the value of report pointers. They do not establish a need for broadcast or a second live board. Historical records are design examples, not execution grants.

## Goals / Non-Goals

**Goals:**
- One predictable single-recipient command for peer communication, with exact identity validation hidden behind optional task-local aliases.
- Concise, source-linked briefs and a clearer outcome-sized collaboration template.
- A small testable Python module structure that keeps receipt/mutation ordering visible.
- Compatibility for existing create, inspect, pair routing, model lookup and return commands.

**Non-Goals:**
- A common Codex/Pi adapter framework, transport plugin system, second Web session writer or HarnessDock lifecycle adoption.
- Passive notify-only semantics, background polling, message coalescing/withdrawal, read acknowledgments, group delivery or task scheduling.
- Automatically modifying research state, accepting scientific results, editing active assignments, or running existing research sessions.

## Decisions

### 1. Keep collaboration and transport at separate existing owners

`lead-worker/SKILL.md` owns persistent-peer usage and points to the generic outcome/continuation/acceptance rules in `native-agent-team-guide/SKILL.md`. Add a compact `lead-worker/references/coordination-examples.md` for assignment and brief examples rather than expanding the skill into a second constitution. `references/pi-transport.md` owns exact command semantics.

Assignments name relevant peers and dependencies, report destination, unaffected work that may continue, and escalation boundary. Briefs distinguish update, decision request and urgent correction; these are human meanings, not automatic execution or delivery-mode selectors. Independent reviewers record initial judgment before seeing others' conclusions. No hard word limit or mandatory report file for every short message.

Alternative rejected: a shared editable board or duplicate team policy. Existing research owners already serve the reading need and preserve one writer per semantic surface.

### 2. Use four substantive modules, not an extensible runtime

Retain the executable path and add a sibling package:

```text
.codex/skills/lead-worker/scripts/
  pi_worker_turn.py             # CLI, normalization, main, public return-command builder
  peer_transport/
    __init__.py                # package marker; no registration or side effects
    api.py                     # HTTP client/errors, exact-session/model validation
    dispatch.py                # receipts, envelopes, create and existing-peer send workflows
    bindings.py                # read-only task-local participant bindings/value validation
  test_pi_worker_turn.py       # retained caller-level regression fixture
  test_peer_transport.py      # focused pure routing/reference tests where useful
```

Dependency direction is CLI → bindings/dispatch/api, dispatch → api; no module imports the CLI. Keep receipt transitions next to dispatch orchestration. Pass the public entrypoint path or return-command callback into creation: moving `Path(__file__)` into a helper must not generate commands invoking the helper instead of `pi_worker_turn.py`.

Use only the standard library. No packaging installation, broad compatibility re-exports, abstract driver layer or dependency injection framework. Preserve `parse_args` at the public entrypoint; test subprocess execution from another cwd and `runpy` imports explicitly. Retire extracted duplicate bodies after callers/tests have moved.

Alternative rejected: module-per-function fragmentation or merging with Codex `worker_turn.py`. Different backend turn semantics make cross-harness unification unnecessary and risky here.

### 3. Resolve routes once from explicit inputs

Keep legacy flags unchanged. New existing-peer mode uses:

```text
# Explicit IDs; cwd values are mandatory and physical paths are checked.
python <scripts>/pi_worker_turn.py \
  --sender <UUID> --recipient <UUID> \
  --sender-cwd /absolute/a --recipient-cwd /absolute/b \
  --send --text 'Interface ready; see report.' --receipt /task-owned/new.json

# Aliases; directories come only from the explicitly selected state.
python <scripts>/pi_worker_turn.py \
  --participants-state /existing/unit/state.json \
  --sender gradient-worker --recipient runtime-worker \
  --send --text 'Interface ready; see report.' --receipt /task-owned/new.json
```

Without `--send`, both modes inspect only. New peer modes do not combine with legacy pair/create/list-model flags. Alias mode rejects explicit peer cwd overrides. Creation remains on the legacy path; state enrollment is an explicit operation by the task owner, not a transport side effect. Support no multiple-recipient option.

Task state may add this small section under its own owner's edit scope:

```json
{
  "pi_peer_transport": {
    "version": 1,
    "participants": {
      "gradient-worker": {"session_id": "<canonical UUID>", "cwd": "/absolute/physical/a"},
      "runtime-worker": {"session_id": "<canonical UUID>", "cwd": "/absolute/physical/b"}
    }
  }
}
```

Read only the selected JSON file; bound the read using the existing 8-MiB transport payload scale. Reject duplicate JSON keys, unsupported section versions, unknown fields within the transport section/participant entries, invalid aliases, duplicate session IDs and non-absolute/nonexistent cwd values. Unrelated top-level science fields remain uninterpreted. Aliases are case-sensitive simple atoms (`[A-Za-z0-9][A-Za-z0-9_-]{0,63}`). Resolve paths and selected bindings into one invocation snapshot; validate both sessions against their expected directories before any API mutation. Receipt/envelope retains aliases as display metadata and exact IDs/cwds as routing facts.

The mapping is convenience, not caller authentication or permission. Exact session checks do not prove the invoking process controls the sender. Do not infer caller authority from a role label or a binding. Keep endpoint selection in the existing `--base-url` contract, not in state.

Alternative rejected: scanning session lists by name, a global registry, or parsing historical `workers` entries as routes. Those entries can include retired workers and scientific status fields; the small explicit section avoids inference.

### 4. Keep brief input distinct from report pointers

Reuse `--text` and legacy `--message FILE` for the actual message body. Add one optional `--report-path`, plus `--report-section`, `--revision`, and `--supersedes` as concise reference metadata. Section requires a report pointer; revision/supersedes can describe a brief or report. Require a nonempty absolute report-path string, but do not stat, open, hash, canonicalize or expand its contents. A missing artifact can be referenced, but cannot be called verified.

Render metadata with unambiguous escaping outside the brief body and retain it structurally in the receipt. `message_sha256` binds the selected body; `sent_sha256` binds the full envelope including pointers. Preserve the authority-neutral peer disclaimer and dispatch/reply identities. Do not force consumers to read all linked sections. Document explicitly that `--message REPORT_FILE` sends the entire file and is not an attachment.

Legacy generated return commands remain valid; new examples use a short body and `--report-path`. Supersession identifies exactly what changed and what remains valid; it does not erase earlier events, alter original evidence or remove queued messages.

### 5. Preserve transactional boundaries without promising transactions

Normalize legacy and new routes into one existing-peer send workflow. Preserve receipt-before-`get_state`, root/cwd guards, loaded-setting checks, one prompt attempt and unknown-delivery reconciliation. Reused receipts fail before mutation. Keep current safe output and failure categories; add route/reference fields without turning them into work/read/acceptance tracking.

Alias inspection does not reopen unloaded sessions. Queue can activate an idle recipient; steering is not immediate cancellation. No automatic retries, fallback backend, latest-alias refresh or new writer. These limits remain visible in docs rather than hidden by friendly names.

## Risks / Trade-offs

- [Alias drift] → resolve once, validate live identity/cwd, retain exact snapshot in receipts; never silently select a replacement.
- [Import/return-command regression after extraction] → exercise the unchanged CLI path, `runpy`, other-cwd invocation and generated return commands.
- [Brief misconstrued as authority] → preserve the peer disclaimer and separate task decisions from transport metadata; templates name escalation boundaries.
- [Referenced report changes or is missing] → explicit revision/supersession and consumer verification at acceptance; transport does not claim artifact integrity.
- [Queue still wakes idle recipients] → document accurately; passive inbox is a separate deferred requirement, not simulated by this change.
- [Skill adds reading overhead] → concise entrypoint with optional examples; static checks do not claim behavioral token savings.
- [Scope grows into team runtime] → first-version single-recipient, read-only bindings, no board, broadcast or scheduler.

## Migration Plan

1. Characterize existing CLI and receipt ordering through retained CPU fixture tests before extraction.
2. Extract modules without changing observable legacy behavior; verify create/inspect/send and return paths.
3. Add new explicit/alias routes and reference metadata through failure-oriented tests.
4. Update the skill and references using skill-creator guidance, with representative task/correction examples and proportional static validation.
5. Qualify caller → local HTTP fixture → exact prompt → durable receipt, including timeout, drift and referenced-secret non-disclosure. Run the unaffected Codex script tests as a narrow compatibility check.
6. Record evidence in this change; do not retrofit active research states, send live messages, restart services or claim hosted delivery qualification. Real usage may later enroll a selected task through its owner.

Rollback is reverting only this change's modules/docs while keeping the public CLI and legacy semantics. Do not edit historical receipts or touch unrelated work. Git commit/push and live-session tests are not part of planning or required for initial CPU acceptance.
