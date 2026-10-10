# Code verification — tasks 1–4

## Implementation map

- **1.1–1.2 — characterize the retained caller:** `.codex/skills/lead-worker/scripts/test_pi_worker_turn.py` retains the original fixture cases and adds receipt-before-`get_state`, public generated-return execution, unrelated-cwd invocation, and `runpy` CLI coverage. Temporary wrong-order and wrong-entrypoint variants are rejected (RED evidence below).
- **2.1–2.3 — extract transport:** `pi_worker_turn.py` remains the public parser/CLI and return-command builder. `peer_transport/api.py` owns bounded loopback HTTP, errors, exact session/model/cwd validation; `dispatch.py` owns envelopes, durable receipts, creation and the shared guarded send path; `bindings.py` owns strict read-only route resolution. Imports are CLI → modules and dispatch → API; no module imports the CLI.
- **3.1–3.4 — explicit peers and aliases:** the Pi CLI tests cover an independent third root, direct worker-to-worker targeting, one intended recipient, separate cws, read-only inspection, v1 alias routing, malformed/ambiguous modes with no API mutation, target-setting drift, exclusive receipts and one-attempt unknown delivery. `test_peer_transport.py` covers strict JSON keys, versions, fields, aliases, UUIDs, cws and duplicate identities.
- **4.1–4.3 — references and brief integrity:** the Pi CLI tests exercise report references with both direct and legacy routes, present-secret and unavailable pointers, unchanged explicit message-file delivery, reply correlation, body/envelope hashes, default queue and explicit steering. `test_peer_transport.py` checks escaped metadata round-trip and reference validation.

## Baseline, RED/GREEN, and checks

- **Unchanged baseline:** `python .codex/skills/lead-worker/scripts/test_pi_worker_turn.py` — exit **0**, **14 tests** on the pre-change script; local test-owned HTTP fixture only.
- **RED before implementation:** after adding the caller tests, the same suite ran **17 tests** and exited **1** with the three new peer/alias/`runpy` entry paths rejected by the legacy-only parser. The new module contract test exited **1** with `ModuleNotFoundError: peer_transport` before extraction.
- **Wrong-order counterexample:** in a temporary copy only, moving recipient `get_state` ahead of receipt creation makes `test_existing_send_loads_unloaded_target_and_routes_in_both_directions` fail at “receipt must exist before target loading” (exit **1**).
- **Wrong-entrypoint counterexample:** in a temporary copy only, generating a return command to `peer_transport/dispatch.py` makes `test_create_verifies_names_then_sends_with_identity_and_direct_return` fail its public-script-path assertion (exit **1**).
- **Candidate Pi tests:** `RTK_HOOK_DISABLE=1 python .codex/skills/lead-worker/scripts/test_pi_worker_turn.py` — exit **0**, **19 tests**; `RTK_HOOK_DISABLE=1 python .codex/skills/lead-worker/scripts/test_peer_transport.py` — exit **0**, **5 tests**.
- **Syntax and public help:** `RTK_HOOK_DISABLE=1 python -m py_compile` over the entrypoint, package and two Pi test files — exit **0**; `RTK_HOOK_DISABLE=1 python .codex/skills/lead-worker/scripts/pi_worker_turn.py --help` — exit **0**. Tests also execute the public CLI by subprocess and `runpy` from an unrelated cwd.

## Qualification boundary

All caller, create/inspect/send, redirect, timeout and receipt checks use only the test-owned loopback HTTP server. The create test simulates the existing endpoint contract; no live Pi Web session was contacted or created. This is CPU-only transport qualification, not hosted delivery or behavioral-efficiency evidence. Parent acceptance and final task status are recorded in `verification.md`.

## Parent acceptance corrections and final GREEN

Parent review requested two bounded corrections: new peer envelopes now include JSON-escaped expected sender/recipient cwd values (legacy envelopes remain unchanged), and the report-pointer test guards actual send execution against `Path.open`, `Path.stat` and `Path.resolve` on the referenced report. The latter checks absence of file access, not merely absence of secret text in the envelope.

Final qualification after these corrections:

- `RTK_HOOK_DISABLE=1 python -B -m unittest discover -s .codex/skills/lead-worker/scripts -p 'test*worker_turn.py' -v` — exit **0**, **24 tests** (19 Pi CLI and 5 unchanged Codex tests).
- `RTK_HOOK_DISABLE=1 python -B .codex/skills/lead-worker/scripts/test_peer_transport.py -v` — exit **0**, **7 tests**, including escaped cwd, unchanged legacy envelope and actual report no-access checks.
- **31 tests total passed.** The earlier 5-test module result above is an intermediate candidate result, not the final total.
- `git diff --exit-code HEAD -- .codex/skills/lead-worker/scripts/worker_turn.py .codex/skills/lead-worker/scripts/test_worker_turn.py` — exit **0**; neither Codex file changed.
- `git diff --check` and strict no-interactive OpenSpec validation — exit **0** before final evidence/checklist updates.
