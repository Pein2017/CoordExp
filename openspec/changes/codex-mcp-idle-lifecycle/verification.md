# Verification receipt

Status: **lead-accepted implementation and isolated package; shared runtime not activated**.

## Frozen delivery

- OpenSpec authority: this change directory under `/data/CoordExp/openspec`.
- Source base: `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a` (0.153.4).
- Source commit: `1208c8f07f480e3e778118a80fad99e5a62e742a` on `codex/mcp-idle-lifecycle`.
- Source worktree: `/data/CoordExp/external/harness/codex-mcp-idle-lifecycle-0.153.4`.
- Package: `/data/CoordExp/.codex/packages/standalone/releases/0.153.4-mcp-idle-20260909-r2-x86_64-unknown-linux-gnu`.
- Packaged CLI SHA256: `62bf83339c08b2ee6c817d687c79fec1555b9ea2558882f688cfcdc075abee21`.
- Unstripped build SHA256: `841c066aec642dde27b1dae46ca01e88d7dacef17256cc3e0041a26675fe580e`.
- Code-mode host: unchanged vendor 0.153.4 binary; SHA256 `3e85d67471825f73d02ff5f7e047ca1f6ca8caa3f59e4c6e8d9ca6ca7302cb45`.
- Only debug sections were stripped from the copied CLI. Tests below used the packaged binary.

The package contains `mcp-idle-manifest.json`, `validation/acceptance.json`, exact source manifests and patches, runnable harnesses, full isolated entry artifacts, and Rust check logs. Paths below are relative to its `validation/` directory unless otherwise noted.

## Real entrypoint acceptance

| Scenario | Result | Evidence |
|---|---|---|
| Main-task idle exit, same-task reuse, preserved ID/workspace/history, passive status | PASS | `entry-main/idle-mcp-acceptance-receipt.json` |
| Another task and a non-opted server remain active | PASS | same main receipt |
| Concurrent cold demand creates one replacement without business replay | PASS | same main receipt |
| Long MCP call and background command remain intact | PASS | same main receipt |
| Installed Serena restarts in the exact startup project and returns the expected symbol | PASS | same main receipt |
| Actual native v2 child completes and uses the short grace | PASS | `entry-child/receipt.json` |
| Real wake frontend is reclaimed/reconstructed; daemon and armed record survive unchanged | PASS | `entry-wake/wake-mcp-idle-acceptance-receipt.json` |

The main harness has 43 passing assertions. Its two separately declared gaps are closed by the child and wake harnesses; its `PASS_WITH_GAPS` label alone is not the composite acceptance claim. The child harness has eight passing assertions: its own process was reclaimed in 1.979 seconds under a two-second child grace, while the parent remained alive under a twenty-second main grace and child history remained readable. Wake returned `armed` before reclamation and after reconstruction; raw monitor fields and daemon identity were unchanged. All harnesses cleaned up their isolated owned processes.

No external model or GPU was used. A local deterministic Responses fixture drives real parent/child turns. Wake uses the installed plugin `0.1.0+codex.20260909065612`, one isolated daemon, and a future condition that does not fire during the bounded test. Production wake records were not touched.

## Rust and configuration checks

| Check | Result | Log |
|---|---|---|
| Selected config/MCP/rmcp package suite | 794 passed, seven existing skips, retries zero | `rust-checks/scoped-final.log` |
| Focused Core lifecycle/MCP suite | 251 passed | `rust-checks/core-focused.log` |
| Final passive-status correction: MCP library | 220 passed | `rust-checks/passive-status-green.log` |
| Final passive-status correction: app-server library | 285 passed | `rust-checks/app-server-r2-tests.log` |
| Remote executor fixture replay with built CLI | three passed | `rust-checks/remote-final.log` |
| Config schema generation, scoped Clippy fix, final formatting | passed | `rust-checks/config-schema.log`, `clippy-fix.log`, `fmt-r2.log` |
| Final CLI `cargo build --locked -p codex-cli` | exit zero | `rust-checks/cli-build-r2.log` |

These suites overlap; their counts are not added into a unique-test total. The final status correction was checked in the affected MCP/app-server packages and through the final packaged entry suite.

The generated config schema hash is `227a4dd5c3175c58cd7201cb95d9d6633f72a50a09729bca104fe66604a962bb`. Positive interval and required recovery-policy validation are exercised by configuration tests. No opt-in preserves existing behavior; remote/nonlocal transports are not suspended.

## Falsification and corrections

- Stock Core successfully performs ordinary calls/history persistence but fails automatic reclamation: `/tmp/codex-idle-stock-smoke.HznULj/run-20260909T094451Z-1349511/idle-mcp-acceptance-receipt.json`.
- Stock actual-child baseline fails only the short-grace expectation: `/tmp/codex-native-child-idle/native-child-20260909T095439Z/receipt.json`. Earlier harness iterations in that root were fixture-development failures, not lifecycle counterexamples.
- Invalid policy, disabled demand reconstruction, ambiguous/missing Serena startup selector, wrong ancestor, shared-gate release notification, and passive observer launching an opted-in server each have failing receipts and subsequent passing checks in `rust-checks/`.
- First packaged candidate exposed the passive-status bug and remains rejected at the sibling package without `-r2`. Receipt: `/tmp/codex-idle-packaged/run-20260909T102349Z-1486243/idle-mcp-acceptance-receipt.json`. The final package closes this exact counterexample.
- Serena continuing-body error behavior is source-pinned to installed Serena 1.7.0 / MCP 1.28.1 and tested with a synthetic server; it is not presented as a captured live timeout response.

## Limits and activation boundary

`just bazel-lock-update` could not run because Bazel is absent (exit 127). The lockfile proof shows exactly 149 workspace version changes from 0.0.0 to 0.153.4, with no external dependency drift. Workspace-wide tests were not run. These limitations do not replace the scoped checks and real entry evidence above.

Production grace examples are 900 seconds for main tasks and 120 seconds for completed native subagents; isolated acceptance uses shorter positive intervals. No production configuration was changed. The installed shared Core remains stock 0.153.4, and both the standalone proxy and separate legacy 0.151.0 instance were preserved. Activation must follow `OPERATIONS.md` and a fresh identity/active-task check within an authorized maintenance boundary. The installed wake plugin remains non-opted unless its manifest is separately updated; the implementation supports its explicit manifest policy without rewriting plugin caches.
