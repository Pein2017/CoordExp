# Official Serena Per-Worktree Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one unmodified official Serena backend per Git worktree, shared by concurrent Codex and Claude Code clients through stdio bridges, with guarded startup and automatic lease-based retirement.

**Architecture:** A tracked Python wrapper resolves the startup Git worktree, owns a service-local slot protected by `flock`, validates PID identity, starts official Serena in localhost Streamable HTTP mode, and runs a pinned stdio-to-HTTP bridge. Each wrapper owns one lease; the last departing client schedules a 60-second guarded reaper. Client configuration invokes the wrapper under `setpriv --pdeathsig TERM`.

**Tech Stack:** Python 3.12 standard library, Linux `/proc`, `fcntl.flock`, official Serena 1.7.x installed by `uv tool`, `mcp-proxy` 0.12.0 at Git revision `153a96a61fde2bf5a23961c64a3dd96b5e385108`, MCP SDK 1.27.1, pytest.

## Global Constraints

- Do not patch or vendor Serena, SolidLSP, Pyright, or another language server.
- Bind each backend to one canonical Git worktree root and localhost only.
- Never terminate by process name, wildcard, port alone, or unverified PID.
- Keep `serena-hooks remind` enabled and blocking for Codex and Claude Code.
- Do not change global proxy or `NO_PROXY`; remove all case variants of `*_PROXY` only from localhost backend and bridge child environments.
- Keep the fixed Serena tool allowlist unchanged.
- Work in `/data/CoordExp` and stage only task-owned paths because unrelated dirty changes are present.
- Runtime state belongs under `/data/CoordExp/.codex/serena/shared/` and must remain ignored.

---

### Task 1: Pure worktree, slot, and process identity primitives

**Files:**
- Create: `.codex/serena/serena_worktree_mcp.py`
- Create: `tests/runtime/test_serena_worktree_mcp.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `resolve_worktree(start: Path) -> Path`
- Produces: `slot_for(root: Path, runtime_base: Path) -> Slot`
- Produces: `read_process_identity(pid: int) -> ProcessIdentity | None`
- Produces: `identity_matches(expected: ProcessIdentity) -> bool`
- Produces dataclasses `Slot(root, key, path)` and `ProcessIdentity(pid, start_ticks, executable, argv, project_root)`.

- [ ] **Step 1: Write failing worktree and slot tests**

```python
def test_resolve_worktree_uses_exact_linked_worktree(tmp_path, monkeypatch):
    expected = tmp_path / "worktree"
    expected.mkdir()
    monkeypatch.setattr(module, "_git_toplevel", lambda start: expected)
    assert module.resolve_worktree(tmp_path / "worktree" / "src") == expected.resolve()


def test_slot_records_full_root_and_stable_key(tmp_path):
    root = tmp_path / "repo"
    slot = module.slot_for(root, tmp_path / "runtime")
    assert slot.root == root.resolve()
    assert slot.path == tmp_path / "runtime" / slot.key
    assert len(slot.key) == 24
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `conda run -n ms python -m pytest -q tests/runtime/test_serena_worktree_mcp.py`

Expected: collection fails because `.codex/serena/serena_worktree_mcp.py` and the named interfaces do not exist.

- [ ] **Step 3: Implement canonical worktree and slot derivation**

Use `/usr/bin/git -C <start> rev-parse --show-toplevel` with a minimal environment and a bounded timeout. Resolve the returned absolute path once. Derive `key = sha256(os.fsencode(root)).hexdigest()[:24]`; do not use only the basename.

- [ ] **Step 4: Write failing PID-reuse and identity tests**

```python
def test_identity_rejects_reused_pid(monkeypatch):
    expected = module.ProcessIdentity(
        123,
        100,
        "/root/.local/share/uv/tools/serena-agent/bin/python3",
        ("/root/.local/share/uv/tools/serena-agent/bin/python3", "/root/.local/bin/serena", "--project", "/repo"),
        "/repo",
    )
    monkeypatch.setattr(module, "read_process_identity", lambda pid: dataclasses.replace(expected, start_ticks=101))
    assert not module.identity_matches(expected)


def test_identity_requires_exact_project_argument(monkeypatch):
    expected = module.ProcessIdentity(
        123,
        100,
        "/root/.local/share/uv/tools/serena-agent/bin/python3",
        ("/root/.local/share/uv/tools/serena-agent/bin/python3", "/root/.local/bin/serena", "--project", "/repo-a"),
        "/repo-a",
    )
    observed = dataclasses.replace(expected, project_root="/repo-b")
    monkeypatch.setattr(module, "read_process_identity", lambda pid: observed)
    assert not module.identity_matches(expected)
```

- [ ] **Step 5: Implement `/proc` identity parsing and atomic JSON metadata**

Read `/proc/<pid>/stat` field 22, `/proc/<pid>/exe`, and NUL-delimited `/proc/<pid>/cmdline`. The console script's observed executable is the Serena uv-tool Python, while argv must contain the exact `/root/.local/bin/serena` launcher. Require exactly one `--project` argument whose following value resolves to the recorded root. Metadata writes use a same-directory temporary file, mode `0600`, `fsync`, and `os.replace`.

- [ ] **Step 6: Add runtime ignore and run GREEN**

Add only `/.codex/serena/shared/` and `/.codex/serena/runtime/` to `.gitignore`. Run the focused test, Ruff on both Python files, and `git diff --check`.

- [ ] **Step 7: Commit Task 1**

```bash
git add .codex/serena/serena_worktree_mcp.py tests/runtime/test_serena_worktree_mcp.py .gitignore
git diff --cached --check
git commit -m "Add Serena worktree runtime primitives"
```

---

### Task 2: Slot locking, leases, backend startup, and guarded reaping

**Files:**
- Modify: `.codex/serena/serena_worktree_mcp.py`
- Modify: `tests/runtime/test_serena_worktree_mcp.py`

**Interfaces:**
- Consumes: `Slot`, `ProcessIdentity`, and `identity_matches` from Task 1.
- Produces: `SlotLease.acquire(slot: Slot) -> SlotLease`
- Produces: `ensure_backend(slot: Slot, config: RuntimeConfig) -> Backend`
- Produces: `release_and_schedule_reap(lease: SlotLease, grace_seconds: float) -> None`
- Produces CLI subcommands `serve`, `reap`, and `doctor`.

- [ ] **Step 1: Write failing concurrent-start and stale-lease tests**

```python
def test_concurrent_clients_start_one_backend(fake_runtime, run_concurrently):
    backends = run_concurrently(2, lambda: module.ensure_backend(fake_runtime.slot, fake_runtime.config))
    assert {backend.identity.pid for backend in backends} == {fake_runtime.started_pid}
    assert fake_runtime.start_count == 1


def test_acquire_removes_only_stale_leases(fake_runtime):
    fake_runtime.write_lease(pid=10, start_ticks=100)
    fake_runtime.set_process(pid=10, start_ticks=101)
    lease = module.SlotLease.acquire(fake_runtime.slot)
    assert not fake_runtime.lease_path(10).exists()
    assert lease.path.exists()
```

- [ ] **Step 2: Confirm RED, then implement slot locking and lease reconciliation**

Hold `fcntl.flock(LOCK_EX)` on `<slot>/startup.lock` for every metadata transition. A lease is `<slot>/clients/<pid>-<start_ticks>.json`, mode `0600`, containing the complete root and wrapper identity. Remove a lease only when its recorded process no longer matches.

- [ ] **Step 3: Write failing readiness, startup-failure, and port-collision tests**

```python
def test_startup_failure_removes_only_attempt_state(fake_runtime):
    fake_runtime.backend_exits_before_ready = True
    with pytest.raises(module.RuntimeFailure, match="backend exited before readiness"):
        module.ensure_backend(fake_runtime.slot, fake_runtime.config)
    assert not fake_runtime.backend_metadata.exists()
    assert fake_runtime.unrelated_process_alive


def test_port_collision_is_detected_not_attached(fake_runtime):
    fake_runtime.bind_selected_port_with_unrelated_process()
    with pytest.raises(module.RuntimeFailure, match="port ownership"):
        module.ensure_backend(fake_runtime.slot, fake_runtime.config)
```

- [ ] **Step 4: Implement backend startup and readiness**

Start exact argv:

```text
/root/.local/bin/serena start-mcp-server
  --transport streamable-http --host 127.0.0.1 --port <slot-port>
  --project <exact-root> --context coordexp-minimal
  --enable-web-dashboard false --enable-gui-log-window false
  --open-web-dashboard false --log-level CRITICAL
```

Allocate a free loopback port while holding a global allocation lock, record it in the slot, start Serena in a new session with stdin `/dev/null`, and wait up to 60 seconds for both exact identity and TCP readiness. Preserve only a bounded rotating backend log.

- [ ] **Step 5: Write failing normal-exit, SIGKILL-recovery, and grace tests**

```python
def test_last_lease_reaps_only_after_grace(fake_runtime, fake_clock):
    lease = module.SlotLease.acquire(fake_runtime.slot)
    module.release_and_schedule_reap(lease, grace_seconds=60)
    fake_clock.advance(59)
    assert fake_runtime.backend_alive
    fake_clock.advance(1)
    assert not fake_runtime.backend_alive


def test_new_lease_cancels_pending_retirement(fake_runtime, fake_clock):
    first = module.SlotLease.acquire(fake_runtime.slot)
    module.release_and_schedule_reap(first, grace_seconds=60)
    second = module.SlotLease.acquire(fake_runtime.slot)
    fake_clock.advance(60)
    assert fake_runtime.backend_alive
    second.release()
```

- [ ] **Step 6: Implement exact-identity reaper and doctor**

The detached `reap` subprocess sleeps outside the lock, reacquires it, reconciles leases, revalidates PID/start-time/executable/project/port, sends SIGTERM to the exact process group, waits 10 seconds, and sends SIGKILL only if the same identity still exists. `doctor` performs reconciliation and prints bounded JSON status without starting a backend.

- [ ] **Step 7: Run deterministic lifecycle gates and commit**

Run the focused pytest file, Ruff, Ty using the `ms` interpreter, `git diff --check`, and a process census that asserts no fake test child remains. Commit only the two Task 2 paths with message `Manage shared Serena worktree lifecycles`.

---

### Task 3: Pinned transport bridge and stdio serve path

**Files:**
- Create: `.codex/serena/bridge.lock`
- Create: `.codex/serena/setup_shared_runtime.sh`
- Modify: `.codex/serena/serena_worktree_mcp.py`
- Modify: `tests/runtime/test_serena_worktree_mcp.py`

**Interfaces:**
- Consumes: `ensure_backend` and `SlotLease` from Task 2.
- Produces: service-owned executable `/data/CoordExp/.codex/serena/runtime/bridge/bin/mcp-proxy`.
- Produces: `serve()` which runs the bridge and returns its exact exit code.

- [ ] **Step 1: Pin bridge provenance and write failing setup verification**

The lock file contains exactly:

```text
mcp-proxy.git=153a96a61fde2bf5a23961c64a3dd96b5e385108
mcp-proxy.version=0.12.0
mcp.version=1.27.1
```

Test that setup rejects a missing or mismatched version and that `mcp-proxy --version` succeeds from the service-owned path.

- [ ] **Step 2: Implement idempotent bridge setup**

Create a Python 3.12 venv below `.codex/serena/runtime/bridge`, then install exact `mcp==1.27.1` and exact Git revision of `mcp-proxy` with hashes/provenance recorded alongside the environment. Setup may use ambient external proxy for download; runtime child environments may not.

- [ ] **Step 3: Write failing bridge-exit and proxy-scrubbing tests**

```python
def test_serve_returns_bridge_exit_and_releases_lease(fake_runtime):
    fake_runtime.bridge_exit = 17
    assert module.serve(fake_runtime.config) == 17
    assert fake_runtime.valid_leases == []


def test_local_children_receive_no_proxy_variables(fake_runtime, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9090")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9090")
    module.serve(fake_runtime.config)
    assert all(not key.upper().endswith("_PROXY") for key in fake_runtime.backend_env)
    assert all(not key.upper().endswith("_PROXY") for key in fake_runtime.bridge_env)
```

- [ ] **Step 4: Implement bridge execution**

Run:

```text
<pinned-mcp-proxy> --transport streamablehttp http://127.0.0.1:<port>/mcp
```

Inherit stdin/stdout directly, keep stderr bounded, forward SIGTERM/SIGINT, and release the lease in `finally`. Do not parse MCP payloads.

- [ ] **Step 5: Install, verify, and commit**

Run setup once, verify exact package versions, execute deterministic tests, then commit the lock, setup script, wrapper, and tests as `Pin Serena stdio HTTP bridge`.

---

### Task 4: Switch Codex and Claude Code client configuration safely

**Files:**
- Modify machine-local: `/data/CoordExp/.codex/config.toml`
- Modify tracked: `/data/CoordExp/.mcp.json`
- Modify tracked: `/data/CoordExp/serena-light/.mcp.json`
- Modify machine-local: `/data/CoordExp/.claude/.claude.json`
- Test: `tests/runtime/test_serena_client_config.py`

**Interfaces:**
- Consumes: wrapper CLI `serve` and the service-owned bridge from Task 3.
- Produces: all clients invoke `/usr/bin/setpriv --pdeathsig TERM /root/miniconda3/envs/ms/bin/python /data/CoordExp/.codex/serena/serena_worktree_mcp.py serve`.

- [ ] **Step 1: Write a failing config contract test**

Parse TOML/JSON and assert all four Serena registrations use the exact wrapper argv and preserve the existing `ms` environment. Assert the context file still exposes exactly the user-approved 15 fixed tools and that every existing `serena-hooks remind` command remains present.

- [ ] **Step 2: Update registrations without broad formatting**

Change only Serena `command` and `args`; preserve unrelated client settings. Do not add global `NO_PROXY`. Do not remove direct Serena rollback information from the design or setup documentation.

- [ ] **Step 3: Verify config parsing and hook behavior**

Run the focused config test, parse every JSON file, parse Codex TOML with Python 3.12 `tomllib`, and invoke `serena-hooks remind --client=codex` with a representative hook fixture to confirm blocking behavior is unchanged.

- [ ] **Step 4: Commit tracked configuration separately**

Commit only `/data/CoordExp/.mcp.json` and the config test in the root repository. Commit only `.mcp.json` in `/data/CoordExp/serena-light`. Report machine-local config changes separately; never stage unrelated dirty files.

---

### Task 5: Real concurrent acceptance, lifecycle cleanup, and handoff

**Files:**
- Create: `docs/history/official-serena-worktree-runtime-acceptance.md`
- Modify: `.codex/serena/serena_worktree_mcp.py`
- Modify: `tests/runtime/test_serena_worktree_mcp.py`
- Modify: `tests/runtime/test_serena_client_config.py`

**Interfaces:**
- Consumes the installed wrapper and all client configurations.
- Produces an evidence record with exact commands, roots, slot keys, process identities, semantic results, cleanup results, and rollback command.

- [ ] **Step 1: Establish before-census and preserve unrelated processes**

Record exact PID, PPID, start time, argv, and project root for existing Serena/Pyright processes. Do not terminate anything not owned by the new runtime slots.

- [ ] **Step 2: Same-worktree two-client acceptance**

Launch two fresh protocol clients concurrently from `CoordExp-swift`. Both must complete MCP initialize, `initial_instructions`, `activate_project` for the exact same root, `get_symbols_overview`, `find_symbol` for `TrainRuntime`, and file diagnostics. Assert one backend PID and one Python language-server process group for the slot.

- [ ] **Step 3: Cross-worktree acceptance**

Concurrently run one client in `CoordExp-swift` and one in `research-probes`. Resolve `TrainRuntime` and `PipelinePlanner` respectively. Assert distinct slot keys, backend identities, ports, and project roots.

- [ ] **Step 4: Crash and poisoned-proxy acceptance**

Start clients with poisoned upper/lower-case proxy variables, then SIGKILL one wrapper. Confirm the surviving client remains semantic-functional, the dead lease is reconciled, localhost traffic bypasses proxy, and no unrelated process changes.

- [ ] **Step 5: Retirement acceptance**

Close the final clients, wait 65 seconds, and verify exact backend identities have retired. Confirm no slot-owned Serena, Pyright, bridge, or reaper survives and that runtime metadata cannot authorize a reused PID.

- [ ] **Step 6: Fresh Codex and Claude Code acceptance**

Restart clients once after configuration reload. For each client, record the actual tool list, manual result, active project, symbol overview, symbol lookup, and diagnostics. A Codex app-server restart with two restored same-worktree tasks must not create duplicate backends.

- [ ] **Step 7: Convert every acceptance defect into a focused regression before correction**

For a lifecycle defect, add the smallest failing case to `test_serena_worktree_mcp.py`; for a registration or Hook defect, add it to `test_serena_client_config.py`. Run the exact new test to record RED, make the minimal wrapper/config correction, then rerun it to GREEN. If acceptance reveals no defect, leave these three files unchanged in Task 5.

- [ ] **Step 8: Document evidence and run final gates**

Write the acceptance record, run focused tests, Ruff, Ty, config parsers, `git diff --check`, and exact process census. If any live client witness is unavailable, report HOLD and do not claim completion.

- [ ] **Step 9: Commit and publish only after live PASS**

Commit the evidence and any test-driven corrections with explicit paths. Reinspect all repositories, fetch and compare upstream state, then push only repositories whose scoped commits are current and authorized. Leave unrelated dirty changes untouched.
