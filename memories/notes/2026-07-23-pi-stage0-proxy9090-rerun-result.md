# Pi Stage 0 Proxy-9090 Rerun Result

Captured: `2026-07-23T01:35:06Z`.

The user specified that networked processes require the local port-9090 proxy
and authorized reopening the Pi evaluation. Worktree-local
`.pi-worker/home/.bashrc` now exports uppercase and lowercase HTTP, HTTPS, and
all-proxy variables to `http://127.0.0.1:9090`; `.bash_profile`, `pi-local`, and
the Stage 0 runner consume this initialization. All home, auth, session,
sandbox, and result state remains below `.pi-worker/`.

A zero-model chroot preflight returned the expected unauthenticated `401` from
the OpenAI endpoint. The twelve Pi infrastructure-retry cells then completed
between `2026-07-23T01:29:01Z` and `2026-07-23T01:35:06Z` with positive token
usage and non-error terminal events. No cell used `max`; the frozen rotation of
`medium`, `high`, and `xhigh` remained unchanged. No graphics-processing unit
was used.

All three models pass artifact inventory and mechanical aggregation. An
additive audited Task 2 verifier passes Luna and Terra. Pi Sol repeats native
Sol's exact `HFBackendSession` capitalization error, arguing against a
Pi-specific cause. All three Task 3 answers contain the correct verdict,
qualifying image identifiers, and complementarity argument, but the original
lexical limitations verifier rejects them. Luna explicitly bounds population,
training, and architecture claims; Sol and Terra use narrower wording. Keep
both strict and semantic interpretations visible.

Every disposable workspace is byte-for-byte unchanged. Recorded tool calls
show no network command, parent traversal, host `/data/` access, or Git
command. Pi exposes token, provider-cost, wall-time, and tool-call receipts,
but native Codex has no comparable token or cost receipt, so no complete cost
advantage is established.

Authoritative handles:

- `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/results.md`;
- `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/pi-proxy9090-rerun1-receipt.md`.

Stage 0 is complete with bounded verified evidence. It supports a larger
frozen benchmark for mechanically verifiable delegation, not an adapter or
default worker-route promotion.
