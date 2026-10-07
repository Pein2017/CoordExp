## 1. Source and runtime ownership

- [x] 1.1 Prepare an unmodified official Pi source reference at `codex-tools/pi-core` pinned to v1.0.3; verify tag, origin and clean tracked source.
- [x] 1.2 Change CLI/Web callers and release validation to use the canonical install root explicitly; verify binder fixtures, shell syntax and resolved package identities.

## 2. Safe Web update adoption

- [x] 2.1 Add the version observer and idle/admission boundary, covering session starts, commands, SDK busy/queues and terminal shells; verify RED/GREEN at the retained consumers and failed activation recovery.
- [x] 2.2 Integrate observer with the local Next dev worker; verify one observer across HMR and real isolated Next restart/import/persisted-history smoke with test-owned releases.

## 3. Local cutover and navigation

- [x] 3.1 Reconcile idle live runtime, gracefully stop its owned service, move the existing managed install/launcher, and start canonical Web; verify CLI version, shared SDK physical path, profile/local extension preservation and HTTP consumers with no legacy runtime fallback.
- [x] 3.2 Initialize independent CodeGraph projects for active codex-tools repositories, remove the new parent index/config, and verify core compaction/Web session symbols plus runtime/dependency exclusion at the actual repository owners.
- [x] 3.3 Update owner documentation with paths, update semantics, source refresh and validation limits; verify diffs, targeted checks, OpenSpec validation and retained receipts.
