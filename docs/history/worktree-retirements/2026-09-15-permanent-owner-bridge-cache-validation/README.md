# Permanent-owner bridge cache-validation retirement capture

This is a frozen source capture for the clean, detached
`permanent-owner-bridge-cache-validation` worktree at commit
`4f6fca72e91ae8b752a2a20edb69f772b3f69196`. It is not a current launch
surface, a production claim, or authority to resume the historical Stage1
program.

The capture makes retirement possible without treating a clean worktree as a
claim that its research was absorbed. The [manifest](manifest.json) binds each
materialized source path to its Git blob, SHA-256, and byte count. It records
the complete document comparison against `research-probes` at capture:

- 1,682 source Markdown files were inspected;
- 1,382 already had byte-identical content in `research-probes`;
- all remaining 300 Markdown files are copied verbatim under `sources/`;
- the 31 permanent-owner-bridge OpenSpec, results, and machine-readable
  receipt files are also materialized, including files that happened to have
  an identical Markdown peer elsewhere.

The annotated Git tag
`archive/permanent-owner-bridge-cache-validation-20260915` retains the entire
source tree, including code and configurations outside this document capture.
The worktree's ignored set at retirement contained only local test, lint, and
Python caches; no ignored run-output root was present to transfer.

Read [synthesis.md](synthesis.md) for the narrow current interpretation. Read
the source files when a historical protocol, receipt, or exact statement is
needed. Paths under `sources/` retain their original logical coordinates, so
their historical relative links are not promised to resolve at this archive
depth.
