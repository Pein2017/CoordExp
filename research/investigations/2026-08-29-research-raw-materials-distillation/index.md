---
title: Research output raw-material distillation (2026-08-29)
description: Session-backed Markdown synthesis and an exact deletion boundary for disposable research outputs.
type: investigation
role: archival-synthesis
authority: non_normative_research
status: active
updated: 2026-08-29
---

# Research output raw-material distillation

This packet is an importable synthesis layer. It is **not** the semantic owner
of any experiment: the linked `unit.md`/`results.md` records in the research
worktrees remain authoritative for scientific meaning, and
`outputs/research/` remains the execution-artifact namespace.
The native Validation-200 benchmark had no separate worktree copy; its
decision-bearing Markdown is preserved here before deleting its raw root.

## Decision

Delete non-Image2299 raw outputs after this packet and the deletion manifest
are written. Keep only:

1. roots explicitly bound to or produced by the Image2299 worktree;
2. roots with live process/file handles at deletion time;
3. the eight-coordinate closeout because Image2299 explicitly consumes its
   `four-coordinate-xy/checkpoints/step-2444` checkpoint (the whole small root
   is retained this pass to avoid splitting a provenance bundle); and
4. canonical Markdown records, which live outside `outputs/research/`.

The deletion manifest is generated from the live tree immediately before the
mutation. A path not listed there is not deleted.

## Reading path

- [Qwen3-VL dense enumeration synthesis](qwen3-vl-dense-enumeration.md)
- [PVCI causal proposal bridge synthesis](pvci-causal-proposal-bridge.md)
- [Native text-coordinate validation synthesis](native-text-coordinate-val200.md)
- [Pi lightweight-worker ablation synthesis](pi-lightweight-worker-ablation.md)
- [Eight-coordinate supervision synthesis](eight-coordinate-bbox-supervision.md)
- [Session provenance and limits](session-provenance.md)
- [Exact retention/deletion manifest](retention-and-deletion-manifest.md)

## Status vocabulary

- **KEEP-IMAGE2299**: user-exempt, current Image2299 line or explicitly bound
  input.
- **HOLD-ACTIVE**: live process/file handle; revisit only after the run closes.
- **KEEP-DEPENDENCY**: non-Image2299 root still supplies an explicit current
  input or frozen source checkpoint.
- **DISTILL-DELETE**: result is represented by canonical records plus this
  packet; raw execution material is disposable.
- **UNKNOWN**: no exact per-file authorship or checksum was recovered; this is
  recorded as a limitation, not silently promoted to a scientific claim.
